"""Whisper backends: MLX on Apple Silicon, faster-whisper (CTranslate2) everywhere else."""

from __future__ import annotations

import platform
import sys

import numpy as np

from ingest.cli import progress_wanted, report_progress
from ingest.errors import ModelError
from ingest.models import announce_hf_download, hf_local_path, quiet_libraries

from .segments import Segment, Word

SR = 16_000
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

MLX_MODELS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}
FASTER_MODELS = {"large-v3": "large-v3", "turbo": "large-v3-turbo"}
FASTER_REPOS = {"large-v3": "Systran/faster-whisper-large-v3", "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"}
SEED = 0  # temperature fallback samples at random; a fixed seed makes runs repeatable (MLX)

# Decoding settings shared by both backends. condition_on_previous_text=False stops a
# hallucination in one window from being copied into the next.
DECODE = dict(
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8),
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
    condition_on_previous_text=False,
    word_timestamps=True,
)


def _segment(raw: dict, offset: float, language: str | None) -> Segment:
    words = [
        Word(offset + float(w["start"]), offset + float(w["end"]), w["word"], float(w.get("probability", 1.0)))
        for w in raw.get("words") or []
    ]
    return Segment(
        start=offset + float(raw["start"]), end=offset + float(raw["end"]), text=raw["text"].strip(),
        language=language, avg_logprob=float(raw.get("avg_logprob", 0.0)),
        no_speech_prob=float(raw.get("no_speech_prob", 0.0)),
        compression_ratio=float(raw.get("compression_ratio", 1.0)), words=words,
    )


class MLXWhisper:
    name = "mlx-whisper"

    def __init__(self, model: str):
        quiet_libraries("mlx")
        self.repo = MLX_MODELS[model]
        probe = "weights.npz" if model == "large-v3" else "weights.safetensors"
        announce_hf_download(self.repo, probe, what=f"Whisper {model} (MLX)")
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        from . import mlx_fixes

        mlx_fixes.apply()

        try:
            self.path = hf_local_path(self.repo)  # transcribe() must pass the same key to reuse it
            self.model = ModelHolder.get_model(self.path, mx.float16)
        except Exception as exc:
            raise ModelError(f"could not load {self.repo} ({exc.__class__.__name__}: {exc})") from None

    def detect(self, audio: np.ndarray) -> dict[str, float]:
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import detect_language

        mel = log_mel_spectrogram(audio.astype(np.float32), n_mels=self.model.dims.n_mels)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = detect_language(self.model, mel)
        return dict(probs)

    def transcribe(self, audio: np.ndarray, language: str | None, offset: float = 0.0,
                   clips: list[tuple[float, float]] | None = None, prompt: str | None = None,
                   show_progress: bool = False) -> tuple[list[Segment], str | None]:
        import mlx.core as mx
        import mlx_whisper

        mx.random.seed(SEED)
        clip = "0" if not clips else [t for pair in clips for t in pair]
        result = mlx_whisper.transcribe(
            audio.astype(np.float32), path_or_hf_repo=self.path, language=language,
            # verbose=False draws mlx-whisper's tqdm bar, which the desktop window reads as progress
            verbose=False if show_progress and progress_wanted() else None,
            clip_timestamps=clip, initial_prompt=prompt, **DECODE,
        )
        lang = result.get("language") or language
        return [_segment(s, offset, lang) for s in result.get("segments", [])], lang


class FasterWhisper:
    name = "faster-whisper"

    def __init__(self, model: str, device: str = "auto"):
        quiet_libraries("faster_whisper")
        announce_hf_download(FASTER_REPOS[model], "model.bin", what=f"Whisper {model} (CTranslate2)")
        from faster_whisper import WhisperModel

        compute = "float16" if device == "cuda" else "int8"
        try:
            local = hf_local_path(FASTER_REPOS[model])
            source = local if local != FASTER_REPOS[model] else FASTER_MODELS[model]
            self.model = WhisperModel(source, device="cuda" if device == "cuda" else "cpu", compute_type=compute)
        except Exception as exc:
            raise ModelError(f"could not load Whisper {model} ({exc.__class__.__name__}: {exc})") from None

    def detect(self, audio: np.ndarray) -> dict[str, float]:
        _, _, all_probs = self.model.detect_language(audio.astype(np.float32))
        return dict(all_probs)

    def transcribe(self, audio: np.ndarray, language: str | None, offset: float = 0.0,
                   clips: list[tuple[float, float]] | None = None, prompt: str | None = None,
                   show_progress: bool = False) -> tuple[list[Segment], str | None]:
        opts = dict(DECODE)
        opts["log_prob_threshold"] = opts.pop("logprob_threshold")
        segments, info = self.model.transcribe(
            audio.astype(np.float32), language=language, vad_filter=False, initial_prompt=prompt,
            clip_timestamps=[t for pair in clips for t in pair] if clips else "0", **opts,
        )
        lang = info.language or language
        out = []
        duration = len(audio) / 16000
        for s in segments:
            if show_progress and duration:
                report_progress("Transcribing", s.end / duration)
            raw = {"start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob,
                   "no_speech_prob": s.no_speech_prob, "compression_ratio": s.compression_ratio,
                   "words": [{"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                             for w in (s.words or [])]}
            out.append(_segment(raw, offset, lang))
        return out, lang


def load(model: str = "large-v3", device: str = "auto"):
    if model not in MLX_MODELS:
        raise ValueError(f"unknown Whisper model {model!r}; choose from {', '.join(MLX_MODELS)}")
    if APPLE_SILICON and device in ("auto", "mps"):
        return MLXWhisper(model)
    return FasterWhisper(model, device)
