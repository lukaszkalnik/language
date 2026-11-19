"""Demucs source separation with results kept in the shared cache."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ingest import audio
from ingest.cache import cache_dir
from ingest.cli import report_progress
from ingest.errors import ModelError
from ingest.models import announce_hf_download, pick_device, quiet_libraries
from ingest.source import Source


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_repo: str  # where Demucs 4.1 fetches the weights from
    stems: tuple[str, ...]
    note: str


FOUR = ("drums", "bass", "other", "vocals")
MODELS = {
    "htdemucs_ft": ModelSpec("htdemucs_ft", "adefossez/HTDemucs-ft", FOUR,
                             "Hybrid Transformer Demucs, fine-tuned per stem: best quality (default, 336 MB)"),
    "htdemucs": ModelSpec("htdemucs", "adefossez/HTDemucs", FOUR, "Hybrid Transformer Demucs: ~3x faster (84 MB)"),
    "htdemucs_6s": ModelSpec("htdemucs_6s", "adefossez/HTDemucs-6s", (*FOUR, "guitar", "piano"),
                             "six stems incl. guitar and piano (55 MB)"),
    "mdx_extra": ModelSpec("mdx_extra", "adefossez/Demucs-mdx_extra", FOUR, "older hybrid Demucs (669 MB)"),
}
DEFAULT_MODEL = "htdemucs_ft"
# Derived stems: everything except the vocals.
DERIVED = {"instrumental": "no_vocals", "no_vocals": "no_vocals", "accompaniment": "no_vocals"}


def stem_choices(model: str) -> list[str]:
    return [*MODELS[model].stems, "instrumental"]


def normalise_stem(stem: str, model: str) -> str:
    stem = stem.lower()
    if stem in DERIVED:
        return DERIVED[stem]
    if stem not in MODELS[model].stems:
        raise ValueError(f"model {model} has no stem {stem!r}; choose from {', '.join(stem_choices(model))}")
    return stem


def cache_tag(model: str, shifts: int = 1) -> str:
    """Settings that change the result are part of the cache key."""
    return model if shifts == 1 else f"{model}-shifts{shifts}"


def stem_path(src: Source, tag: str, stem: str) -> Path:
    return cache_dir("derived", "vocals") / f"{src.key(tag, stem)}.wav"


def output_name(src: Source, stem: str) -> str:
    """Default file name when stdout is a terminal: '<input name>.<stem>.wav'."""
    base = src.path.stem if src.kind == "file" else (src.label or src.source_id)
    if src.kind == "youtube" and src.info is not None:
        base = f"{src.info.title}"
    base = re.sub(r"[^\w\-. ]+", "_", base).strip(" ._")[:80] or "audio"
    label = "instrumental" if stem == "no_vocals" else stem
    return f"{base}.{label}.wav"


class Progress:
    """Demucs callback -> one updating line on stderr."""

    def __init__(self):
        self.last = -1

    def __call__(self, data: dict) -> None:
        if data.get("state") != "end":
            return
        length = data.get("audio_length") or 0
        done = data.get("segment_offset", 0) + (data.get("segment_length") or 0)
        model_idx, models = data.get("model_idx_in_bag", 0), max(1, data.get("models", 1))
        frac = min(1.0, (model_idx + (done / length if length else 0)) / models)
        pct = int(frac * 100)
        report_progress("Separating vocals", frac)
        if pct != self.last and sys.stderr.isatty():
            print(f"\rSeparating... {pct:3d}%", end="", file=sys.stderr, flush=True)
        self.last = pct


def separate(src: Source, model: str, device: str = "auto", use_cache: bool = True, shifts: int = 1) -> dict[str, Path]:
    """Separate src into all stems of `model` (plus no_vocals) at the source's sample rate.

    All stems are cached, so asking for another stem later is instant.
    """
    spec = MODELS[model]
    wanted = [*spec.stems, "no_vocals"]
    tag = cache_tag(model, shifts)
    paths = {s: stem_path(src, tag, s) for s in wanted}
    if use_cache and all(p.exists() and p.stat().st_size > 44 for p in paths.values()):
        if src.timeline_start and not audio.wav_origin(paths["vocals"]):
            for p in paths.values():  # cached before stems carried their place in the recording
                audio.add_wav_tags(p, {audio.ORIGIN_TAG: f"{src.timeline_start:.3f}"})
        return paths

    quiet_libraries("demucs")
    import torch
    from demucs.api import Separator

    info = audio.probe(src.path)
    source_rate = info.sample_rate
    channels = min(info.channels, 2)

    dev = pick_device(device)
    announce_hf_download(spec.hf_repo, "README.md", what=f"Demucs {model} model")
    try:
        separator = Separator(model=model, device=dev, shifts=shifts, progress=False, callback=Progress())
    except Exception as exc:
        raise ModelError(f"could not load Demucs model {model} ({exc.__class__.__name__}: {exc})") from None

    mix = audio.decode(src.path, separator.samplerate, 2, src.trim_start, src.trim_end)
    if mix.size == 0:
        raise ValueError("no audio in the selected range")
    wav = torch.from_numpy(np.ascontiguousarray(mix.T))
    print(f"Separating {mix.shape[0] / separator.samplerate:.0f} s with {model} on {dev}...", file=sys.stderr, flush=True)
    try:
        _, stems = separator.separate_tensor(wav, separator.samplerate)
    except RuntimeError as exc:
        if dev == "mps" and "mps" in str(exc).lower():
            print("\nMPS failed; retrying on CPU...", file=sys.stderr)
            separator.update_parameter(device="cpu")
            _, stems = separator.separate_tensor(wav, separator.samplerate)
        else:
            raise
    if sys.stderr.isatty():
        print("\r" + " " * 24 + "\r", end="", file=sys.stderr)

    arrays = {name: t.detach().cpu().numpy().T for name, t in stems.items()}
    arrays["no_vocals"] = sum(v for k, v in arrays.items() if k != "vocals")
    tmp_dir = cache_dir("derived", "vocals", "tmp")
    for name, data in arrays.items():
        if channels == 1:
            data = data.mean(axis=1)
        tmp = tmp_dir / f"{paths[name].stem}.model.wav"
        audio.write_wav(tmp, data, separator.samplerate)
        final_tmp = tmp_dir / f"{paths[name].stem}.wav"
        if source_rate != separator.samplerate:
            audio.to_wav(tmp, final_tmp, sample_rate=source_rate)  # back to the source's rate
            tmp.unlink()
        else:
            tmp.replace(final_tmp)
        title = src.info.title if src.info is not None else src.label
        tags = {"ISFT": f"vocals/{model}/{name}", "INAM": title or "", "ISRC": src.source_id}
        if src.timeline_start:
            tags[audio.ORIGIN_TAG] = f"{src.timeline_start:.3f}"
        audio.add_wav_tags(final_tmp, tags)
        final_tmp.replace(paths[name])
    return paths
