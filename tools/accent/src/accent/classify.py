"""Load the CommonAccent SpeechBrain models and turn audio chunks into accent probabilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from ingest.cache import cache_dir
from ingest.errors import ModelError
from ingest.models import (
    hf_is_cached,
    hf_repo_size,
    human_size,
    load_cached_first,
    pick_device,
    quiet_libraries,
)

# As published on the Hugging Face Hub (model cards checked 2026-11-18).
ECAPA_SOURCE = "Jzuluaga/accent-id-commonaccent_ecapa"
XLSR_SOURCE = "Jzuluaga/accent-id-commonaccent_xlsr-en-english"
XLSR_BASE = "facebook/wav2vec2-large-xlsr-53"  # fetched by the XLSR model's hyperparams.yaml

# The ECAPA head returns cosine similarities, not probabilities. It was trained with an
# additive angular margin softmax of scale 30, so softmax(30 * cosine) recovers its posterior.
ECAPA_SCALE = 30.0

DISPLAY_NAMES = {
    "african": "African", "australia": "Australian", "bermuda": "Bermudian", "canada": "Canadian",
    "england": "English (England)", "hongkong": "Hong Kong", "indian": "Indian", "ireland": "Irish",
    "malaysia": "Malaysian", "newzealand": "New Zealand", "philippines": "Filipino", "scotland": "Scottish",
    "singapore": "Singaporean", "southatlandtic": "South Atlantic",  # sic: the models' label is misspelled
    "us": "American (US)", "wales": "Welsh",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str
    description: str


MODELS = {
    "ecapa": ModelSpec("ecapa", ECAPA_SOURCE, "ECAPA-TDNN (~85 MB, fast)"),
    "xlsr": ModelSpec("xlsr", XLSR_SOURCE, "wav2vec2 XLSR-53 (~2.5 GB with its base model, slower)"),
}


def display_name(label: str) -> str:
    return DISPLAY_NAMES.get(label, label)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def average_probabilities(per_chunk: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Average chunk posteriors, weighted (by speech time) so a short chunk counts less."""
    if not per_chunk:
        raise ValueError("no chunks to average")
    stacked = np.stack(per_chunk)
    w = np.ones(len(per_chunk)) if weights is None else np.asarray(weights, dtype=np.float64)
    return (stacked * w[:, None]).sum(axis=0) / w.sum()


def top_k(probs: np.ndarray, labels: list[str], k: int = 3) -> list[tuple[str, float]]:
    order = np.argsort(probs)[::-1][:k]
    return [(labels[i], float(probs[i])) for i in order]


def _local_strategy():
    from speechbrain.utils.fetching import LocalStrategy

    # Symlinks need admin rights (or developer mode) on Windows: copy there instead.
    return LocalStrategy.COPY if sys.platform == "win32" else LocalStrategy.SYMLINK


class AccentClassifier:
    def __init__(self, model: str = "ecapa", device: str = "auto"):
        if model not in MODELS:
            raise ModelError(f"unknown model {model!r}; choose from {', '.join(MODELS)}")
        self.spec = MODELS[model]
        self.device = pick_device(device)
        self._model = None
        self.labels: list[str] = []

    def _base_dir(self):
        """Where the XLSR model's yaml keeps its transformers base model (a HF cache layout)."""
        return cache_dir("models", "accent", "wav2vec2_checkpoints")

    def is_cached(self, savedir) -> bool:
        """True if every file the model loads is on disk, so loading needs no network."""
        if not (savedir / "hyperparams.yaml").exists():
            return False
        return self.spec.name != "xlsr" or hf_is_cached(XLSR_BASE, "config.json", cache_dir=str(self._base_dir()))

    def _announce_download(self, savedir) -> None:
        size = 0
        if not (savedir / "hyperparams.yaml").exists():
            size += hf_repo_size(self.spec.source, ["*.ckpt", "*.yaml", "*.txt", "*.py"]) or 0
        if self.spec.name == "xlsr" and not hf_is_cached(XLSR_BASE, "config.json", cache_dir=str(self._base_dir())):
            size += hf_repo_size(XLSR_BASE, ["*.bin", "*.json"]) or 0
        print(f"Downloading {self.spec.name} model ({human_size(size) if size else 'size unknown'}), first run only...",
              file=sys.stderr, flush=True)

    def _build(self, savedir):
        run_opts = {"device": self.device}
        if self.spec.name == "ecapa":
            from speechbrain.inference.classifiers import EncoderClassifier

            return EncoderClassifier.from_hparams(
                source=self.spec.source, savedir=str(savedir), run_opts=run_opts,
                local_strategy=_local_strategy(),
            )
        from speechbrain.inference.interfaces import foreign_class

        return foreign_class(
            source=self.spec.source,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=str(savedir),
            run_opts=run_opts,
            local_strategy=_local_strategy(),
            # The yaml would otherwise create ./wav2vec2_checkpoints in the current directory.
            overrides={"wav2vec2": {"save_path": str(self._base_dir())}},
        )

    def load(self) -> None:
        quiet_libraries()
        savedir = cache_dir("models", "accent", self.spec.name)
        cached = self.is_cached(savedir)
        if not cached:
            self._announce_download(savedir)
        try:
            # Files in savedir never go to the Hub, but the XLSR base model is loaded by
            # transformers, which asks the Hub for the latest revision unless it is offline.
            self._model = load_cached_first(lambda: self._build(savedir), cached)
        except OSError as exc:
            raise ModelError(
                f"could not download or load the {self.spec.name} model ({exc.__class__.__name__}: {exc}). "
                "Check your network connection (or try --ipv4)."
            ) from None
        encoder = self._model.hparams.label_encoder
        self.labels = [encoder.ind2lab[i] for i in range(len(encoder.ind2lab))]

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """Probabilities over self.labels for one 16 kHz mono chunk."""
        import torch

        if self._model is None:
            self.load()
        wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            out_prob, _score, _index, _label = self._model.classify_batch(wav)
        scores = out_prob.detach().float().cpu().numpy().reshape(-1)
        if self.spec.name == "ecapa":
            return softmax(ECAPA_SCALE * scores)
        return scores / scores.sum()  # XLSR's head already applies softmax
