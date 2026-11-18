"""Speaker diarization: pyannote when a Hugging Face token gives access, else SpeechBrain ECAPA.

pyannote/speaker-diarization-community-1 (pyannote.audio 4.x) is the best local option, but its
weights are gated: it needs a free Hugging Face account, accepting the model's conditions once,
and a token. Without that, transcribe explains the one-time setup and falls back to clustering
SpeechBrain ECAPA speaker embeddings (speechbrain/spkrec-ecapa-voxceleb, not gated).
"""

from __future__ import annotations

import sys

import numpy as np

from ingest.cache import cache_dir
from ingest.models import announce_hf_download, pick_device, quiet_libraries

PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SR = 16_000

SETUP = f"""\
Speaker labels use {PYANNOTE_MODEL}, which needs a one-time setup:
  1. Create a free account at https://huggingface.co and accept the conditions on
     https://huggingface.co/{PYANNOTE_MODEL}
  2. Create a read token at https://huggingface.co/settings/tokens
  3. Run `hf auth login` (or set HF_TOKEN=...) once.
Until then a simpler built-in method (SpeechBrain ECAPA embeddings + clustering) is used."""


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def hf_token() -> str | None:
    import os

    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


# --- pyannote ---------------------------------------------------------------------

def pyannote_turns(wav_path, num_speakers: int | None, device: str) -> list[tuple[float, float, str]] | None:
    """Turns from pyannote, or None (after explaining the setup) if the model is not accessible."""
    token = hf_token()
    if not token:
        _status(SETUP)
        return None
    quiet_libraries("pyannote", "speechbrain")
    try:
        from pyannote.audio import Pipeline
    except Exception as exc:  # pragma: no cover - broken install
        _status(f"pyannote.audio could not be imported ({exc}); using the built-in method.")
        return None
    try:
        announce_hf_download(PYANNOTE_MODEL, "config.yaml", what="pyannote diarization model")
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, token=token)
    except Exception as exc:
        text = str(exc).lower()
        if any(k in text for k in ("401", "403", "gated", "access", "authenticat", "accept")):
            _status(f"No access to {PYANNOTE_MODEL} with this token.\n{SETUP}")
        else:
            _status(f"pyannote model could not be loaded ({exc.__class__.__name__}); using the built-in method.")
        return None
    if pipeline is None:
        _status(SETUP)
        return None
    import torch

    dev = pick_device(device)
    try:
        pipeline.to(torch.device(dev))
    except Exception:
        dev = "cpu"
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    _status(f"Diarizing with pyannote on {dev}...")
    output = pipeline(str(wav_path), **kwargs)
    annotation = getattr(output, "speaker_diarization", output)
    return [(float(t.start), float(t.end), str(spk)) for t, _, spk in annotation.itertracks(yield_label=True)]


# --- ECAPA fallback -------------------------------------------------------------------

WINDOW = 2.0
STEP = 1.0
DISTANCE = 0.55  # cosine distance at which clusters stop merging (when the count is unknown)
MIN_SHARE = 0.04  # clusters with less speaking time than this are merged into their nearest


def embedding_windows(regions: list[tuple[float, float]], window: float = WINDOW, step: float = STEP) -> list[tuple[float, float]]:
    out = []
    for a, b in regions:
        if b - a < 0.8:
            continue
        if b - a <= window:
            out.append((a, b))
            continue
        t = a
        while t + window <= b + 1e-6:
            out.append((t, t + window))
            t += step
        if out[-1][1] < b - 0.5:
            out.append((b - window, b))
    return out


def cluster(embeddings: np.ndarray, durations: np.ndarray, num_speakers: int | None = None,
            distance: float = DISTANCE) -> np.ndarray:
    """Agglomerative clustering (average linkage, cosine). Returns labels 0..k-1."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1 or num_speakers == 1:
        return np.zeros(n, dtype=int)
    x = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
    tree = linkage(x, method="average", metric="cosine")
    if num_speakers:
        labels = fcluster(tree, t=num_speakers, criterion="maxclust") - 1
    else:
        labels = fcluster(tree, t=distance, criterion="distance") - 1
        # Merge clusters that hold too little speech into the closest big one.
        total = durations.sum()
        while True:
            ids = np.unique(labels)
            if len(ids) <= 1:
                break
            shares = {i: durations[labels == i].sum() / total for i in ids}
            small = min(shares, key=shares.get)
            if shares[small] >= MIN_SHARE:
                break
            centroids = {i: x[labels == i].mean(axis=0) for i in ids}
            others = [i for i in ids if i != small]
            nearest = max(others, key=lambda i: float(np.dot(centroids[i], centroids[small])))
            labels[labels == small] = nearest
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def windows_to_turns(windows: list[tuple[float, float]], labels: np.ndarray) -> list[tuple[float, float, str]]:
    """Overlapping labelled windows -> non-overlapping turns (each instant goes to the window centred nearest)."""
    turns: list[list] = []
    for i, ((a, b), lab) in enumerate(zip(windows, labels, strict=True)):
        start = a if i == 0 or windows[i - 1][1] <= a else (a + windows[i - 1][1]) / 2
        end = b if i + 1 == len(windows) or windows[i + 1][0] >= b else (b + windows[i + 1][0]) / 2
        spk = f"S{int(lab)}"
        if turns and turns[-1][2] == spk and start - turns[-1][1] < 0.5:
            turns[-1][1] = end
        else:
            turns.append([start, end, spk])
    return [(a, b, s) for a, b, s in turns]


def ecapa_turns(reader, regions: list[tuple[float, float]], num_speakers: int | None, device: str,
                progress=None) -> list[tuple[float, float, str]]:
    quiet_libraries("speechbrain")
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    savedir = cache_dir("models", "transcribe", "spkrec-ecapa-voxceleb")
    if not (savedir / "hyperparams.yaml").exists():
        announce_hf_download(ECAPA_MODEL, "embedding_model.ckpt", what="SpeechBrain ECAPA speaker model")
    dev = pick_device(device)
    model = EncoderClassifier.from_hparams(
        source=ECAPA_MODEL, savedir=str(savedir), run_opts={"device": dev},
        local_strategy=LocalStrategy.COPY if sys.platform == "win32" else LocalStrategy.SYMLINK,
    )
    windows = embedding_windows(regions)
    embeddings = []
    batch: list[np.ndarray] = []
    lengths: list[int] = []

    def flush():
        if not batch:
            return
        longest = max(len(b) for b in batch)
        wav = torch.zeros(len(batch), longest)
        for i, b in enumerate(batch):
            wav[i, : len(b)] = torch.from_numpy(b)
        rel = torch.tensor([n / longest for n in lengths])
        with torch.inference_mode():
            emb = model.encode_batch(wav, rel).squeeze(1).float().cpu().numpy()
        embeddings.append(emb)
        batch.clear()
        lengths.clear()

    for i, (a, b) in enumerate(windows):
        piece = reader.read(a, b)
        batch.append(np.ascontiguousarray(piece, dtype=np.float32))
        lengths.append(len(piece))
        if len(batch) == 64:
            flush()
            if progress:
                progress(i + 1, len(windows))
    flush()
    if not embeddings:
        return []
    emb = np.concatenate(embeddings)
    labels = cluster(emb, np.array([b - a for a, b in windows]), num_speakers)
    return windows_to_turns(windows, labels)
