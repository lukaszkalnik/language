"""Model download helpers: announce the size on first fetch, pick a torch device."""

from __future__ import annotations

import fnmatch
import logging
import os
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

from .errors import ModelError


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB", "MB") else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


def _status(message: str) -> None:
    import sys

    print(message, file=sys.stderr, flush=True)


def hf_is_cached(repo_id: str, filename: str | None = None, revision: str | None = None,
                 cache_dir: str | None = None) -> bool:
    """True if `filename` (or, without one, any snapshot of the repo) is in the local HF cache."""
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    if filename and isinstance(try_to_load_from_cache(repo_id, filename, revision=revision, cache_dir=cache_dir), str):
        return True
    try:
        snapshot_download(repo_id, revision=revision, cache_dir=cache_dir, local_files_only=True)
        return True
    except Exception:
        return False


@contextmanager
def hub_offline() -> Iterator[None]:
    """Hugging Face Hub calls in this block use the local cache only and open no connection.

    Sets huggingface_hub's offline flag (transformers reads it too) for the duration, then
    restores it.
    """
    from huggingface_hub import constants

    before = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = before


T = TypeVar("T")


def load_cached_first(load: Callable[[], T], cached: bool) -> T:
    """`load()` without asking the Hub when the model is `cached`.

    If that fails (a file missing after an interrupted download), or the model is not cached,
    load normally, which may download.
    """
    if cached:
        try:
            with hub_offline():
                result = load()
            if result is not None:
                return result
        except Exception:
            pass
    return load()


def hf_local_path(repo_id: str, revision: str | None = None) -> str:
    """The local snapshot folder of a cached repo, else the repo ID.

    Libraries given a repo ID ask the Hub for its latest revision even when the files are
    cached; on a network with broken IPv6 that request hangs for minutes. A local path skips it.
    """
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id, revision=revision, local_files_only=True)
    except Exception:
        return repo_id


def hf_repo_size(repo_id: str, patterns: list[str] | None = None, revision: str | None = None) -> int | None:
    """Total size of the files a download will fetch (None if offline or unknown)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    except Exception:
        return None
    total = 0
    for sibling in info.siblings or []:
        if patterns and not any(fnmatch.fnmatch(sibling.rfilename, p) for p in patterns):
            continue
        total += sibling.size or 0
    return total


def announce_hf_download(repo_id: str, probe_file: str, patterns: list[str] | None = None, what: str | None = None,
                         revision: str | None = None) -> None:
    """Print the download size if `repo_id` is not in the local Hugging Face cache yet."""
    if hf_is_cached(repo_id, probe_file, revision):
        return
    size = hf_repo_size(repo_id, patterns, revision)
    label = what or repo_id
    _status(f"Downloading {label} ({human_size(size) if size else 'size unknown'}), first run only...")


def hf_snapshot(repo_id: str, patterns: list[str] | None = None, probe_file: str = "config.json", what: str | None = None,
                revision: str | None = None) -> str:
    """Download (or reuse) a Hugging Face repo snapshot, printing the size on first fetch."""
    from huggingface_hub import snapshot_download

    announce_hf_download(repo_id, probe_file, patterns, what, revision)
    if revision is not None and (local := hf_local_path(repo_id, revision)) != repo_id:
        return local  # a pinned revision never changes: no need to ask the Hub
    try:
        return snapshot_download(repo_id, revision=revision, allow_patterns=patterns)
    except Exception as exc:
        raise ModelError(f"could not download {repo_id} ({exc.__class__.__name__}: {exc}). Check your network connection.") from None


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise ModelError("--device cuda was requested but CUDA is not available.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ModelError("--device mps was requested but Apple MPS is not available.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def quiet_libraries(*names: str) -> None:
    """Silence noisy library loggers and warnings so stderr stays readable."""
    os.environ.setdefault("SB_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore")
    for name in ("speechbrain", "transformers", "huggingface_hub", "torch", "pyannote", "lightning", *names):
        logging.getLogger(name).setLevel(logging.ERROR)
