"""One cache folder shared by every tool.

Layout (under cache_root()):
    downloads/   YouTube audio, keyed by video ID + time range
    inputs/      WAV streams read from stdin, keyed by content hash
    derived/     audio or results computed by a tool (e.g. separated vocals),
                 keyed by source ID + time range + tool settings
    models/      model weights that are not kept in the Hugging Face cache

The location is ~/.cache/language by default (XDG_CACHE_HOME is respected;
%LOCALAPPDATA%\\language\\cache on Windows). Override it with --cache-dir or
the LANGUAGE_CACHE environment variable.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ENV_VAR = "LANGUAGE_CACHE"
_override: Path | None = None


def set_cache_root(path: str | Path | None) -> None:
    global _override
    _override = Path(path).expanduser() if path else None


def cache_root() -> Path:
    if _override is not None:
        return _override
    if env := os.environ.get(ENV_VAR):
        return Path(env).expanduser()
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "language" / "cache"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "language"


def cache_dir(*parts: str) -> Path:
    path = cache_root().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fmt(seconds: float) -> str:
    return f"{seconds:g}".replace(".", "_")


def range_tag(start: float | None, end: float | None) -> str:
    """'0-end', '30-90', '1_5-end'. None start is the same as 0, so keys stay stable."""
    return f"{_fmt(start or 0.0)}-{_fmt(end) if end is not None else 'end'}"


def cache_key(source_id: str, start: float | None, end: float | None, *extra: str) -> str:
    """Key for anything derived from (source, range), plus tool settings such as a model name."""
    safe = [source_id, range_tag(start, end), *extra]
    return "_".join(p.replace("/", "-").replace(os.sep, "-") for p in safe if p)


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    """Content hash of a local file (first 16 hex chars of SHA-256)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()[:16]


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]
