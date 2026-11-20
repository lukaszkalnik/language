"""The window's choices kept between starts: a small JSON file next to the app's other data."""

from __future__ import annotations

import json
from pathlib import Path

from . import setup


def path() -> Path:
    return setup.data_dir() / "settings.json"


def load(file: Path | None = None) -> dict:
    """The saved settings; {} when there are none or the file cannot be read."""
    try:
        data = json.loads((file or path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(changes: dict, file: Path | None = None) -> None:
    """Merge changes into the saved settings; a failure to write is not worth an error."""
    file = file or path()
    data = load(file) | changes
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(file)
    except OSError:
        pass
