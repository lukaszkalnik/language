"""Ollama, the better translator: is it installed, running, with translate's model? Start it if needed.

An installed Ollama whose server is not running (Homebrew's, or the app not opened since a
restart) is started here with `ollama serve`, and stopped again when the window closes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from enum import Enum
from pathlib import Path
from typing import assert_never

MODEL = "gemma4:12b"  # translate's default Ollama model
# Apps started from the Finder get a minimal PATH, so the usual places are looked at too.
PLACES = [Path("/opt/homebrew/bin/ollama"), Path("/usr/local/bin/ollama"),
          Path("/Applications/Ollama.app/Contents/Resources/ollama")]
START_TIMEOUT = 20.0  # seconds for a started server to answer


class Status(Enum):
    READY = "ready"  # running, with the model
    NO_MODEL = "no model"  # running, without the model
    NOT_STARTED = "not started"  # installed, but its server did not come up
    NOT_INSTALLED = "not installed"


_started: subprocess.Popen | None = None


def models(host: str | None = None, timeout: float = 2) -> list[str] | None:
    """Models in a running Ollama, or None when it is not running."""
    host = host or os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    url = (host if host.startswith("http") else f"http://{host}") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return [m.get("name", "") for m in json.load(resp).get("models", [])]
    except (OSError, ValueError):
        return None


def has_model(names: list[str], model: str = MODEL) -> bool:
    return any(m == model or m.startswith(model + "-") for m in names)


def binary() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    return next((str(p) for p in PLACES if p.is_file()), None)


def start(command: str, timeout: float = START_TIMEOUT) -> list[str] | None:
    """Start `ollama serve` and wait until it answers; its models, or None if it does not come up."""
    global _started
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        _started = subprocess.Popen([command, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, creationflags=flags)
    except OSError:
        return None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _started.poll() is None:
        found = models(timeout=1)
        if found is not None:
            return found
        time.sleep(0.3)
    return None


def status(model: str = MODEL) -> Status:
    """Whether translate can use Ollama, starting its server when it is installed but not running."""
    found = models()
    if found is None:
        command = binary()
        if command is None:
            return Status.NOT_INSTALLED
        found = start(command)
        if found is None:
            return Status.NOT_STARTED
    return Status.READY if has_model(found, model) else Status.NO_MODEL


def stop() -> None:
    """Stop the server started here, if any (one the user started stays)."""
    global _started
    if _started is not None and _started.poll() is None:
        _started.terminate()
    _started = None


def tip(state: Status, model: str = MODEL) -> str | None:
    """Why NLLB translated instead, and what makes Ollama do it; None when Ollama translated."""
    lead = "Translated offline with NLLB. Ollama translates better"
    match state:
        case Status.READY:
            return None
        case Status.NO_MODEL:
            return f"{lead}: it is running but lacks its model; download it with: ollama pull {model}"
        case Status.NOT_STARTED:
            return f"{lead}: it is installed but did not start; open the Ollama app and translate again."
        case Status.NOT_INSTALLED:
            return f"{lead}: install it from https://ollama.com, then download its model with: ollama pull {model}"
    assert_never(state)
