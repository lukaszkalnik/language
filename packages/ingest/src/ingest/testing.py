"""Helpers for the tools' integration tests: catch any network connection a run opens."""

from __future__ import annotations

import os
import socket
import threading
import wave
from pathlib import Path

PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")


class ConnectionTrap:
    """A local listener posing as the proxy: records each connection and closes it at once.

    Run a tool with `env()` and any HTTP(S) request it makes (Hugging Face Hub, telemetry)
    lands here instead of on the network and fails fast. `requests` holds what each
    connection sent first (e.g. b"CONNECT huggingface.co:443 ...").
    """

    def __init__(self) -> None:
        self.requests: list[bytes] = []
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(16)
        self._server.settimeout(0.2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.getsockname()[1]}"

    def env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        for name in PROXY_VARS:
            env[name] = self.url
        env.pop("NO_PROXY", None)
        env.pop("no_proxy", None)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):  # the point is to work without them
            env.pop(name, None)
        return env

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except (TimeoutError, OSError):
                continue
            with conn:
                conn.settimeout(1.0)
                try:
                    self.requests.append(conn.recv(4096).split(b"\r\n", 1)[0])
                except OSError:
                    self.requests.append(b"")

    def __enter__(self) -> ConnectionTrap:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._server.close()


def write_noise_wav(path: Path, seconds: float = 8.0, rate: int = 16000) -> Path:
    """A mono 16-bit WAV of quiet random noise, different on each call (so no result is cached)."""
    import numpy as np

    samples = np.random.default_rng().integers(-2000, 2000, int(seconds * rate), dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())
    return path


def run_twice_second_trapped(module: str, *args: str, stdin: str | None = None, timeout: float = 1800):
    """Run `python -m <module> args` once normally (downloading what it needs), then again with
    every connection going to a ConnectionTrap. Returns (second run, trapped requests)."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", module, *args]
    first = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout)
    assert first.returncode == 0, first.stderr
    with ConnectionTrap() as trap:
        second = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout, env=trap.env())
    return second, trap.requests


def skip_if_youtube_refuses_ci(stderr: str) -> None:
    """Skip a YouTube test when YouTube's bot check refused the download on a CI machine.

    YouTube refuses most downloads from data-centre addresses (GitHub's runners); there it says
    nothing about our code. On a person's machine the same refusal fails the test, since it may
    mean yt-dlp needs an upgrade.
    """
    if os.environ.get("CI") and "confirm this is not a bot" in stderr:
        import pytest

        pytest.skip("YouTube refused the download from this CI machine (bot check)")
