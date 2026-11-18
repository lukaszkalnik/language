"""ffmpeg-based audio helpers. Resampling happens in the tool that needs it, not here:
`to_wav(..., sample_rate=None)` keeps the source rate so `vocals` keeps full quality."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import AudioError


def ffmpeg_install_hint() -> str:
    if sys.platform == "darwin":
        return "Install it with: brew install ffmpeg"
    if sys.platform == "win32":
        return "Install it with: winget install ffmpeg (then open a new terminal)"
    return "Install it with your package manager, e.g.: sudo apt install ffmpeg"


def require_ffmpeg() -> str:
    """Return the ffmpeg executable path, or raise AudioError with an install hint."""
    path = shutil.which("ffmpeg")
    if not path:
        raise AudioError(f"ffmpeg was not found on PATH. {ffmpeg_install_hint()}")
    return path


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        lines = proc.stderr.decode("utf-8", "replace").strip().splitlines() or ["unknown error"]
        raise AudioError(f"ffmpeg could not read {what}: {lines[-1]}")
    return proc


@dataclass
class AudioInfo:
    sample_rate: int
    channels: int
    duration: float | None


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_STREAM_RE = re.compile(r"Audio:.*?(\d+) Hz,\s*([^,]+)")


def probe(path: Path) -> AudioInfo:
    """Sample rate, channel count and duration of the first audio stream."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        proc = _run(
            [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate,channels:format=duration", "-of", "json", str(path)],
            path.name,
        )
        data = json.loads(proc.stdout or b"{}")
        streams = data.get("streams") or []
        if not streams:
            raise AudioError(f"{path.name} has no audio stream")
        duration = (data.get("format") or {}).get("duration")
        return AudioInfo(int(streams[0]["sample_rate"]), int(streams[0]["channels"]),
                         float(duration) if duration not in (None, "N/A") else None)
    # Fallback: parse `ffmpeg -i` output.
    proc = subprocess.run([require_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True)
    text = proc.stderr.decode("utf-8", "replace")
    stream = _STREAM_RE.search(text)
    if not stream:
        raise AudioError(f"{path.name} has no audio stream")
    layout = stream.group(2).strip()
    count = re.match(r"(\d+)", layout)
    channels = 1 if layout == "mono" else int(count.group(1)) if count else 2
    dur = _DUR_RE.search(text)
    duration = int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)) if dur else None
    return AudioInfo(int(stream.group(1)), channels, duration)


def to_wav(
    src: Path,
    dst: Path,
    sample_rate: int | None = None,
    channels: int | None = None,
    start: float | None = None,
    end: float | None = None,
    codec: str = "pcm_s16le",
) -> Path:
    """Convert anything ffmpeg reads to WAV. None keeps the source's rate / channel count."""
    cmd = [require_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]  # input seeking: fast, timestamps restart at 0
    cmd += ["-i", str(src)]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0.0):.3f}"]
    cmd += ["-vn", "-map", "0:a:0"]
    if channels:
        cmd += ["-ac", str(channels)]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    cmd += ["-c:a", codec, "-f", "wav", str(dst)]
    _run(cmd, src.name)
    return dst


def convert(src: Path, dst: Path) -> Path:
    """Re-encode `src` into whatever format `dst`'s extension implies (flac, mp3, m4a...)."""
    _run([require_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), str(dst)], src.name)
    return dst


def decode(
    src: Path,
    sample_rate: int,
    channels: int = 1,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    """Decode to float32 samples, shape (n,) for mono or (n, channels)."""
    cmd = [require_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0.0):.3f}"]
    cmd += ["-vn", "-map", "0:a:0", "-ac", str(channels), "-ar", str(sample_rate), "-f", "f32le", "-"]
    proc = _run(cmd, src.name)
    samples = np.frombuffer(proc.stdout, dtype="<f4").copy()
    return samples if channels == 1 else samples.reshape(-1, channels)


def write_wav(dst, samples: np.ndarray, sample_rate: int) -> None:
    """Write float samples (n,) or (n, ch) as 16-bit PCM WAV to a path or binary file object."""
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).round().astype("<i2")
    target = str(dst) if isinstance(dst, Path) else dst
    with wave.open(target, "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class WavReader:
    """Read slices of a 16-bit PCM WAV without loading the whole file (flat memory for long audio)."""

    def __init__(self, path: Path):
        self.path = path
        self._wf = wave.open(str(path), "rb")
        if self._wf.getsampwidth() != 2:
            raise AudioError(f"{path.name} is not 16-bit PCM")
        self.sample_rate = self._wf.getframerate()
        self.channels = self._wf.getnchannels()
        self.frames = self._wf.getnframes()

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    def read(self, start: float = 0.0, end: float | None = None) -> np.ndarray:
        a = max(0, int(round(start * self.sample_rate)))
        b = self.frames if end is None else min(self.frames, int(round(end * self.sample_rate)))
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        self._wf.setpos(a)
        raw = self._wf.readframes(b - a)
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return data if self.channels == 1 else data.reshape(-1, self.channels)

    def close(self) -> None:
        self._wf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return float(20 * np.log10(rms)) if rms > 0 else float("-inf")


# --- WAV metadata (RIFF LIST/INFO), so tools can pass facts down a pipe -----------------
# vocals writes e.g. ISFT="vocals/htdemucs_ft/vocals" and INAM=<video title>; transcribe reads
# it to know its input is already separated. ffmpeg and the wave module ignore the chunk.

def add_wav_tags(path: Path, tags: dict[str, str]) -> None:
    """Append a LIST/INFO chunk to a WAV file. Keys are 4-letter RIFF INFO IDs (INAM, ISFT, ICMT...)."""
    body = b"INFO"
    for key, value in tags.items():
        if len(key) != 4:
            raise ValueError(f"RIFF INFO keys have 4 characters: {key!r}")
        data = value.encode("utf-8") + b"\0"
        body += key.encode("ascii") + len(data).to_bytes(4, "little") + data + (b"\0" if len(data) % 2 else b"")
    chunk = b"LIST" + len(body).to_bytes(4, "little") + body
    with open(path, "r+b") as fh:
        header = fh.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise AudioError(f"{path.name} is not a WAV file")
        fh.seek(0, 2)
        if fh.tell() % 2:
            fh.write(b"\0")
        fh.write(chunk)
        size = fh.tell() - 8
        fh.seek(4)
        fh.write(size.to_bytes(4, "little"))


def read_wav_tags(path: Path) -> dict[str, str]:
    """RIFF INFO tags of a WAV file ({} for other formats or files without tags)."""
    tags: dict[str, str] = {}
    try:
        with open(path, "rb") as fh:
            header = fh.read(12)
            if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                return tags
            while True:
                head = fh.read(8)
                if len(head) < 8:
                    break
                cid, size = head[:4], int.from_bytes(head[4:], "little")
                if cid == b"LIST":
                    body = fh.read(size)
                    if body[:4] == b"INFO":
                        pos = 4
                        while pos + 8 <= len(body):
                            key = body[pos:pos + 4].decode("ascii", "replace")
                            n = int.from_bytes(body[pos + 4:pos + 8], "little")
                            tags[key] = body[pos + 8:pos + 8 + n].rstrip(b"\0").decode("utf-8", "replace")
                            pos += 8 + n + (n % 2)
                elif cid == b"data" and size in (0, 0xFFFFFFFF):
                    break  # streamed WAV of unknown length: nothing reliable after it
                else:
                    fh.seek(size + (size % 2), 1)
                if size % 2 and cid == b"LIST":
                    fh.seek(1, 1)
    except OSError:
        return {}
    return tags
