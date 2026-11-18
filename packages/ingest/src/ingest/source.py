"""Resolve a tool's input (file, YouTube URL or `-` for a WAV stream on stdin) to local audio."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from . import audio, cache, youtube
from .errors import AudioError
from .timeparse import validate_range


@dataclass
class Source:
    kind: str  # "youtube" | "file" | "stdin"
    path: Path  # local audio in its original format (for YouTube: only the downloaded section)
    source_id: str  # video ID or content hash: the stable part of every cache key
    start: float  # range in the original timeline
    end: float | None
    trim_start: float | None = None  # what is still to be cut from `path` when decoding
    trim_end: float | None = None
    info: youtube.VideoInfo | None = None
    capped: bool = False
    from_cache: bool = False
    label: str = ""
    extra: dict = field(default_factory=dict)

    def key(self, *extra: str) -> str:
        """Cache key for something derived from this input and range, plus tool settings."""
        return cache.cache_key(self.source_id, self.start, self.end, *extra)

    def to_wav(self, dst: Path, sample_rate: int | None = None, channels: int | None = None) -> Path:
        return audio.to_wav(self.path, dst, sample_rate, channels, self.trim_start, self.trim_end)

    def cached_wav(self, sample_rate: int, channels: int = 1, use_cache: bool = True) -> Path:
        """This input's range as WAV at a model's rate, kept in the shared cache."""
        out = cache.cache_dir("derived", "wav") / f"{self.key(f'{sample_rate}hz', f'{channels}ch')}.wav"
        if use_cache and out.exists() and out.stat().st_size > 44:
            return out
        tmp = out.with_suffix(".tmp.wav")
        self.to_wav(tmp, sample_rate, channels)
        tmp.replace(out)
        return out

    def describe(self) -> dict:
        data: dict = {"type": self.kind, "id": self.source_id, "range": {"start": self.start, "end": self.end, "capped": self.capped}}
        if self.info is not None:
            data.update(url=youtube.watch_url(self.info.id), title=self.info.title,
                        channel=self.info.channel, duration=self.info.duration, from_cache=self.from_cache)
        elif self.kind == "file":
            data["path"] = str(self.path)
        return data


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _read_stdin(stream: BinaryIO, use_cache: bool) -> tuple[Path, str]:
    data = stream.read()
    if not data:
        raise AudioError("nothing on stdin: pipe a WAV stream in, e.g. `vocals song.mp3 | transcribe -`")
    digest = cache.bytes_hash(data)
    path = cache.cache_dir("inputs") / f"stdin_{digest}.wav"
    if not (use_cache and path.exists()):
        path.write_bytes(data)
    return path, digest


def open_input(
    arg: str,
    start: float | None = None,
    end: float | None = None,
    cap: float | None = None,
    use_cache: bool = True,
    ipv4: bool = False,
    stdin: BinaryIO | None = None,
    show_info: bool = True,
) -> Source:
    """Turn the command-line input into a Source. Prints video info to stderr."""
    validate_range(start, end)
    audio.require_ffmpeg()

    if arg == "-":
        stream = stdin if stdin is not None else sys.stdin.buffer
        if stdin is None and sys.stdin.isatty():
            raise AudioError("`-` reads a WAV stream from stdin, but stdin is a terminal; pipe audio in")
        path, digest = _read_stdin(stream, use_cache)
        return Source("stdin", path, f"stdin-{digest}", start or 0.0, end, start, end, label="stdin")

    if youtube.looks_like_url(arg):
        def on_info(info, begin, finish, capped):
            if show_info:
                from .timeparse import format_range, format_seconds

                _status(f"Video:    {info.title}")
                _status(f"Channel:  {info.channel}")
                _status(f"Duration: {format_seconds(info.duration) if info.duration else 'unknown'}")
                if begin > 0 or finish is not None:
                    note = f" (first {format_seconds(cap)} only; use --full for the whole video)" if capped else ""
                    _status(f"Range:    {format_range(begin, finish)}{note}")

        dl = youtube.fetch(arg, cache.cache_dir("downloads"), start, end, cap, use_cache, ipv4, on_info)
        if show_info:
            _status(f"Audio:    {'cached ' if dl.from_cache else 'downloaded '}{dl.path}")
        return Source("youtube", dl.path, dl.info.id, dl.start, dl.end, info=dl.info,
                      capped=dl.capped, from_cache=dl.from_cache, label=dl.info.title)

    path = Path(arg).expanduser()
    if not path.is_file():
        raise AudioError(f"file not found: {arg}")
    return Source("file", path, f"file-{cache.file_hash(path)}", start or 0.0, end, start, end, label=path.name)


# --- audio output convention ----------------------------------------------------------

def audio_destination(output: str | None, stdout_isatty: bool, default_name: str) -> Path | None:
    """Where an audio tool writes: a Path, or None for a WAV stream on stdout.

    -o FILE writes FILE, -o - forces stdout. Without -o, a pipe/redirect gets the stream,
    and a terminal gets `default_name` in the current directory (no binary on screen).
    """
    if output == "-":
        return None
    if output:
        return Path(output).expanduser()
    return Path(default_name) if stdout_isatty else None


def emit_wav(wav: Path, destination: Path | None, stdout: BinaryIO | None = None) -> None:
    """Copy a finished WAV to its destination; announce file paths on stderr."""
    if destination is None:
        out = stdout if stdout is not None else sys.stdout.buffer
        with open(wav, "rb") as fh:
            shutil.copyfileobj(fh, out, 1 << 20)
        out.flush()
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in ("", ".wav"):
        shutil.copyfile(wav, destination)
    else:  # e.g. -o vocals.flac: let ffmpeg pick the codec from the extension
        audio.convert(wav, destination)
    _status(f"Wrote {destination}")
