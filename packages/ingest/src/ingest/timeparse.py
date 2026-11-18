"""Time ranges: parse --start/--end values and format durations."""

from __future__ import annotations

import re

_TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$")


def parse_time(value: str) -> float:
    """Parse '90', '90.5', '1:30', '01:02:03' into seconds."""
    text = value.strip()
    match = _TIME_RE.match(text)
    if not match:
        raise ValueError(f"invalid time {value!r} (use seconds, mm:ss or hh:mm:ss)")
    parts = [p for p in match.groups() if p is not None]
    seconds = 0.0
    for i, part in enumerate(parts):
        number = float(part)
        if i > 0 and number >= 60:  # every field after the first must be below 60
            raise ValueError(f"invalid time {value!r}: minutes and seconds must be below 60")
        seconds = seconds * 60 + number
    return seconds


def validate_range(start: float | None, end: float | None) -> None:
    if start is not None and start < 0:
        raise ValueError("--start must not be negative")
    if start is not None and end is not None and end <= start:
        raise ValueError("--end must be after --start")


def format_seconds(seconds: float) -> str:
    """Format as m:ss or h:mm:ss."""
    total = int(round(seconds))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_range(start: float, end: float | None) -> str:
    return f"{format_seconds(start)}-{format_seconds(end) if end is not None else 'end'}"
