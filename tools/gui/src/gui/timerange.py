"""The time range the window runs on: values of the two-handle slider, typed times, tool arguments.

No Tk here, so the logic is unit-tested without a display.
"""

from __future__ import annotations

import math
import re

MIN_SPAN = 1.0  # seconds the two handles stay apart


def format_time(seconds: float, total: float | None = None) -> str:
    """`m:ss`, or `h:mm:ss` when the file (`total`) or the time itself is an hour or longer."""
    s = int(round(max(0.0, seconds)))
    if s >= 3600 or (total is not None and total >= 3600):
        return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


_TIME = re.compile(r"^\s*(?:(\d+):)?(?:(\d+):)?(\d+(?:[.,]\d*)?)\s*$")


def parse_time(text: str) -> float:
    """Seconds from `90`, `1:30`, `1:02:03` (also `1.5` or `1,5`). Raises ValueError."""
    m = _TIME.match(text)
    if not m:
        raise ValueError(f"not a time: {text!r} (use seconds, m:ss or h:mm:ss)")
    a, b, sec = m.groups()
    if b is None and a is not None:  # m:ss
        hours, minutes = 0, int(a)
    else:
        hours, minutes = int(a or 0), int(b or 0)
    seconds = float(sec.replace(",", "."))
    if (a is not None and seconds >= 60) or (b is not None and minutes >= 60):
        raise ValueError(f"not a time: {text!r} (seconds and minutes go up to 59)")
    return hours * 3600 + minutes * 60 + seconds


class TimeRange:
    """Start and end within `duration`, snapped to whole seconds, at least MIN_SPAN apart.

    The end can also sit exactly on `duration` (which is rarely a whole second): "to the end".
    """

    def __init__(self, duration: float, min_span: float = MIN_SPAN):
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.duration = float(duration)
        self.min_span = min(min_span, self.duration)
        self.start = 0.0
        self.end = self.duration

    def _snap(self, value: float) -> float:
        """The nearest allowed position: a whole second, or the very end."""
        value = min(max(0.0, value), self.duration)
        whole = float(min(round(value), math.floor(self.duration)))
        return self.duration if abs(self.duration - value) < abs(whole - value) else whole

    def set_start(self, value: float) -> float:
        """Move the start handle; it stops min_span before the end."""
        self.start = max(0.0, min(self._snap(value), float(math.floor(self.end - self.min_span))))
        return self.start

    def set_end(self, value: float) -> float:
        """Move the end handle; it stops min_span after the start."""
        lowest = float(math.ceil(self.start + self.min_span))
        self.end = min(self.duration, max(self._snap(value), lowest))
        return self.end

    def set_text(self, which: str, text: str) -> float:
        """Set "start" or "end" from a typed time. Raises ValueError for text that is not a time."""
        value = parse_time(text)
        return self.set_start(value) if which == "start" else self.set_end(value)

    def reset(self) -> None:
        self.start, self.end = 0.0, self.duration

    @property
    def narrowed(self) -> bool:
        return self.start > 0 or self.end < self.duration

    def bounds(self) -> tuple[float | None, float | None]:
        """(start, end) for the tools: None where the range reaches the file's start or end."""
        return (self.start or None, self.end if self.end < self.duration else None)

    def label(self, which: str) -> str:
        return format_time(self.start if which == "start" else self.end, self.duration)


def range_args(start: float | None, end: float | None) -> list[str]:
    """`--start` / `--end` only for the parts that are narrowed, so whole-file caches are reused."""
    args: list[str] = []
    if start:
        args += ["--start", f"{start:g}"]
    if end is not None:
        args += ["--end", f"{end:g}"]
    return args
