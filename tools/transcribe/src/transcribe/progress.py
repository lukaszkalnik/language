"""Progress on stderr while transcribing: a line for people, `@progress` and `@partial` for the window."""

from __future__ import annotations

import sys
import time

from ingest.cli import report_partial, report_progress
from ingest.timeparse import format_seconds

from .segments import Segment


def piece_text(segments: list[Segment]) -> str:
    """A transcribed piece as the window shows it while the run goes on: one line per piece."""
    text = " ".join(s.text.strip() for s in segments if s.text.strip())
    return text + "\n" if text else ""


class Progress:
    def __init__(self, label: str, total: float):
        self.label, self.total, self.t0 = label, total, time.time()
        self.tty = sys.stderr.isatty()
        self.last_print = 0.0

    def __call__(self, done: float, segments: list[Segment] | None = None) -> None:
        """`done` seconds of `total`; `segments`: what the piece just finished says, for the window."""
        report_partial(piece_text(segments or []))
        report_progress(self.label, done / self.total if self.total else 1.0)
        now = time.time()
        if not self.tty and now - self.last_print < 30 and done < self.total:
            return  # in logs, one line every 30 s is enough
        self.last_print = now
        frac = done / self.total if self.total else 1.0
        elapsed = now - self.t0
        eta = f", ~{format_seconds(elapsed / frac - elapsed)} left" if 0.02 < frac < 1 else ""
        msg = f"{self.label} {format_seconds(done)} / {format_seconds(self.total)} ({frac:.0%}{eta})"
        print(("\r" + msg + "\033[K") if self.tty else msg, file=sys.stderr, end="" if self.tty else "\n", flush=True)

    def finish(self) -> None:
        if self.tty:
            print(file=sys.stderr, flush=True)
