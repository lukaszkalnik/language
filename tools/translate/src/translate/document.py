"""Read translate's input: plain text or transcribe --json, detected automatically."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

SPEAKER_RE = re.compile(r"^(Speaker \d+|[A-ZÀ-Ý][\w.'\- ]{0,30}?):\s+(\S.*)$")


@dataclass
class Unit:
    """One line (text input) or segment (JSON input) to translate."""

    index: int  # position in the input, 0-based (blank lines included for text)
    text: str
    number: int | None = None  # 1-based number shown next to the line; None for blank lines
    speaker: str | None = None
    language: str | None = None
    start: float | None = None
    end: float | None = None
    paragraph: int | None = None
    translation: str | None = None

    @property
    def blank(self) -> bool:
        return not self.text.strip()


@dataclass
class Document:
    kind: str  # "json" | "text"
    units: list[Unit]
    mode: str | None = None  # "song" | "speech" from transcribe JSON, else guessed
    language: str | None = None  # main source language if known
    data: dict | None = field(default=None, repr=False)  # the JSON input, kept for --json output
    via: str | None = None  # a model that took some lines into French first (Breton)
    backend_used: bool = False  # False when the lines in --via already gave every translation

    def content(self) -> list[Unit]:
        return [u for u in self.units if not u.blank]


def looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped)
    except ValueError:
        return False
    return isinstance(data, dict) and isinstance(data.get("segments"), list)


def parse(text: str) -> Document:
    """Detect transcribe JSON vs plain text and build the document."""
    if looks_like_json(text):
        return parse_json(json.loads(text))
    return parse_text(text)


def parse_json(data: dict) -> Document:
    units = []
    for i, seg in enumerate(data["segments"]):
        units.append(Unit(
            index=i, text=str(seg.get("text", "")).strip(), number=i + 1, speaker=seg.get("speaker"),
            language=seg.get("language"), start=seg.get("start"), end=seg.get("end"), paragraph=seg.get("paragraph"),
        ))
    mode = data.get("mode")
    if mode == "speech" and units and all(u.paragraph is None for u in units):
        for u in units:  # no paragraph info: every segment is its own group
            u.paragraph = u.index
    return Document("json", units, mode=mode, language=data.get("language"), data=data)


def parse_text(text: str) -> Document:
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    content = [ln for ln in lines if ln.strip()]
    # "Speaker 1: ..." prefixes (transcribe --speakers) are kept out of the translation,
    # but only when most lines have them: a lyric line like "Baby: come back" is not a label.
    prefixed = [SPEAKER_RE.match(ln.strip()) for ln in content]
    use_speakers = bool(content) and sum(1 for m in prefixed if m) >= max(2, 0.6 * len(content))
    units, number, paragraph = [], 0, 0
    for i, line in enumerate(lines):
        if not line.strip():
            units.append(Unit(i, ""))
            paragraph += 1
            continue
        number += 1
        speaker, body = None, line.strip()
        if use_speakers and (m := SPEAKER_RE.match(body)):
            speaker, body = m.group(1), m.group(2)
        units.append(Unit(i, body, number=number, speaker=speaker, paragraph=paragraph))
    has_blank_groups = any(u.blank for u in units)
    avg_len = sum(len(u.text) for u in units if not u.blank) / max(1, number)
    mode = "speech" if (use_speakers or avg_len > 120 or (has_blank_groups and avg_len > 80)) else "song"
    return Document("text", units, mode=mode)
