"""Output: side-by-side text, translation only, or JSON that keeps timestamps and speakers."""

from __future__ import annotations

import copy
import json
import textwrap
import unicodedata

from .document import Document, Unit

SEP = " │ "


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 0 if unicodedata.combining(ch) else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def _label(unit: Unit, text: str) -> str:
    return f"{unit.speaker}: {text}" if unit.speaker else text


def side_by_side(doc: Document, width: int = 120) -> str:
    """Numbered rows, original left and translation right, wrapped so they stay aligned."""
    content = doc.content()
    num_w = len(str(content[-1].number)) if content else 1
    col = max(20, (width - num_w - 2 - len(SEP)) // 2)
    longest = max((display_width(_label(u, u.text)) for u in content), default=0)
    left_w = min(col, longest)
    right_w = max(20, width - num_w - 2 - len(SEP) - left_w)
    out = []
    for unit in doc.units:
        if unit.blank:
            out.append("")
            continue
        left = textwrap.wrap(_label(unit, unit.text), left_w) or [""]
        right = textwrap.wrap(_label(unit, unit.translation or ""), right_w) or [""]
        for i in range(max(len(left), len(right))):
            num = str(unit.number).rjust(num_w) if i == 0 else " " * num_w
            a = left[i] if i < len(left) else ""
            b = right[i] if i < len(right) else ""
            out.append(f"{num}  {_pad(a, left_w)}{SEP}{b}".rstrip())
    return "\n".join(out) + "\n"


def only_translation(doc: Document) -> str:
    lines = ["" if u.blank else _label(u, u.translation or "") for u in doc.units]
    return "\n".join(lines) + "\n"


def to_json(doc: Document, source: str | None, target: str, backend: str) -> str:
    if doc.kind == "json":
        data = copy.deepcopy(doc.data)
        for unit, seg in zip(doc.units, data["segments"], strict=True):
            seg["translation"] = unit.translation
            seg["source_language"] = unit.language or source
    else:
        data = {"mode": doc.mode, "segments": []}
        for unit in doc.content():
            seg = {"id": unit.number, "text": unit.text, "translation": unit.translation,
                   "source_language": unit.language or source, "paragraph": unit.paragraph}
            if unit.speaker:
                seg["speaker"] = unit.speaker
            data["segments"].append(seg)
    data["translation"] = {"source_language": source, "target_language": target, "backend": backend}
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
