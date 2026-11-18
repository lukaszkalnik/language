"""Split a document into translation requests that keep whole paragraphs together.

Each chunk carries a few units from the end of the previous chunk as context (with their
translations once known), so meaning carries across chunk borders. Speaker labels and
timestamps are never sent to the model; they stay on the units.
"""

from __future__ import annotations

from dataclasses import dataclass

from .document import Document, Unit

MAX_CHARS = 2400
OVERLAP_UNITS = 3
SONG_MAX_CHARS = 12000  # a song goes in one request unless it is unusually long


@dataclass
class Chunk:
    units: list[Unit]  # to translate
    context: list[Unit]  # previous units, for reference only


def paragraphs(units: list[Unit]) -> list[list[Unit]]:
    groups: list[list[Unit]] = []
    last = object()
    for u in units:
        if u.blank:
            continue
        if not groups or u.paragraph != last:
            groups.append([])
            last = u.paragraph
        groups[-1].append(u)
    return groups


def make_chunks(doc: Document, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_UNITS,
                whole: bool | None = None) -> list[Chunk]:
    content = doc.content()
    if not content:
        return []
    whole = (doc.mode == "song") if whole is None else whole
    if whole and sum(len(u.text) for u in content) <= SONG_MAX_CHARS:
        return [Chunk(content, [])]
    groups = paragraphs(content)
    batches: list[list[Unit]] = []
    current: list[Unit] = []
    size = 0
    for group in groups:
        g_size = sum(len(u.text) for u in group)
        if current and size + g_size > max_chars:
            batches.append(current)
            current, size = [], 0
        if g_size > max_chars:  # one huge paragraph: split it by units
            for u in group:
                if current and size + len(u.text) > max_chars:
                    batches.append(current)
                    current, size = [], 0
                current.append(u)
                size += len(u.text)
            continue
        current += group
        size += g_size
    if current:
        batches.append(current)
    chunks, previous = [], []
    for batch in batches:
        chunks.append(Chunk(batch, previous[-overlap:] if overlap else []))
        previous = batch
    return chunks
