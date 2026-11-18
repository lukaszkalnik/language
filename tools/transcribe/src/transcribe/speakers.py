"""Speaker labels: assigning diarization turns to segments and naming speakers."""

from __future__ import annotations

from .segments import Segment


def assign_speakers(segments: list[Segment], turns: list[tuple[float, float, str]]) -> None:
    """Give each segment the speaker whose turns overlap it most (nearest turn if none overlaps)."""
    if not turns:
        return
    for seg in segments:
        overlap: dict[str, float] = {}
        for a, b, spk in turns:
            o = min(seg.end, b) - max(seg.start, a)
            if o > 0:
                overlap[spk] = overlap.get(spk, 0.0) + o
        if overlap:
            seg.speaker = max(overlap, key=overlap.get)
        else:
            mid = (seg.start + seg.end) / 2
            seg.speaker = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))[2]


def parse_names(text: str | None) -> list[str]:
    if not text:
        return []
    return [n.strip() for n in text.split(",") if n.strip()]


def name_speakers(segments: list[Segment], names: list[str] | None = None) -> list[str]:
    """Rename raw labels to 'Speaker 1', 'Speaker 2'... (or given names) in order of first appearance.

    Returns the display names in that order.
    """
    names = names or []
    mapping: dict[str, str] = {}
    for seg in segments:
        if seg.speaker is None or seg.speaker in mapping:
            continue
        idx = len(mapping)
        mapping[seg.speaker] = names[idx] if idx < len(names) else f"Speaker {idx + 1}"
    for seg in segments:
        if seg.speaker is not None:
            seg.speaker = mapping[seg.speaker]
    return list(mapping.values())
