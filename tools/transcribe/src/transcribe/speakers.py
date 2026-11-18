"""Speaker labels: assigning diarization turns to segments and naming speakers."""

from __future__ import annotations

from .segments import Segment, Word


def _speaker_at(t: float, turns: list[tuple[float, float, str]]) -> str:
    for a, b, spk in turns:
        if a <= t < b:
            return spk
    return min(turns, key=lambda x: min(abs(t - x[0]), abs(t - x[1])))[2]


def _piece(seg: Segment, words: list[Word], speaker: str) -> Segment:
    return Segment(
        start=words[0].start, end=words[-1].end, text="".join(w.text for w in words).strip(),
        language=seg.language, avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob,
        compression_ratio=seg.compression_ratio, speaker=speaker, words=words,
        confidence=sum(w.prob for w in words) / len(words), low_confidence=seg.low_confidence,
    )


def split_at_turns(segments: list[Segment], turns: list[tuple[float, float, str]],
                   min_words: int = 2) -> list[Segment]:
    """Split segments whose words belong to different speakers (a reply often starts inside a
    Whisper segment). Runs shorter than min_words stay with their neighbour, so a stray
    misattributed word does not create a turn."""
    if not turns:
        return segments
    out: list[Segment] = []
    for seg in segments:
        if len(seg.words) < 2 * min_words:
            out.append(seg)
            continue
        labels = [_speaker_at((w.start + w.end) / 2, turns) for w in seg.words]
        runs: list[list] = []
        for w, lab in zip(seg.words, labels, strict=True):
            if runs and runs[-1][0] == lab:
                runs[-1][1].append(w)
            else:
                runs.append([lab, [w]])
        merged: list[list] = []
        for lab, ws in runs:  # absorb short runs into the previous one
            if merged and len(ws) < min_words:
                merged[-1][1].extend(ws)
            elif merged and merged[-1][0] == lab:
                merged[-1][1].extend(ws)
            else:
                merged.append([lab, ws])
        if len(merged) > 1 and len(merged[0][1]) < min_words:
            merged[1][1][:0] = merged[0][1]
            merged.pop(0)
        if len(merged) == 1:
            out.append(seg)
        else:
            out += [_piece(seg, ws, lab) for lab, ws in merged]
    return out


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
