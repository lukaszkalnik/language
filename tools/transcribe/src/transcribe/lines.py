"""Song mode: one sung line per output line, split using word timestamps."""

from __future__ import annotations

from .segments import Segment, Word

LINE_PAUSE = 0.45  # a gap between words longer than this starts a new line
MAX_LINE_WORDS = 12
MAX_LINE_SECONDS = 7.0


def _new_line(words: list[Word], template: Segment) -> Segment:
    text = "".join(w.text for w in words).strip()
    return Segment(
        start=words[0].start, end=words[-1].end, text=text, language=template.language,
        avg_logprob=template.avg_logprob, no_speech_prob=template.no_speech_prob,
        compression_ratio=template.compression_ratio, speaker=template.speaker, words=list(words),
    )


ORPHAN_WORDS = 2  # a line this short is joined to its neighbour (a held last word, a stray "too")
ORPHAN_GAP = 1.5


def _joinable(left: Segment, right: Segment) -> bool:
    return right.start - left.end <= ORPHAN_GAP and not left.text.rstrip().endswith((".", "!", "?")) \
        and len(left.words) + len(right.words) <= MAX_LINE_WORDS + ORPHAN_WORDS


def merge_orphans(lines: list[Segment]) -> list[Segment]:
    """Join one- or two-word lines to the previous line (or else the next one)."""
    out: list[Segment] = []
    for line in lines:
        if out and line.words and len(line.words) <= ORPHAN_WORDS and _joinable(out[-1], line):
            out[-1] = _new_line(out[-1].words + line.words, out[-1])
        elif out and out[-1].words and len(out[-1].words) <= ORPHAN_WORDS and line.words and _joinable(out[-1], line) \
                and (len(out) < 2 or not _joinable(out[-2], out[-1])):
            out[-1] = _new_line(out[-1].words + line.words, line)
        else:
            out.append(line)
    return out


def split_lines(segments: list[Segment]) -> list[Segment]:
    """Split Whisper segments into sung lines at pauses, sentence ends and capitalised restarts."""
    return merge_orphans(_split(segments))


def _split(segments: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        if not seg.words:
            out.append(seg)
            continue
        current: list[Word] = []
        for word in seg.words:
            if current:
                gap = word.start - current[-1].end
                stripped = word.text.strip()
                prev = current[-1].text.strip()
                ends_sentence = prev.endswith((".", "!", "?", "…"))
                capital_restart = stripped[:1].isupper() and stripped not in ("I", "I'm", "I'll", "I've", "I'd") \
                    and (len(current) >= 4 or gap > 0.15) and len(current) >= 2
                too_long = len(current) >= MAX_LINE_WORDS or word.end - current[0].start > MAX_LINE_SECONDS
                if gap > LINE_PAUSE or ends_sentence or capital_restart or too_long:
                    out.append(_new_line(current, seg))
                    current = []
            current.append(word)
        if current:
            out.append(_new_line(current, seg))
    return out
