"""Speech mode: sentences grouped into paragraphs at pauses, speaker changes or topic shifts."""

from __future__ import annotations

from .segments import Segment

PARAGRAPH_PAUSE = 1.8  # seconds of silence that end a paragraph
MAX_PARAGRAPH_SECONDS = 90.0  # a monologue is still broken up
MIN_PARAGRAPH_SECONDS = 15.0  # ...but only at a sentence end and after this much
TOPIC_WORDS = ("so ", "now ", "okay ", "ok ", "alright ", "anyway", "next ", "moving on", "let's talk",
               "also,", "first", "second", "finally", "in summary")


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "…", '."', '?"', "。", "？", "！"))


def group_paragraphs(segments: list[Segment]) -> list[list[Segment]]:
    """Group segments into paragraphs. Sets each segment's `paragraph` index."""
    paragraphs: list[list[Segment]] = []
    current: list[Segment] = []
    for seg in segments:
        if current:
            last = current[-1]
            gap = seg.start - last.end
            length = last.end - current[0].start
            speaker_change = seg.speaker != last.speaker
            sentence_end = _ends_sentence(last.text)
            topic = sentence_end and length >= MIN_PARAGRAPH_SECONDS and \
                seg.text.strip().lower().startswith(TOPIC_WORDS)
            long_enough = sentence_end and length >= MAX_PARAGRAPH_SECONDS
            if speaker_change or (gap >= PARAGRAPH_PAUSE and sentence_end) or gap >= 2 * PARAGRAPH_PAUSE \
                    or topic or long_enough or seg.language != last.language:
                paragraphs.append(current)
                current = []
        current.append(seg)
    if current:
        paragraphs.append(current)
    for i, para in enumerate(paragraphs):
        for seg in para:
            seg.paragraph = i
    return paragraphs


def paragraph_text(paragraph: list[Segment], mark_low: bool = True) -> str:
    parts = []
    for seg in paragraph:
        text = seg.text.strip()
        parts.append(f"{text} [?]" if mark_low and seg.low_confidence else text)
    return " ".join(p for p in parts if p)
