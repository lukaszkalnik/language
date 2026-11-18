"""Hallucination control for Whisper output.

Whisper invents text on silence, music and noise: "Thanks for watching", subtitle credits,
the same phrase repeated, or text where nobody speaks. Segments that are clearly invented are
dropped; doubtful ones are kept and marked low-confidence, so nothing real is silently hidden.
"""

from __future__ import annotations

import math
import re
import zlib

from .segments import Segment

# Phrases Whisper is known to produce on silence or music (lower-case, punctuation stripped).
KNOWN_PHRASES = (
    "thanks for watching", "thank you for watching", "thank you so much for watching",
    "please subscribe", "subscribe to my channel", "like and subscribe", "see you in the next video",
    "subtitles by", "subtitled by", "captions by", "transcribed by", "amaraorg", "amara org",
    "untertitel im auftrag des zdf", "untertitel der amaraorg community", "untertitelung",
    "napisy wykonane przez", "napisy stworzone przez", "sous-titres réalisés par", "sous titres",
    "sottotitoli creati dalla comunità", "subtítulos realizados por", "продолжение следует",
)
MUSIC_ONLY = re.compile(r"^[\s♪♫🎵🎶\[\]\(\)*#.,!-]*(music|musik|musique|música|muzyka|instrumental|applause)?[\s♪♫🎵🎶\[\]\(\)*#.,!-]*$", re.I)

COMPRESSION_LIMIT = 2.4  # gzip ratio above this: repetitive loop
NO_SPEECH_LIMIT = 0.6
LOGPROB_LIMIT = -1.0
LOW_CONFIDENCE = 0.55  # mark lines whose confidence is below this
MIN_VOICE_OVERLAP = 0.2  # fraction of a segment that must overlap detected voice


def normalise(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data))) if data else 0.0


def is_known_phrase(text: str) -> bool:
    norm = re.sub(r"\s+", " ", normalise(text))
    return any(p in norm for p in KNOWN_PHRASES)


def collapse_repeats(text: str, max_repeats: int = 2) -> str:
    """'na na na na na na' -> 'na na' style loops are cut to max_repeats repetitions of the unit."""
    words = text.split()
    for unit in range(1, 7):
        i, out = 0, []
        while i < len(words):
            chunk = words[i:i + unit]
            reps = 1
            while words[i + reps * unit:i + (reps + 1) * unit] == chunk and len(chunk) == unit:
                reps += 1
            if reps > max_repeats + 1 and unit * reps >= 6:
                out.extend(chunk * max_repeats)
            else:
                out.extend(words[i:i + reps * unit])
            i += reps * unit
        words = out
    return " ".join(words)


def voice_overlap(seg: Segment, regions: list[tuple[float, float]]) -> float:
    """Fraction of the segment covered by voice regions."""
    if seg.duration <= 0:
        return 0.0
    covered = sum(max(0.0, min(seg.end, b) - max(seg.start, a)) for a, b in regions)
    return min(1.0, covered / seg.duration)


def confidence_of(seg: Segment) -> float:
    if seg.words:
        return sum(w.prob for w in seg.words) / len(seg.words)
    return math.exp(min(0.0, seg.avg_logprob))


def score(seg: Segment, low_confidence: float = LOW_CONFIDENCE) -> Segment:
    """Set confidence and the low-confidence mark of a segment."""
    seg.confidence = confidence_of(seg)
    seg.low_confidence = seg.confidence < low_confidence or compression_ratio(seg.text) > COMPRESSION_LIMIT \
        or seg.compression_ratio > COMPRESSION_LIMIT or seg.avg_logprob < LOGPROB_LIMIT
    return seg


def filter_segments(
    segments: list[Segment],
    voice: list[tuple[float, float]] | None = None,
    low_confidence: float = LOW_CONFIDENCE,
) -> tuple[list[Segment], list[tuple[Segment, str]]]:
    """Return (kept, dropped-with-reason). Kept segments get confidence and low_confidence set."""
    kept: list[Segment] = []
    dropped: list[tuple[Segment, str]] = []
    previous = None
    for seg in segments:
        text = seg.text.strip()
        reason = None
        if not text or MUSIC_ONLY.match(text):
            reason = "no words"
        elif voice is not None and voice_overlap(seg, voice) < MIN_VOICE_OVERLAP:
            reason = "no voice detected"
        elif seg.no_speech_prob > NO_SPEECH_LIMIT and seg.avg_logprob < LOGPROB_LIMIT:
            reason = "no speech"
        elif is_known_phrase(text) and (seg.avg_logprob < -0.5 or seg.no_speech_prob > 0.3 or seg.duration < 4):
            reason = "known hallucination"
        elif previous is not None and normalise(text) == normalise(previous.text) and seg.start - previous.end < 1.0 \
                and seg.avg_logprob < -0.7:
            reason = "repeated line"
        if reason:
            dropped.append((seg, reason))
            continue
        seg.text = collapse_repeats(text)
        score(seg, low_confidence)
        kept.append(seg)
        previous = seg
    return kept, dropped
