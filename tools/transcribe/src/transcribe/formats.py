"""Output formats: plain text, LRC, SRT, WebVTT, Markdown and JSON."""

from __future__ import annotations

import json

from .paragraphs import group_paragraphs, paragraph_text
from .segments import Segment

LOW_MARK = " [?]"
MD_TIMESTAMP_EVERY = 300.0  # Markdown: a timestamp every 5 minutes


def _mark(seg: Segment, mark_low: bool) -> str:
    return seg.text.strip() + (LOW_MARK if mark_low and seg.low_confidence else "")


def _clock(seconds: float, sep: str = ",") -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _short_clock(seconds: float) -> str:
    total = int(seconds)
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def to_text(segments: list[Segment], mode: str, mark_low: bool = True) -> str:
    if mode == "song":
        return "\n".join(_mark(s, mark_low) for s in segments) + ("\n" if segments else "")
    paragraphs = group_paragraphs(segments)
    blocks = []
    for para in paragraphs:
        text = paragraph_text(para, mark_low)
        speaker = para[0].speaker
        blocks.append(f"{speaker}: {text}" if speaker else text)
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_lrc(segments: list[Segment], title: str | None = None, language: str | None = None, mark_low: bool = True) -> str:
    out = []
    if title:
        out.append(f"[ti:{title}]")
    if language:
        out.append(f"[la:{language}]")
    out.append("[re:transcribe (Whisper)]")
    for seg in segments:
        cs = int(round(seg.start * 100))
        m, rest = divmod(cs, 6000)
        s, c = divmod(rest, 100)
        out.append(f"[{m:02d}:{s:02d}.{c:02d}]{_mark(seg, mark_low)}")
    return "\n".join(out) + "\n"


def _cue_text(seg: Segment, mark_low: bool) -> str:
    return f"{seg.speaker}: {_mark(seg, mark_low)}" if seg.speaker else _mark(seg, mark_low)


def to_srt(segments: list[Segment], mark_low: bool = True) -> str:
    cues = []
    for i, seg in enumerate(segments, 1):
        cues.append(f"{i}\n{_clock(seg.start)} --> {_clock(max(seg.end, seg.start + 0.3))}\n{_cue_text(seg, mark_low)}\n")
    return "\n".join(cues)


def to_vtt(segments: list[Segment], mark_low: bool = True) -> str:
    cues = ["WEBVTT", ""]
    for seg in segments:
        text = _mark(seg, mark_low)
        if seg.speaker:
            text = f"<v {seg.speaker}>{text}"
        cues.append(f"{_clock(seg.start, '.')} --> {_clock(max(seg.end, seg.start + 0.3), '.')}\n{text}\n")
    return "\n".join(cues)


def to_markdown(segments: list[Segment], meta: dict, mark_low: bool = True,
                every: float = MD_TIMESTAMP_EVERY) -> str:
    title = meta.get("title") or "Transcript"
    lines = [f"# {title}", ""]
    facts = []
    if meta.get("channel"):
        facts.append(f"Channel: {meta['channel']}")
    if meta.get("language"):
        facts.append(f"Language: {meta['language']}")
    if meta.get("duration"):
        facts.append(f"Duration: {_short_clock(meta['duration'])}")
    if meta.get("speakers"):
        facts.append(f"Speakers: {', '.join(meta['speakers'])}")
    if facts:
        lines += [" · ".join(facts), ""]
    next_mark = 0.0
    for para in group_paragraphs(segments):
        start = para[0].start
        if start >= next_mark:
            lines += [f"**[{_short_clock(start)}]**", ""]
            next_mark = (int(start // every) + 1) * every
        text = paragraph_text(para, mark_low)
        speaker = para[0].speaker
        lines += [f"**{speaker}:** {text}" if speaker else text, ""]
    if mark_low and any(s.low_confidence for s in segments):
        lines += ["---", "", f"`{LOW_MARK.strip()}` marks low-confidence passages.", ""]
    return "\n".join(lines)


def to_json(segments: list[Segment], meta: dict, words: bool = False) -> str:
    if segments and segments[0].paragraph is None and meta.get("mode") == "speech":
        group_paragraphs(segments)
    data = dict(meta)
    data["segments"] = [{"id": i, **s.to_dict(words)} for i, s in enumerate(segments, 1)]
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
