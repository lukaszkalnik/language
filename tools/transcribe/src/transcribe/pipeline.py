"""transcribe: input -> VAD -> mode -> language -> Whisper (chunked, resumable) -> filter -> speakers."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ingest.audio import WavReader, read_wav_tags
from ingest.cache import cache_dir
from ingest.cli import report_progress
from ingest.errors import AudioError
from ingest.source import Source, open_input
from ingest.timeparse import format_seconds

from . import hallucination, langdetect, lines, mode, speakers, vad
from .segments import Segment

SR = 16_000
BATCH_SECONDS = 600.0  # speech mode: results are cached every ~10 minutes of audio (resume point)


def status(message: str = "", end: str = "\n") -> None:
    print(message, file=sys.stderr, end=end, flush=True)


@dataclass
class Options:
    input: str
    start: float | None = None
    end: float | None = None
    mode: str = "auto"
    language: str | None = None
    model: str = "auto"
    speakers: bool = False
    num_speakers: int | None = None
    names: list[str] = field(default_factory=list)
    use_cache: bool = True
    ipv4: bool = False
    device: str = "auto"


@dataclass
class Result:
    source: Source
    mode: str
    mode_reason: str
    separated: bool
    language: langdetect.LanguageReport
    segments: list[Segment]
    dropped: int
    speaker_names: list[str]
    duration: float
    backend: str
    title: str | None = None

    def meta(self) -> dict:
        src = self.source.describe()
        return {
            "source": src,
            "title": self.title or src.get("title") or (self.source.label if self.source.kind == "file" else None),
            "channel": src.get("channel"),
            "duration": round(self.duration, 2),
            "mode": self.mode,
            "mode_reason": self.mode_reason,
            "separated_vocals": self.separated,
            "language": self.language.main,
            "languages": self.language.languages,
            "speakers": self.speaker_names,
            "model": self.backend,
            "dropped_segments": self.dropped,
        }


class Progress:
    def __init__(self, label: str, total: float):
        self.label, self.total, self.t0 = label, total, time.time()
        self.tty = sys.stderr.isatty()
        self.last_print = 0.0

    def __call__(self, done: float) -> None:
        report_progress(self.label, done / self.total if self.total else 1.0)
        now = time.time()
        if not self.tty and now - self.last_print < 30 and done < self.total:
            return  # in logs, one line every 30 s is enough
        self.last_print = now
        frac = done / self.total if self.total else 1.0
        elapsed = now - self.t0
        eta = f", ~{format_seconds(elapsed / frac - elapsed)} left" if 0.02 < frac < 1 else ""
        msg = f"{self.label} {format_seconds(done)} / {format_seconds(self.total)} ({frac:.0%}{eta})"
        status(("\r" + msg + "\033[K") if self.tty else msg, end="" if self.tty else "\n")

    def finish(self) -> None:
        if self.tty:
            status()


def _json_cache(path: Path, compute, use_cache: bool):
    if use_cache and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    value = compute()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value), encoding="utf-8")
    tmp.replace(path)
    return value


def _is_separated(tags: dict) -> bool:
    tag = tags.get("ISFT", "")
    return tag.startswith("vocals/") and tag.endswith("/vocals")


def detect_mode(reader: WavReader, separated: bool) -> mode.ModeDecision:
    feats = [mode.features(reader.read(a, b)) for a, b in mode.sample_windows(reader.duration)]
    feats = [f for f in feats if f.seconds > 0]
    if not feats:
        raise AudioError("no audio in the selected range")
    weights = np.array([f.seconds for f in feats])
    avg = mode.ModeFeatures(*(float(np.average([getattr(f, k) for f in feats], weights=weights))
                              for k in ("voiced_ratio", "sustained_ratio", "silence_ratio")), float(weights.sum()))
    return mode.decide(avg, separated)


def detect_language(model, reader: WavReader, voice, cache_path: Path, use_cache: bool) -> langdetect.LanguageReport:
    windows = langdetect.pick_windows(reader.duration, voice)

    def compute():
        return [model.detect(reader.read(a, b)) for a, b in windows]

    status(f"Detecting language on {len(windows)} segment(s) across the file...")
    probs = _json_cache(cache_path, compute, use_cache)
    return langdetect.aggregate(probs)


def _restrict(probs: dict[str, float], allowed: list[str]) -> str:
    return max(allowed, key=lambda c: probs.get(c, 0.0))


def transcribe_speech(model, reader, voice, report, workdir: Path, use_cache: bool, fixed_language: str | None,
                      suffix: str = ""):
    def cut_at(lo: float, hi: float) -> float:
        return vad.quietest_point(reader.read(lo, hi), lo)

    pieces = vad.plan_pieces(voice, duration=reader.duration, cut_at=cut_at)
    batches: list[list[tuple[float, float]]] = []
    for piece in pieces:
        if not batches or piece[1] - batches[-1][0][0] > BATCH_SECONDS:
            batches.append([])
        batches[-1].append(piece)
    allowed = [lang["code"] for lang in report.languages]
    per_piece_language = fixed_language is None and report.multilingual
    if per_piece_language:
        status(f"Several languages found ({', '.join(allowed)}): labelling each passage separately.")
    total = sum(b - a for a, b in pieces)
    progress = Progress("Transcribing", total)
    progress(0.0)  # the first piece can take a while (model warm-up); say what is happening
    done = 0.0
    segments: list[Segment] = []
    resumed = 0
    for i, batch in enumerate(batches):
        path = workdir / f"speech{suffix}_{i:04d}_{batch[0][0]:.2f}_{batch[-1][1]:.2f}.json"
        if use_cache and path.exists():
            resumed += 1
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = []
            for a, b in batch:
                audio = reader.read(a, b)  # only this piece is in memory
                lang = fixed_language or report.main
                if per_piece_language:
                    lang = _restrict(model.detect(audio), allowed)
                segs, _ = model.transcribe(audio, lang, offset=a)
                raw += [s.to_dict(words=True) for s in segs]
                done += b - a
                progress(done)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        done = max(done, sum(b - a for bt in batches[: i + 1] for a, b in bt))
        progress(done)
        segments += [Segment.from_dict(d) for d in raw]
    progress.finish()
    if resumed:
        status(f"Reused {resumed} of {len(batches)} cached chunk(s).")
    return segments


SONG_GAP = 4.0  # sung stretches closer than this are transcribed as one
SONG_PAD = 0.5  # seconds kept before and after each stretch


def song_clips(voice: list[tuple[float, float]], duration: float) -> list[tuple[float, float]] | None:
    """Where Whisper should listen on separated vocals: the sung stretches, padded and merged.

    Started on an instrumental intro, Whisper tends to write a subtitle credit there and skip
    the first sung lines; starting where the singing starts avoids that. None: everything.
    """
    if not voice:
        return None
    merged = vad.merge_regions(sorted(voice), SONG_GAP)
    return [(max(0.0, a - SONG_PAD), min(duration, b + SONG_PAD)) for a, b in merged]


LOST_MIN = 1.5  # shorter stretches are breaths and ad-libs, not lost lines
LOST_COVERED = 0.5  # a stretch counts as lost when lines cover less than this fraction of it


def _overlap(a: float, b: float, spans) -> float:
    return sum(max(0.0, min(b, e) - max(a, s)) for s, e in spans)


def lost_stretches(voice: list[tuple[float, float]], lines_: list[Segment], duration: float) -> list[tuple[float, float]]:
    """Sung stretches that no transcribed line covers, padded and merged, to transcribe again.

    Inside a whole song Whisper sometimes skips a passage (it writes a made-up line and jumps
    past the singing before it); on its own the passage usually comes out right.
    """
    spans = [(ln.start, ln.end) for ln in lines_]
    lost = [(a, b) for a, b in sorted(voice)
            if b - a >= LOST_MIN and _overlap(a, b, spans) < LOST_COVERED * (b - a)]
    return song_clips(lost, duration) or []


def add_missing(lines_: list[Segment], extra: list[Segment]) -> list[Segment]:
    """Lines of the second pass that fill gaps: those mostly outside the lines already there."""
    spans = [(ln.start, ln.end) for ln in lines_]
    new = [ln for ln in extra if _overlap(ln.start, ln.end, spans) < LOST_COVERED * max(ln.duration, 1e-6)]
    return sorted(lines_ + new, key=lambda ln: ln.start)


def transcribe_lost(model, reader, stretches, language: str, workdir: Path, use_cache: bool, suffix: str = ""):
    """Second pass: each lost stretch transcribed on its own. Cached with the stretches it covers."""
    path = workdir / f"song-lost{suffix}.json"
    key = [[round(a, 2), round(b, 2)] for a, b in stretches]
    if use_cache and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("stretches") == key:
                return [Segment.from_dict(d) for d in cached["segments"]]
        except ValueError:
            pass
    where = ", ".join(f"{format_seconds(a)}-{format_seconds(b)}" for a, b in stretches)
    status(f"Listening again to {len(stretches)} stretch(es) of singing no line covers: {where}")
    out: list[Segment] = []
    for a, b in stretches:
        segs, _ = model.transcribe(reader.read(a, b), language, offset=a)
        out += segs
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"stretches": key, "segments": [s.to_dict(words=True) for s in out]},
                              ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return out


def song_lines(kept: list[Segment]) -> list[Segment]:
    segments = lines.split_lines(kept)
    for line in segments:  # confidence per sung line, not per Whisper segment
        hallucination.score(line)
    return hallucination.collapse_line_loops(segments)


def transcribe_song(model, reader, voice, language: str, separated: bool, workdir: Path, use_cache: bool,
                    suffix: str = ""):
    path = workdir / f"song-clips{suffix}.json"  # not song.json: earlier results came from the whole audio

    def compute():
        audio = reader.read()
        clips = song_clips(voice, reader.duration) if separated else None
        status(f"Transcribing {format_seconds(reader.duration)} of singing...")
        segs, _ = model.transcribe(audio, language, clips=clips, show_progress=True)
        return [s.to_dict(words=True) for s in segs]

    return [Segment.from_dict(d) for d in _json_cache(path, compute, use_cache)]


def run(opts: Options) -> Result:
    from . import whisper

    src = open_input(opts.input, opts.start, opts.end, use_cache=opts.use_cache, ipv4=opts.ipv4)
    tags = read_wav_tags(src.path)
    separated = _is_separated(tags)
    wav = src.cached_wav(SR, 1, use_cache=opts.use_cache)
    # The general model detects the language (and keys the cache); a language's own model, if
    # one is used, only transcribes, and its results are cached under its name.
    detector = "large-v3" if opts.model in ("auto", "kb-whisper") else opts.model
    workdir = cache_dir("derived", "transcribe", src.key(detector))

    with WavReader(wav) as reader:
        if reader.duration < 0.5:
            raise AudioError("no audio in the selected range")
        status(f"Audio: {format_seconds(reader.duration)}" + (" (separated vocals)" if separated else ""))

        # Voice activity, block by block.
        vad_model = vad.SileroVAD()
        vad_progress = Progress("Finding voice", reader.duration) if reader.duration > 1200 else None
        voice = [tuple(r) for r in _json_cache(
            workdir / "vad.json",
            lambda: vad_model.regions(reader, (lambda d, t: vad_progress(d)) if vad_progress else None),
            opts.use_cache,
        )]
        if vad_progress:
            vad_progress.finish()

        if opts.mode == "auto":
            decision = detect_mode(reader, separated)
            chosen, reason = decision.mode, decision.reason
            status(f"Mode: {chosen} (auto: {reason}; override with --mode)")
        else:
            chosen, reason = opts.mode, "set with --mode"
            status(f"Mode: {chosen}")
        if chosen == "song" and not separated:
            status("Warning: this looks like a full mix; lyrics come out better on separated vocals: "
                   f"vocals {opts.input!r} | transcribe -")
        if chosen == "speech" and not voice:
            raise AudioError("no speech found in the selected range")
        if chosen == "speech" and opts.speakers is False and opts.num_speakers:
            opts.speakers = True

        model = None
        if opts.language:
            report = langdetect.LanguageReport(opts.language, [{"code": opts.language, "share": 1.0, "windows": 0}])
            status(f"Language: {opts.language} (set with --language)")
        else:
            model = whisper.load(detector, opts.device)
            # Songs on a full mix: VAD misses much of the singing, so detection windows use the whole range.
            det_voice = voice if (chosen == "speech" or separated) else [(0.0, reader.duration)]
            report = detect_language(model, reader, det_voice, workdir / "languages.json", opts.use_cache)
            others = ", ".join(f"{lang['code']} ({lang['share']:.0%})" for lang in report.languages[1:])
            status(f"Language: {report.main} ({report.languages[0]['share']:.0%})" + (f"; also found: {others}" if others else ""))

        per_passage = chosen == "speech" and opts.language is None and report.multilingual
        name = whisper.model_for(opts.model, report.main, per_passage)
        if model is None or name != detector:
            if opts.model == "auto" and name != "large-v3":
                status(f"Transcribing with {whisper.DESCRIPTIONS[name]} (--model large-v3 to use the general model)")
            model = None  # one large model in memory at a time
            model = whisper.load(name, opts.device)
        backend = f"{model.name}:{name}"
        suffix = "" if name == detector else f"-{name}"

        if chosen == "song" and separated:
            voice = vad.energy_regions(reader)  # Silero misses sung notes; the vocal stem's level does not
        if chosen == "song":
            raw = transcribe_song(model, reader, voice, report.main, separated, workdir, opts.use_cache, suffix)
            kept, dropped = hallucination.filter_segments(raw, voice if separated else None)
            segments = song_lines(kept)
            lost = lost_stretches(voice, segments, reader.duration) if separated else []
            if lost:
                extra = transcribe_lost(model, reader, lost, report.main, workdir, opts.use_cache, suffix)
                kept2, dropped2 = hallucination.filter_segments(extra, voice)
                before = len(segments)
                segments = add_missing(segments, song_lines(kept2))
                dropped += dropped2
                status(f"Second pass: {len(segments) - before} line(s) added.")
        else:
            raw = transcribe_speech(model, reader, voice, report, workdir, opts.use_cache, opts.language, suffix)
            segments, dropped = hallucination.filter_segments(raw, voice)

        names: list[str] = []
        if opts.speakers:
            if chosen != "speech":
                status("Note: --speakers is only used in speech mode; ignoring it.")
            else:
                from . import diarize

                turns = diarize.pyannote_turns(wav, opts.num_speakers, opts.device)
                if turns is None:
                    status("Diarizing with SpeechBrain ECAPA embeddings...")
                    turns = diarize.ecapa_turns(reader, voice, opts.num_speakers, opts.device)
                segments = speakers.split_at_turns(segments, turns)
                speakers.assign_speakers(segments, turns)
                names = speakers.name_speakers(segments, opts.names)
                status(f"Speakers: {len(names)} ({', '.join(names)})")
        duration = reader.duration

    # Timestamps in the original timeline: a --start offset is added back, and so is the place
    # in the recording of a section that vocals separated (its WAV says where it begins).
    shift = src.timeline_start
    if shift:
        for seg in segments:
            seg.start += shift
            seg.end += shift
            for w in seg.words:
                w.start += shift
                w.end += shift
    if dropped:
        reasons: dict[str, int] = {}
        for _, why in dropped:
            reasons[why] = reasons.get(why, 0) + 1
        status("Dropped " + ", ".join(f"{n} segment(s): {why}" for why, n in reasons.items()))
    low = sum(s.low_confidence for s in segments)
    if low:
        status(f"{low} line(s) marked [?] as low confidence.")
    return Result(src, chosen, reason, separated, report, segments, len(dropped), names, duration, backend,
                  title=tags.get("INAM") or None)
