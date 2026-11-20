"""`transcribe <file | YouTube URL | ->`: language detection and transcription of speech or singing."""

from __future__ import annotations

import argparse
import sys

from ingest.cli import add_input_args, apply_common, run, status

from . import __version__, formats
from .pipeline import Options
from .speakers import parse_names

FORMATS = ("text", "lrc", "srt", "vtt", "json", "md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Detect the language and transcribe speech or singing with Whisper large-v3 "
        "(KB-Whisper for Swedish). "
        "Transcript on stdout; progress and notes on stderr.",
    )
    add_input_args(parser)
    parser.add_argument("--mode", choices=["auto", "song", "speech"], default="auto",
                        help="song: one sung line per line; speech: sentences in paragraphs; auto (default): guess")
    parser.add_argument("--language", metavar="XX", help="skip detection, e.g. en, de, pl, fr")
    parser.add_argument("--model", choices=["auto", "large-v3", "turbo", "kb-whisper"], default="auto",
                        help="auto (default): KB-Whisper for Swedish, Whisper large-v3 for everything else; "
                        "large-v3; turbo (large-v3-turbo, faster); kb-whisper (KB-Whisper, Swedish)")
    out = parser.add_mutually_exclusive_group()
    out.add_argument("--lrc", dest="format", action="store_const", const="lrc", help="LRC lyrics with timestamps (song mode)")
    out.add_argument("--srt", dest="format", action="store_const", const="srt", help="SRT subtitles")
    out.add_argument("--vtt", dest="format", action="store_const", const="vtt", help="WebVTT subtitles")
    out.add_argument("--json", dest="format", action="store_const", const="json",
                     help="JSON with timestamps, language, confidence and speaker per segment")
    out.add_argument("--md", dest="format", action="store_const", const="md",
                     help="readable Markdown transcript with timestamps every few minutes (speech mode)")
    parser.add_argument("--words", action="store_true", help="--json: include word timestamps")
    parser.add_argument("--no-marks", action="store_true", help="do not mark low-confidence lines with [?]")
    parser.add_argument("--speakers", action="store_true", help="speech mode: label speakers (diarization)")
    parser.add_argument("--num-speakers", type=int, metavar="N", help="number of speakers, if known (implies --speakers)")
    parser.add_argument("--names", metavar="A,B", help='speaker names in order of first appearance, e.g. "Anna,Ben"')
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.set_defaults(format="text")
    return parser


def render(result, fmt: str, words: bool = False, mark_low: bool = True) -> str:
    segs, meta = result.segments, result.meta()
    if fmt == "json":
        return formats.to_json(segs, meta, words)
    if fmt == "lrc":
        return formats.to_lrc(segs, meta.get("title"), meta.get("language"), mark_low)
    if fmt == "srt":
        return formats.to_srt(segs, mark_low)
    if fmt == "vtt":
        return formats.to_vtt(segs, mark_low)
    if fmt == "md":
        return formats.to_markdown(segs, meta, mark_low)
    return formats.to_text(segs, result.mode, mark_low)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply_common(args)

    def go() -> int:
        from .pipeline import run as run_pipeline

        if args.num_speakers is not None and args.num_speakers < 1:
            raise ValueError("--num-speakers must be at least 1")
        names = parse_names(args.names)
        if args.format == "lrc" and args.mode == "speech":
            status("Note: --lrc is meant for songs; writing LRC anyway.")
        if args.format == "md" and args.mode == "song":
            status("Note: --md is meant for speech; writing Markdown anyway.")
        opts = Options(
            input=args.input, start=args.start, end=args.end, mode=args.mode,
            language=args.language.lower() if args.language else None, model=args.model,
            speakers=args.speakers or bool(args.num_speakers) or bool(names), num_speakers=args.num_speakers,
            names=names, use_cache=not args.no_cache, ipv4=args.ipv4, device=args.device,
        )
        result = run_pipeline(opts)
        if not result.segments:
            status("No words were recognised.")
        sys.stdout.write(render(result, args.format, args.words, not args.no_marks))
        sys.stdout.flush()
        return 0

    return run(go, "transcribe")


if __name__ == "__main__":
    sys.exit(main())
