"""`vocals <file | YouTube URL | ->`: the vocals (or another stem) as a WAV at the source's sample rate."""

from __future__ import annotations

import argparse
import sys

from ingest.cli import add_input_args, apply_common, run, status
from ingest.source import audio_destination, emit_wav, open_input

from . import __version__
from .separate import DEFAULT_MODEL, MODELS, normalise_stem, output_name, separate


def build_parser() -> argparse.ArgumentParser:
    models = "; ".join(f"{m.name}: {m.note}" for m in MODELS.values())
    parser = argparse.ArgumentParser(
        prog="vocals",
        description="Separate the vocals from the music. Writes a WAV stream to stdout when piped, "
        "otherwise a file (-o, or '<name>.vocals.wav').",
    )
    add_input_args(parser)
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="write here (.wav, or .flac/.mp3/... via ffmpeg); '-' forces stdout")
    parser.add_argument("--stem", default="vocals",
                        help="vocals (default), instrumental (everything else), drums, bass, other; "
                        "htdemucs_6s also has guitar and piano")
    parser.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL, help=models)
    parser.add_argument("--shifts", type=int, default=1, metavar="N",
                        help="average N time-shifted passes: slightly better, N times slower (default 1)")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply_common(args)

    def go() -> int:
        stem = normalise_stem(args.stem, args.model)
        if args.shifts < 1:
            raise ValueError("--shifts must be at least 1")
        src = open_input(args.input, args.start, args.end, use_cache=not args.no_cache, ipv4=args.ipv4)
        paths = separate(src, args.model, args.device, use_cache=not args.no_cache, shifts=args.shifts)
        destination = audio_destination(args.output, sys.stdout.isatty(), output_name(src, stem))
        if destination is None:
            status(f"Writing {'instrumental' if stem == 'no_vocals' else stem} WAV to stdout")
        emit_wav(paths[stem], destination)
        return 0

    return run(go, "vocals")


if __name__ == "__main__":
    sys.exit(main())
