"""`accent <file | YouTube URL | ->`: top predicted English accents with probabilities."""

from __future__ import annotations

import argparse
import json
import sys

from ingest.audio import WavReader
from ingest.cli import add_input_args, apply_common, report_progress, run, status
from ingest.errors import AudioError
from ingest.source import open_input
from ingest.timeparse import format_seconds

from . import __version__
from .chunks import SAMPLE_RATE, make_chunks
from .classify import MODELS, AccentClassifier, average_probabilities, display_name, top_k

YOUTUBE_CAP_SECONDS = 600.0  # YouTube without a range: first 10 minutes unless --full
NOTE = (
    "Note: this is a statistical guess over a fixed set of 16 English accent labels; "
    "mixed, regional or non-native accents are often misclassified."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="accent", description="Guess the English accent of the speaker in a recording.")
    add_input_args(parser)
    parser.add_argument("--model", choices=sorted(MODELS), default="ecapa",
                        help="ecapa: ECAPA-TDNN, small and fast (default); xlsr: larger wav2vec2 XLSR model")
    parser.add_argument("--full", action="store_true",
                        help=f"YouTube: analyse the whole video, not just the first {YOUTUBE_CAP_SECONDS / 60:.0f} minutes")
    parser.add_argument("-v", "--verbose", action="store_true", help="also print a per-chunk table")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON on stdout")
    parser.add_argument("--top", type=int, default=3, metavar="N", help="number of accents to show (default 3)")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def analyse(args: argparse.Namespace) -> dict:
    if args.top < 1:
        raise ValueError("--top must be at least 1")
    cap = None if args.full else YOUTUBE_CAP_SECONDS
    src = open_input(args.input, args.start, args.end, cap=cap, use_cache=not args.no_cache, ipv4=args.ipv4)
    if src.capped:
        status(f"Analysing the first {YOUTUBE_CAP_SECONDS / 60:.0f} minutes only; use --full for the whole video.")

    wav = src.cached_wav(SAMPLE_RATE, 1, use_cache=not args.no_cache)
    with WavReader(wav) as reader:
        samples = reader.read()
    if samples.size < SAMPLE_RATE // 2:
        raise AudioError("no audio in the selected range")

    chunks = make_chunks(samples)
    speech = [c for c in chunks if not c.silent]
    if not speech:
        raise AudioError("no speech found: every chunk is below the silence threshold")

    classifier = AccentClassifier(args.model, args.device)
    status(f"Loading {args.model} model on {classifier.device}...")
    classifier.load()
    status(f"Classifying {len(speech)} chunk(s) of ~20 s ({len(chunks) - len(speech)} silent skipped)...")

    offset = src.start
    rows, per_chunk = [], []
    for n, chunk in enumerate(chunks):
        report_progress("Recognising the accent", n / len(chunks))
        row = {
            "index": chunk.index + 1,
            "start": round(offset + chunk.start, 2),
            "end": round(offset + chunk.end, 2),
            "rms_dbfs": None if chunk.rms_dbfs == float("-inf") else round(chunk.rms_dbfs, 1),
            "speech_seconds": round(chunk.speech_seconds, 2),
            "silent": chunk.silent,
        }
        if not chunk.silent:
            probs = classifier.predict(chunk.speech(samples))
            per_chunk.append(probs)
            label, p = top_k(probs, classifier.labels, 1)[0]
            row.update(top_label=label, top_name=display_name(label), top_probability=round(p, 4))
        rows.append(row)

    mean = average_probabilities(per_chunk, weights=[c.speech_seconds for c in speech])
    return {
        "source": src.describe(),
        "model": args.model,
        "model_id": classifier.spec.source,
        "device": classifier.device,
        "top": [{"label": lab, "name": display_name(lab), "probability": round(p, 4)}
                for lab, p in top_k(mean, classifier.labels, args.top)],
        "probabilities": {lab: round(float(p), 4) for lab, p in zip(classifier.labels, mean, strict=True)},
        "chunks": rows,
        "summary": {
            "chunks": len(chunks),
            "silent_chunks": len(chunks) - len(speech),
            "analysed_seconds": round(sum(c.speech_seconds for c in speech), 2),
        },
        "note": NOTE,
    }


def render_text(result: dict, verbose: bool) -> str:
    summary = result["summary"]
    lines = [
        f"Top accents ({format_seconds(summary['analysed_seconds'])} of speech in "
        f"{summary['chunks'] - summary['silent_chunks']} chunk(s), model {result['model']}):"
    ]
    width = max(len(t["name"]) for t in result["top"])
    for rank, t in enumerate(result["top"], 1):
        lines.append(f"  {rank}. {t['name']:<{width}}  {t['probability'] * 100:5.1f}%")
    if verbose:
        lines += ["", f"  {'#':>3}  {'Time':<17}  {'Level':>7}  {'Speech':>6}  Top accent"]
        for row in result["chunks"]:
            span = f"{format_seconds(row['start'])}-{format_seconds(row['end'])}"
            level = "-inf dB" if row["rms_dbfs"] is None else f"{row['rms_dbfs']:.0f} dB"
            detail = "(silent, skipped)" if row["silent"] else f"{row['top_name']} {row['top_probability'] * 100:.0f}%"
            lines.append(f"  {row['index']:>3}  {span:<17}  {level:>7}  {row['speech_seconds']:>4.0f} s  {detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply_common(args)

    def go() -> int:
        result = analyse(args)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(render_text(result, args.verbose))
        status(NOTE)
        return 0

    return run(go, "accent")


if __name__ == "__main__":
    sys.exit(main())
