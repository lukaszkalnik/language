"""`translate --to <lang> [file | -]`: translate text or transcribe JSON, line by line."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ingest.cli import run

from . import __version__, languages, render
from .core import translate_document
from .document import parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translate",
        description="Translate text or `transcribe --json` output locally. Reads a file or stdin.",
    )
    parser.add_argument("input", nargs="?", default="-", help="text or JSON file; - or nothing for stdin")
    parser.add_argument("--to", required=True, metavar="LANG", help="target language, e.g. de, en, pl")
    parser.add_argument("--from", dest="source", metavar="LANG", help="source language (default: from the JSON, or detected)")
    parser.add_argument("--translator", choices=["llm", "nllb"], default="llm",
                        help="llm: a local model through Ollama (default); nllb: NLLB-200, offline, no Ollama")
    parser.add_argument("--llm-model", default=None, metavar="NAME", help="Ollama model (default: gemma4:12b)")
    parser.add_argument("--nllb-model", default="600m", metavar="NAME", help="600m (default) or 1.3b, or a HF model ID")
    out = parser.add_mutually_exclusive_group()
    out.add_argument("--only-translation", action="store_true", help="print just the translation")
    out.add_argument("--json", action="store_true", help="JSON keeping timestamps and speaker labels from the input")
    parser.add_argument("--width", type=int, default=None, help="side-by-side width (default: terminal width, or 140)")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="NLLB device")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def read_input(arg: str) -> str:
    if arg == "-":
        if sys.stdin.isatty():
            raise ValueError("nothing to translate: pass a file or pipe text in, e.g. `transcribe song.mp3 | translate --to de`")
        return sys.stdin.read()
    path = Path(arg).expanduser()
    if not path.is_file():
        raise ValueError(f"file not found: {arg}")
    return path.read_text(encoding="utf-8")


def make_translator(args):
    if args.translator == "nllb":
        from ingest.cli import choose_ipv4

        from .nllb import NLLBTranslator

        choose_ipv4()  # NLLB comes from the Hugging Face Hub; Ollama is local

        return NLLBTranslator(args.nllb_model, args.device)
    from .llm import DEFAULT_MODEL, OllamaTranslator

    return OllamaTranslator(args.llm_model or DEFAULT_MODEL)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def go() -> int:
        target = languages.normalise(args.to)
        source = languages.normalise(args.source) if args.source else None
        doc = parse(read_input(args.input))
        if not doc.content():
            raise ValueError("the input contains no text")
        translator = make_translator(args)
        main_src = translate_document(doc, translator, target, source)
        backend = f"{translator.name}:{getattr(translator, 'model', None) if args.translator == 'llm' else args.nllb_model}"
        if args.json:
            sys.stdout.write(render.to_json(doc, main_src, target, backend))
        elif args.only_translation:
            sys.stdout.write(render.only_translation(doc))
        else:
            width = args.width or (shutil.get_terminal_size((140, 24)).columns if sys.stdout.isatty() else 140)
            sys.stdout.write(render.side_by_side(doc, width))
        return 0

    return run(go, "translate")


if __name__ == "__main__":
    sys.exit(main())
