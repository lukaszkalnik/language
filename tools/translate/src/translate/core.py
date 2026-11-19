"""Translate a Document: resolve source languages, chunk, call the backend, fill in translations."""

from __future__ import annotations

import sys

from ingest.cli import progress_wanted, report_progress

from . import languages
from .chunks import MAX_CHARS, make_chunks
from .document import Document


def status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def resolve_languages(doc: Document, source: str | None, target: str) -> str | None:
    """Set each unit's source language; return the main source language."""
    if source:
        for u in doc.content():
            u.language = source
        return source
    if doc.kind == "json" and (doc.language or any(u.language for u in doc.content())):
        main = doc.language or next(u.language for u in doc.content() if u.language)
        for u in doc.content():
            u.language = u.language or main
        return main
    text = "\n".join(u.text for u in doc.content())
    main = languages.detect(text)
    for u in doc.content():
        own = languages.detect(u.text) if len(u.text) >= 200 else None  # long paragraphs: their own language
        u.language = own or main
    return main


def translate_document(doc: Document, translator, target: str, source: str | None = None,
                       max_chars: int = MAX_CHARS, progress: bool = True) -> str | None:
    main = resolve_languages(doc, source, target)
    langs = sorted({u.language for u in doc.content() if u.language})
    if progress:
        origin = "set with --from" if source else ("from the JSON" if doc.kind == "json" and main else "detected")
        status(f"Source: {', '.join(langs) or 'unknown'} ({origin}) -> {target}; {doc.mode or 'text'}, backend {translator.name}")
    if main is None and translator.name == "nllb":
        raise ValueError("could not detect the source language; pass --from xx")

    # Lines already in the target language are kept as they are.
    todo = []
    for u in doc.content():
        if u.language == target:
            u.translation = u.text
        else:
            todo.append(u)
    if not todo:
        status("Everything is already in the target language.")
        return main
    sub = Document(doc.kind, todo, mode=doc.mode, language=main)
    chunks = make_chunks(sub, max_chars=max_chars, whole=(doc.mode == "song" and translator.name == "llm"))
    total, done = sum(len(c.units) for c in chunks), 0
    for i, chunk in enumerate(chunks, 1):
        if progress and len(chunks) > 1:
            status(f"Translating part {i}/{len(chunks)} ({len(chunk.units)} line(s))...")
        if progress_wanted() and hasattr(translator, "on_progress"):
            base, size = done, len(chunk.units)
            translator.on_progress = lambda f, base=base, size=size: report_progress(
                "Translating", (base + min(1.0, f) * size) / total)
        out = translator.translate(chunk, main, target, song=doc.mode == "song")
        done += len(chunk.units)
        report_progress("Translating", done / total)
        for unit, text in zip(chunk.units, out, strict=True):
            unit.translation = text
    empty = sum(1 for u in doc.content() if not (u.translation or "").strip())
    if empty:
        status(f"Warning: {empty} line(s) came back without a translation.")
    return main
