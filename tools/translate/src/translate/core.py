"""Translate a Document: resolve source languages, chunk, call the backend, fill in translations."""

from __future__ import annotations

import sys
from dataclasses import replace

from ingest.cli import progress_wanted, report_partial, report_progress

from . import languages, render
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


def breton_first(todo: list, target: str, breton, progress: bool) -> list[tuple]:
    """Breton lines into French with the Breton model.

    Returns (unit, French copy) pairs for the backend to take on into the target; into French
    the model's output already is the translation, and nothing is left.
    """
    lines = [u for u in todo if u.language == "br"]
    if progress:
        then = "" if target == "fr" else f", then from French into {languages.name(target)}"
        status(f"Breton: translating {len(lines)} line(s) into French first with {breton.name}{then}")
    on_progress = (lambda f: report_progress("Translating from Breton", f)) if progress_wanted() else None
    french = breton([u.text for u in lines], on_progress)
    if target == "fr":
        for unit, text in zip(lines, french, strict=True):
            unit.translation = text
        return []
    return [(unit, replace(unit, text=text, language="fr")) for unit, text in zip(lines, french, strict=True)]


def fill_in(doc: Document, units: list, translations: list[str], shown: int) -> int:
    """Put a chunk's translations in and report its lines, with the input's lines before them.

    `shown`: the input's index up to which lines are reported already; returns the new one.
    Lines in between that are not in the chunk are blank or already in the target language.
    """
    for unit, text in zip(units, translations, strict=True):
        unit.translation = text
    upto = max(u.index for u in units) + 1
    own = {u.index: u for u in units}  # Breton lines: their French copies hold the translation
    report_partial(render.translation_lines([own.get(u.index, u) for u in doc.units if shown <= u.index < upto]))
    return max(shown, upto)


def translate_document(doc: Document, translator, target: str, source: str | None = None,
                       max_chars: int = MAX_CHARS, progress: bool = True, breton=None) -> str | None:
    """Fill in each unit's translation; returns the main source language.

    `breton` makes the Breton-French model (called only when there are Breton lines).
    """
    main = resolve_languages(doc, source, target)
    needs_backend = any(u.language != target and not (u.language == "br" and target == "fr") for u in doc.content())
    if needs_backend and hasattr(translator, "prepare"):
        translator.prepare()  # fail now, with its hint, rather than after the Breton step
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
    if target == "br":
        raise ValueError("translating into Breton is not possible yet: only from Breton")
    pairs: list[tuple] = []
    if any(u.language == "br" for u in todo):
        if breton is None:
            from .breton import BretonToFrench

            breton = BretonToFrench
        model = breton()
        doc.via = model.name
        pairs = breton_first(todo, target, model, progress)
        french = {id(unit): copy for unit, copy in pairs}
        todo = [french.get(id(u), u) for u in todo if u.language != "br" or id(u) in french]
    if not todo:
        return main
    doc.backend_used = True
    sub = Document(doc.kind, todo, mode=doc.mode, language="fr" if main == "br" else main)
    chunks = make_chunks(sub, max_chars=max_chars, whole=(doc.mode == "song" and translator.name == "llm"))
    total, done, shown = sum(len(c.units) for c in chunks), 0, 0
    for i, chunk in enumerate(chunks, 1):
        if progress and len(chunks) > 1:
            status(f"Translating part {i}/{len(chunks)} ({len(chunk.units)} line(s))...")
        if progress_wanted() and hasattr(translator, "on_progress"):
            base, size = done, len(chunk.units)
            translator.on_progress = lambda f, base=base, size=size: report_progress(
                "Translating", (base + min(1.0, f) * size) / total)
        out = translator.translate(chunk, sub.language, target, song=doc.mode == "song")
        done += len(chunk.units)
        report_progress("Translating", done / total)
        shown = fill_in(doc, chunk.units, out, shown)
    for unit, copy in pairs:
        unit.translation = copy.translation
    empty = sum(1 for u in doc.content() if not (u.translation or "").strip())
    if empty:
        status(f"Warning: {empty} line(s) came back without a translation.")
    return main
