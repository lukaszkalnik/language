import itertools
import json
from typing import ClassVar

import pytest

from translate import render
from translate.chunks import make_chunks
from translate.core import translate_document
from translate.document import looks_like_json, parse
from translate.llm import build_messages, parse_reply


class FakeTranslator:
    """Upper-cases each line and records the chunks it saw."""

    name = "llm"

    def __init__(self):
        self.calls = []

    def translate(self, chunk, source, target, song=False):
        self.calls.append((chunk, source, song))
        return [f"{u.text.upper()}" for u in chunk.units]


TRANSCRIBE_JSON = {
    "mode": "speech", "language": "en",
    "segments": [
        {"id": 1, "start": 0.0, "end": 2.5, "text": "Hello and welcome.", "language": "en", "speaker": "Anna", "paragraph": 0},
        {"id": 2, "start": 2.6, "end": 5.0, "text": "Glad to be here.", "language": "en", "speaker": "Ben", "paragraph": 1},
        {"id": 3, "start": 5.2, "end": 7.0, "text": "Dzień dobry wszystkim.", "language": "pl", "speaker": "Ben", "paragraph": 1},
    ],
}


# --- input detection ----------------------------------------------------------------

def test_detects_transcribe_json():
    assert looks_like_json(json.dumps(TRANSCRIBE_JSON))
    doc = parse(json.dumps(TRANSCRIBE_JSON))
    assert doc.kind == "json" and doc.mode == "speech" and len(doc.content()) == 3


@pytest.mark.parametrize("text", ["{not json at all", "Just lyrics\n{like this}", '{"other": 1}', "[1, 2]"])
def test_other_text_is_plain(text):
    assert not looks_like_json(text)
    assert parse(text).kind == "text"


def test_plain_text_lines_numbering_and_blank_lines():
    doc = parse("Line one\nLine two\n\nLine three\n\n")
    assert [u.number for u in doc.units] == [1, 2, None, 3]
    assert doc.mode == "song"


def test_speaker_prefixes_only_when_most_lines_have_them():
    speech = parse("Speaker 1: Hello there.\n\nSpeaker 2: Hi!\n\nSpeaker 1: How are you?")
    assert [u.speaker for u in speech.content()] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    assert speech.content()[0].text == "Hello there." and speech.mode == "speech"
    lyrics = parse("Baby: come back\nI miss you so\nEvery night\nI cry")
    assert all(u.speaker is None for u in lyrics.content())


# --- line alignment ---------------------------------------------------------------

def test_side_by_side_keeps_numbering_aligned():
    doc = parse("Short\nA much longer line that will need wrapping in a narrow layout\nEnd")
    translate_document(doc, FakeTranslator(), "de", source="en", progress=False)
    out = render.side_by_side(doc, width=60)
    rows = out.splitlines()
    assert rows[0].startswith("1  Short") and rows[0].endswith("SHORT")
    numbered = [r for r in rows if r[:1].strip()]
    assert [r.split()[0] for r in numbered] == ["1", "2", "3"]
    assert all(" │ " in r for r in rows)
    sep = {r.index(" │ ") for r in rows}
    assert len(sep) == 1  # one column boundary for every row


def test_only_translation_keeps_blank_lines_and_speakers():
    doc = parse("Speaker 1: one.\n\nSpeaker 2: two.")
    translate_document(doc, FakeTranslator(), "de", source="en", progress=False)
    assert render.only_translation(doc) == "Speaker 1: ONE.\n\nSpeaker 2: TWO.\n"


def test_parse_reply_tolerates_formats_and_ignores_extras():
    reply = "Here you go:\n1| Eins\n2) Zwei\n[3] Drei\n4: Vier\n9| out of range"
    assert parse_reply(reply, 4) == {1: "Eins", 2: "Zwei", 3: "Drei", 4: "Vier"}


def test_parse_reply_skips_echoed_source_lines():
    reply = "1| Alors on danse\n1| Also tanzen wir\n2| Merci\n2| Danke"
    assert parse_reply(reply, 2, ["Alors on danse", "Merci"]) == {1: "Also tanzen wir", 2: "Danke"}
    # A line that really stays the same (a name) is still accepted.
    assert parse_reply("1| Stromae", 1, ["Stromae"]) == {1: "Stromae"}


def test_prompt_numbers_every_line_and_asks_for_faithful_translation():
    doc = parse("Alors on danse\nQui dit étude dit travail")
    (chunk,) = make_chunks(doc)
    msgs = build_messages(chunk, "fr", "de", song=True)
    assert "1| Alors on danse\n2| Qui dit étude dit travail" in msgs[1]["content"]
    assert "not a poetic" in msgs[0]["content"] and "German" in msgs[0]["content"]


# --- JSON: languages, speakers, timestamps ------------------------------------------

def test_json_keeps_timestamps_speakers_and_per_segment_language():
    doc = parse(json.dumps(TRANSCRIBE_JSON))
    fake = FakeTranslator()
    main = translate_document(doc, fake, "pl", progress=False)
    assert main == "en"
    data = json.loads(render.to_json(doc, main, "pl", "llm:test"))
    segs = data["segments"]
    assert [s["start"] for s in segs] == [0.0, 2.6, 5.2]
    assert [s["speaker"] for s in segs] == ["Anna", "Ben", "Ben"]
    assert segs[0]["translation"] == "HELLO AND WELCOME."
    assert segs[2]["translation"] == "Dzień dobry wszystkim."  # already Polish: kept
    assert [s["source_language"] for s in segs] == ["en", "en", "pl"]
    assert data["translation"]["target_language"] == "pl"


def test_from_overrides_json_language():
    doc = parse(json.dumps(TRANSCRIBE_JSON))
    translate_document(doc, FakeTranslator(), "pl", source="de", progress=False)
    assert all(u.language == "de" for u in doc.content())


# --- chunking of long inputs --------------------------------------------------------

def long_podcast(paragraphs=30, per=3):
    segs, t, i = [], 0.0, 0
    for p in range(paragraphs):
        for _ in range(per):
            i += 1
            segs.append({"id": i, "start": t, "end": t + 5, "text": f"Sentence {i} " + "word " * 25,
                         "language": "en", "speaker": f"S{p % 2}", "paragraph": p})
            t += 5.5
    return {"mode": "speech", "language": "en", "segments": segs}


def test_chunks_hold_whole_paragraphs_with_overlapping_context():
    doc = parse(json.dumps(long_podcast()))
    chunks = make_chunks(doc, max_chars=1200, overlap=2)
    assert len(chunks) > 3
    for prev, nxt in itertools.pairwise(chunks):
        assert prev.units[-1].paragraph != nxt.units[0].paragraph  # cut between paragraphs
        assert nxt.context == prev.units[-2:]  # overlap: the end of the previous chunk
    assert sum(len(c.units) for c in chunks) == 90


def test_chunked_translation_keeps_speakers_and_timestamps():
    data = long_podcast()
    doc = parse(json.dumps(data))
    fake = FakeTranslator()
    translate_document(doc, fake, "de", max_chars=1200, progress=False)
    assert len(fake.calls) > 3
    out = json.loads(render.to_json(doc, "en", "de", "x"))
    for before, after in zip(data["segments"], out["segments"], strict=True):
        assert (after["start"], after["end"], after["speaker"]) == (before["start"], before["end"], before["speaker"])
        assert after["translation"] == before["text"].strip().upper()


def test_context_carries_previous_translations():
    doc = parse(json.dumps(long_podcast(paragraphs=10)))
    fake = FakeTranslator()
    translate_document(doc, fake, "de", max_chars=600, progress=False)
    second_chunk = fake.calls[1][0]
    assert second_chunk.context and all(u.translation for u in second_chunk.context)


def test_songs_go_in_one_request():
    doc = parse("\n".join(f"Line number {i}" for i in range(60)))
    fake = FakeTranslator()
    translate_document(doc, fake, "de", source="fr", max_chars=100, progress=False)
    assert len(fake.calls) == 1 and fake.calls[0][2] is True  # whole song, song flag set


def test_plain_text_language_detection():
    doc = parse("Ceci est une phrase en français.\nEt voici une autre phrase, toujours en français.")
    main = translate_document(doc, FakeTranslator(), "de", progress=False)
    assert main == "fr"


# --- progress for the desktop window ------------------------------------------------

def test_progress_lines_follow_translated_lines(monkeypatch, capsys):
    from ingest import cli

    class Reporting(FakeTranslator):
        on_progress = None

        def translate(self, chunk, source, target, song=False):
            self.on_progress(0.5)
            return super().translate(chunk, source, target, song)

    monkeypatch.setenv(cli.PROGRESS_ENV, "1")
    monkeypatch.setattr(cli, "_last_progress", {})
    doc = parse("one\n\ntwo\n\nthree\n\nfour")
    translate_document(doc, Reporting(), "de", source="en", max_chars=8, progress=False)
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("@progress")]
    fractions = [float(ln.split()[1]) for ln in lines]
    assert 0 < fractions[0] < 1 and fractions[-1] == 1.0  # partway through the chunk, then done
    assert fractions == sorted(fractions)


# --- Breton: into French first --------------------------------------------------------

class FakeBreton:
    """Stands in for the Breton-French model: marks each line as French."""

    name = "m2m100-br-fr"
    seen: ClassVar[list] = []

    def __call__(self, texts, on_progress=None):
        FakeBreton.seen = list(texts)
        return [f"fr:{t}" for t in texts]


BRETON_JSON = {
    "mode": "speech", "language": "br",
    "segments": [
        {"id": 1, "start": 0.0, "end": 2.0, "text": "Demat deoc'h.", "language": "br"},
        {"id": 2, "start": 2.0, "end": 4.0, "text": "Good morning.", "language": "en"},
        {"id": 3, "start": 4.0, "end": 6.0, "text": "Penaos 'mañ kont ?", "language": "br"},
    ],
}


def test_breton_goes_through_french_to_other_languages():
    doc = parse(json.dumps(BRETON_JSON))
    fake = FakeTranslator()
    translate_document(doc, fake, "de", progress=False, breton=FakeBreton)
    assert FakeBreton.seen == ["Demat deoc'h.", "Penaos 'mañ kont ?"]
    sent = [u for chunk, _, _ in fake.calls for u in chunk.units]
    assert [u.text for u in sent] == ["fr:Demat deoc'h.", "Good morning.", "fr:Penaos 'mañ kont ?"]
    assert [u.language for u in sent] == ["fr", "en", "fr"]
    assert fake.calls[0][1] == "fr"  # the backend is told the source is French, not Breton
    assert [u.translation for u in doc.content()] == ["FR:DEMAT DEOC'H.", "GOOD MORNING.", "FR:PENAOS 'MAÑ KONT ?"]
    assert next(u.text for u in doc.content()) == "Demat deoc'h."  # the original line is kept
    assert doc.via == "m2m100-br-fr" and doc.backend_used


def test_breton_into_french_needs_only_the_breton_model():
    data = dict(BRETON_JSON, segments=[s for s in BRETON_JSON["segments"] if s["language"] == "br"])
    doc = parse(json.dumps(data))
    fake = FakeTranslator()
    translate_document(doc, fake, "fr", progress=False, breton=FakeBreton)
    assert not fake.calls and not doc.backend_used
    assert [u.translation for u in doc.content()] == ["fr:Demat deoc'h.", "fr:Penaos 'mañ kont ?"]


def test_no_breton_lines_no_breton_model():
    doc = parse(json.dumps(TRANSCRIBE_JSON))

    def no_model():
        raise AssertionError("the Breton model must not be loaded")

    translate_document(doc, FakeTranslator(), "de", progress=False, breton=no_model)
    assert doc.via is None


def test_into_breton_is_refused_and_breton_is_not_an_nllb_language():
    from translate import languages

    with pytest.raises(ValueError, match="into Breton"):
        translate_document(parse("Hello there, how are you?"), FakeTranslator(), "br", source="en", progress=False)
    assert languages.normalise("Breton") == "br"
    with pytest.raises(ValueError, match="NLLB"):
        languages.nllb_code("br")


def test_breton_fillers_are_left_out_before_translating():
    from translate.breton import without_fillers

    assert without_fillers("peogwir euh me oa bet ganet euh ba Plouven, er bloavezh") == \
        "peogwir me oa bet ganet ba Plouven, er bloavezh"
    assert without_fillers("Hag euh… e oa") == "Hag e oa"
    assert without_fillers("Euh, setu.") == "setu."
    assert without_fillers("Heuliañ an hent") == "Heuliañ an hent"  # words starting like a filler stay


def test_the_backend_is_set_up_only_when_a_line_needs_it():
    from translate.cli import Deferred

    made = []
    lazy = Deferred("llm", lambda: made.append(1) or FakeTranslator(), "gemma4:12b")
    data = dict(BRETON_JSON, segments=[s for s in BRETON_JSON["segments"] if s["language"] == "br"])
    translate_document(parse(json.dumps(data)), lazy, "fr", progress=False, breton=FakeBreton)
    assert not made  # Breton into French: no Ollama needed
    translate_document(parse(json.dumps(BRETON_JSON)), lazy, "de", progress=False, breton=FakeBreton)
    assert made == [1]
