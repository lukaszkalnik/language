import json

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
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
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
