import json

from helpers import seg, words

from transcribe import formats
from transcribe.speakers import assign_speakers, name_speakers, parse_names, split_at_turns


def song():
    return [seg("First line", 1.5, 4.0), seg("Second line", 65.25, 68.0, low_confidence=True)]


def test_lrc():
    out = formats.to_lrc(song(), title="Song", language="fr")
    assert out.splitlines() == ["[ti:Song]", "[la:fr]", "[re:transcribe (Whisper)]",
                                "[00:01.50]First line", "[01:05.25]Second line [?]"]


def test_srt():
    out = formats.to_srt([seg("Hi", 3661.5, 3663.25, speaker="Anna")])
    assert out == "1\n01:01:01,500 --> 01:01:03,250\nAnna: Hi\n"


def test_vtt():
    out = formats.to_vtt([seg("Hi", 1, 2, speaker="Ben"), seg("Yo", 2, 2)])
    lines = out.splitlines()
    assert lines[0] == "WEBVTT"
    assert "00:00:01.000 --> 00:00:02.000" in lines and "<v Ben>Hi" in lines
    assert "00:00:02.000 --> 00:00:02.300" in lines  # zero-length cues get a minimum length


def test_text_song_and_speech():
    assert formats.to_text(song(), "song") == "First line\nSecond line [?]\n"
    speech = [seg("Hello.", 0, 1, speaker="Speaker 1"), seg("Hi!", 5, 6, speaker="Speaker 2")]
    assert formats.to_text(speech, "speech") == "Speaker 1: Hello.\n\nSpeaker 2: Hi!\n"


def test_markdown_has_timestamps_every_few_minutes():
    segs = [seg(f"Paragraph {i}.", i * 100.0, i * 100.0 + 5) for i in range(8)]  # 0..700 s
    out = formats.to_markdown(segs, {"title": "Pod", "language": "en", "duration": 705, "speakers": []})
    assert out.startswith("# Pod\n")
    stamps = [ln for ln in out.splitlines() if ln.startswith("**[")]
    assert stamps == ["**[0:00]**", "**[5:00]**", "**[10:00]**"]


def test_markdown_speaker_names_bold():
    out = formats.to_markdown([seg("Hello.", 0, 1, speaker="Anna")], {"title": "T"})
    assert "**Anna:** Hello." in out


def test_json_round_trip_fields():
    data = json.loads(formats.to_json([seg("Hi.", 0, 1, language="en", speaker="Anna", confidence=0.91)],
                                      {"mode": "speech", "language": "en"}))
    s = data["segments"][0]
    assert s["id"] == 1 and s["start"] == 0 and s["language"] == "en" and s["speaker"] == "Anna"
    assert s["confidence"] == 0.91 and s["paragraph"] == 0 and "words" not in s


def test_speaker_names_in_order_of_first_appearance():
    segs = [seg("a", speaker="SPK_7"), seg("b", speaker="SPK_2"), seg("c", speaker="SPK_7"), seg("d", speaker="SPK_5")]
    assert name_speakers(segs, parse_names(" Anna, Ben ")) == ["Anna", "Ben", "Speaker 3"]
    assert [s.speaker for s in segs] == ["Anna", "Ben", "Anna", "Speaker 3"]


def test_default_speaker_names():
    segs = [seg("a", speaker="x"), seg("b", speaker="y")]
    assert name_speakers(segs) == ["Speaker 1", "Speaker 2"]
    assert parse_names("") == [] and parse_names(None) == []


def test_assign_speakers_by_overlap_and_nearest():
    segs = [seg("a", 0, 4), seg("b", 4, 10), seg("c", 20, 21)]
    turns = [(0, 5, "A"), (5, 12, "B")]
    assign_speakers(segs, turns)
    assert [s.speaker for s in segs] == ["A", "B", "B"]


def test_segment_is_split_where_the_other_speaker_starts():
    w = words("so how did they think that is a very good question", step=0.5)  # 0..5.5 s
    turns = [(0, 2.9, "LEX"), (2.9, 10, "LARS")]  # "that" starts at 3.0
    out = split_at_turns([seg("", 0, 5.5, words=w)], turns)
    assign_speakers(out, turns)
    assert [(s.text, s.speaker) for s in out] == [("so how did they think", "LEX"),
                                                  ("that is a very good question", "LARS")]


def test_single_stray_word_does_not_split():
    w = words("one two three four five six", step=0.5)
    turns = [(0, 1.4, "A"), (1.4, 1.9, "B"), (1.9, 10, "A")]  # only "four" is in B's turn
    out = split_at_turns([seg("", 0, 3, words=w)], turns)
    assert len(out) == 1
