from helpers import seg, words

from transcribe.lines import split_lines
from transcribe.paragraphs import group_paragraphs, paragraph_text


def test_song_lines_split_at_pauses():
    w = words("I heard that you settled down that you found a girl", gaps={6: 1.0})
    lines = split_lines([seg("", 0, 5, words=w)])
    assert [ln.text for ln in lines] == ["I heard that you settled down", "that you found a girl"]
    assert lines[1].start > lines[0].end


def test_song_lines_split_at_capitalised_restart_and_sentence_end():
    w = words("Regrets and mistakes they're memories made Who would have known. How bittersweet")
    texts = [ln.text for ln in split_lines([seg("", words=w)])]
    assert texts == ["Regrets and mistakes they're memories made", "Who would have known.", "How bittersweet"]


def test_orphan_word_joins_previous_line():
    w = words("We were born and raised in a summer haze", gaps={8: 0.8})
    texts = [ln.text for ln in split_lines([seg("", words=w)])]
    assert texts == ["We were born and raised in a summer haze"]


def test_long_line_is_capped():
    w = words(" ".join(f"w{i}" for i in range(30)))
    lines = split_lines([seg("", words=w)])
    assert all(len(ln.words) <= 14 for ln in lines) and len(lines) >= 2


def test_segments_without_words_pass_through():
    assert split_lines([seg("no word timing")])[0].text == "no word timing"


def test_paragraphs_break_at_long_pauses_and_speaker_changes():
    segs = [
        seg("Hello and welcome.", 0, 2, speaker="A"),
        seg("Today we talk about bees.", 2.2, 4, speaker="A"),
        seg("Thanks for having me.", 4.3, 6, speaker="B"),
        seg("It's great to be here.", 6.2, 8, speaker="B"),
        seg("Let's begin.", 12, 13, speaker="B"),
    ]
    paras = group_paragraphs(segs)
    assert [len(p) for p in paras] == [2, 2, 1]
    assert [s.paragraph for s in segs] == [0, 0, 1, 1, 2]


def test_paragraphs_do_not_break_mid_sentence_at_short_pause():
    segs = [seg("So what I was going to say", 0, 2), seg("is that bees matter.", 4.0, 6)]
    assert len(group_paragraphs(segs)) == 1


def test_long_monologue_is_broken_at_a_sentence_end():
    segs = [seg(f"Sentence number {i}.", i * 10.0, i * 10.0 + 9.5) for i in range(20)]
    paras = group_paragraphs(segs)
    assert len(paras) >= 2
    assert all(p[-1].text.endswith(".") for p in paras)


def test_topic_shift_starts_a_paragraph():
    segs = [seg("We covered the history in detail.", 0, 20), seg("Now let's look at the future.", 20.5, 25)]
    assert len(group_paragraphs(segs)) == 2


def test_language_switch_starts_a_paragraph():
    segs = [seg("Hello everyone.", 0, 2, language="en"), seg("Dzień dobry.", 2.3, 4, language="pl")]
    assert len(group_paragraphs(segs)) == 2


def test_paragraph_text_marks_low_confidence():
    a, b = seg("Clear."), seg("Mumbled.", low_confidence=True)
    assert paragraph_text([a, b]) == "Clear. Mumbled. [?]"
    assert paragraph_text([a, b], mark_low=False) == "Clear. Mumbled."
