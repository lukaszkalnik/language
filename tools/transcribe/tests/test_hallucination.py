from helpers import seg, words

from transcribe.hallucination import (
    collapse_line_loops,
    collapse_repeats,
    collapse_word_loops,
    compression_ratio,
    filter_segments,
    is_known_phrase,
    voice_overlap,
)


def test_drops_segments_without_voice():
    voice = [(0.0, 10.0)]
    kept, dropped = filter_segments([seg("hello there", 1, 3), seg("invented words", 20, 23)], voice)
    assert [s.text for s in kept] == ["hello there"]
    assert dropped[0][1] == "no voice detected"


def test_drops_known_phrases_on_silence_but_keeps_confident_speech():
    weak = seg("Thanks for watching!", 50, 52, avg_logprob=-0.9)
    kept, dropped = filter_segments([weak])
    assert not kept and dropped[0][1] == "known hallucination"
    real = seg("And that's all, thanks for watching the whole series with me today", 0, 6, avg_logprob=-0.2)
    assert filter_segments([real])[0]


def test_drops_music_only_and_no_speech():
    kept, dropped = filter_segments([seg("♪ ♪"), seg("[Music]"), seg("x", 0, 2, no_speech_prob=0.9, avg_logprob=-1.4)])
    assert kept == [] and len(dropped) == 3


def test_low_confidence_is_marked_not_hidden():
    shaky = seg("maybe these words", 0, 2, words=words("maybe these words", prob=0.3))
    sure = seg("clear words here", 3, 5, words=words("clear words here", start=3, prob=0.95))
    kept, dropped = filter_segments([shaky, sure])
    assert not dropped
    assert kept[0].low_confidence and not kept[1].low_confidence
    assert 0.25 < kept[0].confidence < 0.35


def test_repetition_loops_are_collapsed_and_flagged():
    looped = " ".join(["I love you"] * 10)
    assert compression_ratio(looped) > 2.4
    assert collapse_repeats(looped) == "I love you I love you"
    kept, _ = filter_segments([seg(looped, 0, 10)])
    assert kept[0].text == "I love you I love you"


def test_known_phrase_matching_ignores_case_and_punctuation():
    assert is_known_phrase("Untertitel im Auftrag des ZDF, 2021")
    assert is_known_phrase("Subtitles by the Amara.org community")
    assert not is_known_phrase("We watched the subtitles")


def test_voice_overlap():
    assert voice_overlap(seg("x", 0, 10), [(0, 5)]) == 0.5
    assert voice_overlap(seg("x", 0, 10), [(0, 3), (8, 20)]) == 0.5
    assert voice_overlap(seg("x", 5, 5), [(0, 10)]) == 0.0


def test_word_level_loop_collapse_keeps_timings_consistent():
    s = seg("", 0, 10, words=words(" ".join(["la"] * 12)))
    collapse_word_loops(s)
    assert [w.text.strip() for w in s.words] == ["la", "la"]


def test_identical_low_confidence_lines_are_cut_but_choruses_stay():
    chorus = [seg("Alors on danse", i, i + 1) for i in range(6)]
    assert len(collapse_line_loops(chorus)) == 6
    loop = [seg("la la", i, i + 1, low_confidence=True) for i in range(10)]
    assert len(collapse_line_loops(loop)) == 2


def test_subtitle_site_credits_are_cut_out_of_a_real_line():
    from transcribe.segments import Segment, Word

    words = [Word(0.0, 1.0, " Svensktextning.nu"), Word(1.2, 1.6, " Friade"), Word(1.6, 1.9, " till")]
    seg = Segment(0.0, 1.9, "Svensktextning.nu Friade till", words=words)
    kept, dropped = filter_segments([seg])
    assert kept[0].text == "Friade till" and [w.text for w in kept[0].words] == [" Friade", " till"]
    assert kept[0].start == 1.2
    alone = Segment(5.0, 6.0, "Svensktextning.nu")
    kept, dropped = filter_segments([alone])
    assert not kept and dropped[0][1] == "no words"
    keep = Segment(0.0, 2.0, "Visit example.com today")  # ordinary addresses stay
    assert filter_segments([keep])[0][0].text == "Visit example.com today"


def test_song_clips_start_where_the_singing_starts():
    from transcribe.pipeline import song_clips

    assert song_clips([(15.8, 30.0), (31.0, 61.9), (70.0, 80.0)], 116.0) == [(15.3, 62.4), (69.5, 80.5)]
    assert song_clips([(0.2, 5.0), (114.0, 116.0)], 116.0) == [(0.0, 5.5), (113.5, 116.0)]
    assert song_clips([], 116.0) is None


def test_thanks_for_watching_in_other_languages():
    assert is_known_phrase("Tack för att du tittade på den här videon!")
    assert is_known_phrase("Vielen Dank fürs Zuschauen.")
    assert not is_known_phrase("Tack för allt")
