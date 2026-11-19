import pytest

from gui.timerange import TimeRange, format_time, parse_time, range_args


def test_format_time():
    assert format_time(0) == "0:00"
    assert format_time(65.4) == "1:05"
    assert format_time(65, total=4000) == "0:01:05"
    assert format_time(3725) == "1:02:05"


@pytest.mark.parametrize("text, seconds", [
    ("90", 90), ("1:30", 90), ("01:02:03", 3723), ("1.5", 1.5), ("0,5", 0.5), (" 2:00 ", 120), ("75:00", 4500)])
def test_parse_time(text, seconds):
    assert parse_time(text) == seconds


@pytest.mark.parametrize("text", ["", "abc", "1:60", "1:61:00", "-5", "1::2"])
def test_parse_time_rejects(text):
    with pytest.raises(ValueError):
        parse_time(text)


def test_starts_as_the_whole_file():
    r = TimeRange(290.9)
    assert (r.start, r.end, r.narrowed, r.bounds()) == (0.0, 290.9, False, (None, None))


def test_snaps_to_whole_seconds_or_the_very_end():
    r = TimeRange(290.9)
    assert r.set_start(60.4) == 60.0
    assert r.set_end(119.6) == 120.0
    assert r.bounds() == (60.0, 120.0) and r.narrowed
    assert r.set_end(290.6) == 290.9  # nearer the end than 290
    assert r.set_end(290.3) == 290.0
    assert r.set_end(500) == 290.9 and r.bounds() == (60.0, None)
    assert r.set_start(-3) == 0.0


def test_handles_keep_a_minimum_span():
    r = TimeRange(100)
    r.set_end(30)
    assert r.set_start(40) == 29.0
    assert r.set_end(10) == 30.0  # cannot pass the start
    r.set_start(99.5)
    assert r.start == 29.0
    r.set_end(100)
    assert r.set_start(100) == 99.0
    short = TimeRange(0.6)
    assert (short.set_start(0.5), short.set_end(0)) == (0.0, 0.6)


def test_typed_times():
    r = TimeRange(3700)
    assert r.set_text("start", "1:00") == 60.0
    assert r.set_text("end", "0:02:00") == 120.0
    assert (r.label("start"), r.label("end")) == ("0:01:00", "0:02:00")
    with pytest.raises(ValueError):
        r.set_text("end", "two minutes")
    assert r.end == 120.0
    r.reset()
    assert not r.narrowed


def test_range_args_only_for_narrowed_parts():
    assert range_args(None, None) == []
    assert range_args(60.0, None) == ["--start", "60"]
    assert range_args(None, 120.0) == ["--end", "120"]
    assert range_args(60.0, 120.0) == ["--start", "60", "--end", "120"]
