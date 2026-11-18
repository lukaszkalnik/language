import pytest

from ingest.timeparse import format_range, format_seconds, parse_time, validate_range


@pytest.mark.parametrize(
    "text,seconds",
    [("90", 90), ("90.5", 90.5), ("0", 0), ("1:30", 90), ("01:30", 90), ("15:00", 900),
     ("1:02:03", 3723), ("00:00:07.25", 7.25), (" 2:00 ", 120), ("120:00", 7200)],
)
def test_parse_time(text, seconds):
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "abc", "1:2:3:4", "1:60", "1:00:60", "-5", "1m30s", "1,5"])
def test_parse_time_rejects(text):
    with pytest.raises(ValueError):
        parse_time(text)


def test_validate_range():
    validate_range(None, None)
    validate_range(10, None)
    validate_range(None, 10)
    validate_range(10, 20)
    with pytest.raises(ValueError, match="after"):
        validate_range(20, 10)
    with pytest.raises(ValueError, match="after"):
        validate_range(10, 10)


def test_format():
    assert format_seconds(0) == "0:00"
    assert format_seconds(61.4) == "1:01"
    assert format_seconds(3723) == "1:02:03"
    assert format_range(0, None) == "0:00-end"
    assert format_range(90, 900) == "1:30-15:00"
