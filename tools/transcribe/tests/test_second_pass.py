import pytest
from helpers import seg

from transcribe.pipeline import add_missing, lost_stretches


def test_lost_stretches_are_sung_regions_no_line_covers():
    voice = [(0.6, 5.0), (5.7, 9.0), (16.0, 20.0), (40.0, 40.8)]
    lines = [seg("covered", 16.2, 19.8)]
    # the two uncovered regions merge into one padded stretch; the breath at 40 s is too short
    assert lost_stretches(voice, lines, 120.0) == [pytest.approx((0.1, 9.5))]
    assert lost_stretches([(16.0, 20.0)], lines, 120.0) == []


def test_second_pass_only_adds_lines_in_gaps():
    first = [seg("one", 10, 14), seg("two", 20, 24)]
    extra = [seg("again two", 20.5, 23.5), seg("new", 2, 5)]
    assert [ln.text for ln in add_missing(first, extra)] == ["new", "one", "two"]


def test_second_pass_drops_guesses():
    first = [seg("one", 10, 14)]
    extra = [seg("Michael Jackson", 0.1, 2.6, confidence=0.06), seg("real line", 20, 23, confidence=0.7)]
    assert [ln.text for ln in add_missing(first, extra)] == ["one", "real line"]
