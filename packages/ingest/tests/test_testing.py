import pytest

from ingest.testing import skip_if_youtube_refuses_ci

REFUSED = "vocals: error: YouTube asked to confirm this is not a bot and refused the download."


def test_youtube_refusal_skips_on_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    with pytest.raises(pytest.skip.Exception):
        skip_if_youtube_refuses_ci(REFUSED)


def test_youtube_refusal_does_not_skip_elsewhere(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    skip_if_youtube_refuses_ci(REFUSED)  # the test goes on and fails on the error


def test_other_errors_do_not_skip_on_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    skip_if_youtube_refuses_ci("vocals: error: ffmpeg was not found on PATH.")
