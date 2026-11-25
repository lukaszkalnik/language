import json

from ingest import cli
from transcribe.progress import Progress, piece_text
from transcribe.segments import Segment


def test_piece_text_is_one_line_per_piece():
    assert piece_text([Segment(0, 1, " Hello there."), Segment(1, 2, "  "), Segment(2, 3, " Bye.")]) == (
        "Hello there. Bye.\n")
    assert piece_text([Segment(0, 1, " ")]) == ""


def test_each_finished_piece_is_reported_for_the_window(monkeypatch, capsys):
    monkeypatch.setenv(cli.PROGRESS_ENV, "1")
    monkeypatch.setattr(cli, "_last_progress", {})
    progress = Progress("Transcribing", 60)
    progress(0.0)  # no text yet
    progress(30.0, [Segment(0, 30, " First piece.")])
    progress(60.0, [Segment(30, 60, " Second.")])
    lines = capsys.readouterr().err.splitlines()
    assert [json.loads(ln.split(" ", 1)[1]) for ln in lines if ln.startswith("@partial ")] == [
        "First piece.\n", "Second.\n"]
    assert "@progress 1.000 Transcribing" in lines
