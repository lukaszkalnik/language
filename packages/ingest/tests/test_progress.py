from ingest import cli


def test_report_progress_only_when_asked(monkeypatch, capsys):
    monkeypatch.delenv(cli.PROGRESS_ENV, raising=False)
    cli.report_progress("Work", 0.5)
    assert capsys.readouterr().err == ""


def test_report_progress_lines_are_throttled_but_finish_shows(monkeypatch, capsys):
    monkeypatch.setenv(cli.PROGRESS_ENV, "1")
    monkeypatch.setattr(cli, "_last_progress", {})
    cli.report_progress("Work", 0.1)
    cli.report_progress("Work", 0.2)  # within the interval: dropped
    cli.report_progress("Other", 0.3)
    cli.report_progress("Work", 1.5)  # the end always shows, clamped
    assert capsys.readouterr().err.splitlines() == [
        "@progress 0.100 Work", "@progress 0.300 Other", "@progress 1.000 Work"]


def test_report_partial_only_when_asked_as_one_ascii_json_line(monkeypatch, capsys):
    monkeypatch.delenv(cli.PROGRESS_ENV, raising=False)
    cli.report_partial("ignored\n")
    assert capsys.readouterr().err == ""
    monkeypatch.setenv(cli.PROGRESS_ENV, "1")
    cli.report_partial("Zażółć gęślą\njaźń\n")
    cli.report_partial("")  # nothing to add: no line
    assert capsys.readouterr().err == '@partial "Za\\u017c\\u00f3\\u0142\\u0107 g\\u0119\\u015bl\\u0105\\nja\\u017a\\u0144\\n"\n'
