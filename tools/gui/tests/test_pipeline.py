import sys
import textwrap
from pathlib import Path

import pytest

from gui import pipeline
from gui.pipeline import Options, Runner, StepFailed

# Stands in for the real tools: behaves by tool name, takes no models.
FAKE = textwrap.dedent("""
    import sys
    tool, args = sys.argv[1], sys.argv[2:]
    lang = __import__("os").environ.get("FAKE_LANG", "en")
    if tool == "vocals":
        open(args[args.index("-o") + 1], "wb").write(b"RIFF")
        print("Separating... 100%", file=sys.stderr)
    elif tool == "transcribe":
        print(f"Language: {lang} (98%)", file=sys.stderr)
        print("hello world")
    elif tool == "translate":
        print("1  " + sys.stdin.read().strip() + "  |  hallo Welt")
    elif tool == "accent":
        if not args[0]:
            sys.exit(1)
        print("us  80%")
    elif tool == "broken":
        print("broken: error: no such video", file=sys.stderr)
        sys.exit(1)
""")


@pytest.fixture
def fake(tmp_path: Path):
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    return lambda tool: [sys.executable, str(script), tool]


def test_plan_speech_with_translation_and_accent():
    steps = pipeline.plan(Options("URL", "speech", "de", True))
    assert [t for t, _ in steps] == ["transcribe", "translate", "accent"]
    assert steps[0][1] == ["URL", "--mode", "speech"]
    assert steps[1][1][:3] == ["-", "--to", "de"]
    assert steps[2][1] == ["URL"]


def test_plan_song_separates_vocals_first():
    steps = pipeline.plan(Options("URL", "song", None, True))
    assert [t for t, _ in steps] == ["vocals", "transcribe", "accent"]
    assert steps[1][1] == ["{vocals}", "--mode", "song"]
    assert steps[2][1] == ["{vocals}"]


def test_detected_language_and_last_error():
    assert pipeline.detected_language("Mode: speech\nLanguage: de (91%); also found: en\n") == "de"
    assert pipeline.detected_language("Language: en (set with --language)") == "en"
    assert pipeline.detected_language("nothing here") is None
    assert pipeline.last_error("progress\nx: error: boom\nmore") == "x: error: boom"
    assert pipeline.last_error("") == "failed without a message"


def test_tool_command_prefers_path_then_uv(tmp_path: Path):
    assert pipeline.tool_command("accent", which=lambda n: f"/bin/{n}") == ["/bin/accent"]
    (tmp_path / "tools" / "accent").mkdir(parents=True)
    which = {"uv": "/bin/uv"}.get
    assert pipeline.tool_command("accent", which=which, root=tmp_path) == [
        "/bin/uv", "run", "--project", str(tmp_path), "--package", "accent", "accent"]
    with pytest.raises(StepFailed, match="uv tool install"):
        pipeline.tool_command("vocals", which=which, root=tmp_path)


def test_run_speech_translate_accent(fake):
    statuses = []
    result = Runner(statuses.append, command=fake).run(Options("URL", "speech", "de", True))
    assert result.transcript.strip() == "hello world"
    assert "hallo Welt" in result.translation
    assert result.accent.strip() == "us  80%"
    text = pipeline.format_result(result)
    assert text.startswith("ACCENT\nus  80%") and "TRANSLATION\n1  hello world" in text
    assert any(s.startswith("transcribe: Language: en") for s in statuses)


def test_run_song_skips_accent_for_other_languages(fake, monkeypatch):
    monkeypatch.setenv("FAKE_LANG", "pl")
    result = Runner(command=fake).run(Options("URL", "song", None, True))
    assert result.transcript.strip() == "hello world"
    assert result.accent == ""
    assert "English only, this is Polish" in result.notes[0]
    assert pipeline.format_result(result).startswith("TRANSCRIPT\nhello world")


def test_failure_reports_the_error_line(fake, monkeypatch):
    monkeypatch.setattr(pipeline, "plan", lambda opts: [("broken", [])])
    with pytest.raises(StepFailed, match="broken: error: no such video"):
        Runner(command=fake).run(Options("URL"))


def test_languages_are_known_to_translate():
    from translate.languages import LANGUAGES

    assert set(pipeline.LANGUAGES) <= set(LANGUAGES)
