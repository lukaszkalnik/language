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
    if "--info" in args:
        if args[0] == "live":
            print(f"{tool}: error: Live streams and upcoming premieres are not supported", file=sys.stderr)
            sys.exit(1)
        print("IPv6 is not working on this network; using IPv4", file=sys.stderr)
        print('{"type": "youtube", "title": "T", "duration": 212.0}')
    elif tool == "vocals":
        open(args[args.index("-o") + 1], "wb").write(b"RIFF")
        print("Separating... 100%", file=sys.stderr)
    elif tool == "transcribe":
        print("Audio: 2:05", file=sys.stderr)
        print(f"Language: {lang} (98%)", file=sys.stderr)
        print("@progress 0.500 Transcribing", file=sys.stderr)
        print("hello world")
    elif tool == "translate":
        source = sys.stdin.read().strip()
        to = args[args.index("--to") + 1]
        src = args[args.index("--from") + 1] if "--from" in args else "?"
        print(f"[{src}>{to}] " + {"de": "hallo Welt", "pl": "witaj świecie"}.get(to, source))
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
    script.write_text(FAKE, encoding="utf-8")
    return lambda tool: [sys.executable, str(script), tool]


def test_plan_speech_with_translation_and_accent():
    steps = pipeline.plan(Options("URL", "speech", "de", True))
    assert [t for t, _ in steps] == ["transcribe", "translate", "accent"]
    assert steps[0][1] == ["URL", "--mode", "speech"]
    assert steps[1][1] == ["-", "--to", "de", "--only-translation", "--translator", "llm"]
    assert steps[2][1] == ["URL"]  # accent keeps its 10-minute cap on the whole video


def test_plan_song_separates_vocals_first():
    steps = pipeline.plan(Options("URL", "song", None, True))
    assert [t for t, _ in steps] == ["vocals", "transcribe", "accent"]
    assert steps[0][1] == ["URL", "-o", "{vocals}"]
    assert steps[1][1] == ["{vocals}", "--mode", "song"]
    assert steps[2][1] == ["{vocals}"]


def test_plan_with_a_range():
    speech = pipeline.plan(Options("URL", "speech", None, True, start=60.0, end=120.0))
    assert speech[0][1] == ["URL", "--start", "60", "--end", "120", "--mode", "speech"]
    assert speech[1][1] == ["URL", "--start", "60", "--end", "120"]
    song = pipeline.plan(Options("URL", "song", None, True, start=60.0))
    assert song[0][1] == ["URL", "--start", "60", "-o", "{vocals}"]  # only vocals reads the URL
    assert song[1][1] == ["{vocals}", "--mode", "song"] and song[2][1] == ["{vocals}"]


def test_translate_args_pass_only_known_source_languages():
    assert pipeline.translate_args("de", "pl", "nllb") == [
        "-", "--to", "de", "--from", "pl", "--only-translation", "--translator", "nllb"]
    assert "--from" not in pipeline.translate_args("de", "br")  # Breton: translate detects it
    assert "--from" not in pipeline.translate_args("de", None)


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


def test_run_speech_translate_accent(fake, monkeypatch):
    monkeypatch.setattr(pipeline, "ollama_models", lambda: None)
    events = []
    result = Runner(lambda kind, value: events.append((kind, value)), command=fake).run(
        Options("URL", "speech", "de", True))
    assert result.transcript.strip() == "hello world"
    assert result.translation.strip() == "[en>de] hallo Welt" and result.translated_to == "de"
    assert result.accent.strip() == "us  80%"
    assert pipeline.format_summary(result) == "LANGUAGE\nEnglish (en, 98%)\n\nACCENT\nus  80%\n"
    text = pipeline.format_result(result)
    assert text.startswith("LANGUAGE\nEnglish (en, 98%)\n\nACCENT\nus  80%\n\nTRANSCRIPT\nhello world\n\n")
    assert "TRANSLATION (German)\n[en>de] hallo Welt\n\nTranslated offline with NLLB" in text
    assert ("status", "Language: en (98%)") in events
    assert ("language", "English (en, 98%)") in events
    assert ("progress", (0.5, "Transcribing")) in events
    assert [v for k, v in events if k == "step"] == [
        (1, 3, "Transcribing"), (2, 3, "Translating"), (3, 3, "Recognising the accent")]
    assert "NLLB" in result.translation_notes[0] and not result.notes  # no Ollama: offline, with a hint


def test_translate_afterwards_without_transcribing_again(fake, monkeypatch):
    monkeypatch.setenv("FAKE_LANG", "pl")
    monkeypatch.setattr(pipeline, "ollama_models", lambda: [pipeline.LLM_MODEL])
    result = Runner(command=fake).run(Options("URL"))
    assert result.translation == "" and result.language_code == "pl"
    tools = []
    events = []

    def command(tool):
        tools.append(tool)
        return fake(tool)

    runner = Runner(lambda kind, value: events.append((kind, value)), command=command)
    runner.translate(result, "de")
    assert result.translation.strip() == "[pl>de] hallo Welt" and result.translation_notes == []
    runner.translate(result, "pl")  # again, into another language: replaces the translation
    assert result.translation.strip() == "[pl>pl] witaj świecie" and result.translated_to == "pl"
    assert tools == ["translate", "translate"]
    assert [v for k, v in events if k == "step"] == [(1, 1, "Translating")] * 2
    assert result.transcript.strip() == "hello world"


def test_translate_needs_a_transcript(fake):
    with pytest.raises(StepFailed, match="nothing to translate"):
        Runner(command=fake).translate(pipeline.Result(), "de")


def test_info_query(fake):
    assert pipeline.InfoQuery(command=fake).run("URL") == {"type": "youtube", "title": "T", "duration": 212.0}
    with pytest.raises(StepFailed, match="Live streams"):
        pipeline.InfoQuery(command=fake).run("live")
    query = pipeline.InfoQuery(command=fake)
    query.cancel()
    with pytest.raises(pipeline.Cancelled):
        query.run("URL")


def test_parse_info():
    assert pipeline.parse_info('note\n{"duration": 5.0}\n') == {"duration": 5.0}
    with pytest.raises(StepFailed):
        pipeline.parse_info("nothing\n")


def test_looks_like_input(tmp_path):
    song = tmp_path / "song.mp3"
    song.write_bytes(b"x")
    assert pipeline.looks_like_input("https://www.youtube.com/watch?v=abc")
    assert pipeline.looks_like_input("youtu.be/abc") and pipeline.looks_like_input("youtube.com/watch?v=x")
    assert pipeline.looks_like_input(str(song))
    assert not pipeline.looks_like_input("") and not pipeline.looks_like_input("https:/")
    assert not pipeline.looks_like_input(str(tmp_path / "missing.mp3"))


def test_run_song_skips_accent_for_other_languages(fake, monkeypatch):
    monkeypatch.setenv("FAKE_LANG", "pl")
    result = Runner(command=fake).run(Options("URL", "song", None, True))
    assert result.transcript.strip() == "hello world"
    assert result.accent == ""
    assert "English only, this is Polish" in result.notes[0]
    assert pipeline.format_result(result).startswith("LANGUAGE\nPolish (pl, 98%)\n\nAccent recognition skipped")
    assert pipeline.format_transcript(result) == "hello world\n"


def test_failure_reports_the_error_line(fake, monkeypatch):
    monkeypatch.setattr(pipeline, "plan", lambda opts, backend="llm": [("broken", [])])
    with pytest.raises(StepFailed, match="broken: error: no such video"):
        Runner(command=fake).run(Options("URL"))


def test_languages_are_known_to_translate():
    from translate.languages import LANGUAGES

    assert set(pipeline.LANGUAGES) <= set(LANGUAGES)


def test_parse_line():
    assert pipeline.parse_line("@progress 0.425 Downloading audio") == ("progress", (0.425, "Downloading audio"))
    assert pipeline.parse_line("@progress 1.000") == ("progress", (1.0, None))
    assert pipeline.parse_line("model.safetensors:  45%|####      | 1.4G/3.1G [00:10<00:12]") == (
        "progress", (0.45, "Downloading model.safetensors"))
    assert pipeline.parse_line(" 12%|#         | 3000/25000 [00:03<00:22, 900frames/s]") == ("progress", (0.12, None))
    assert pipeline.parse_line("Language: en (97%)") == ("status", "Language: en (97%)")
    assert pipeline.parse_line("   \n") is None


def test_describe_language():
    assert pipeline.describe_language("Language: br (81%); also found: fr (19%)") == (
        "Breton (br, 81%); also French (fr, 19%)")
    assert pipeline.describe_language("Language: de (set with --language)") == "German (de)"
    assert pipeline.describe_language("Language: xx (50%)") == "xx (xx, 50%)"
    assert pipeline.describe_language("Mode: speech") is None


def test_reported_duration():
    assert pipeline.reported_duration("Duration: 11:40", whole=False) == 700.0
    assert pipeline.reported_duration("Duration: 1:02:03", whole=False) == 3723.0
    assert pipeline.reported_duration("Duration: unknown", whole=True) is None
    assert pipeline.reported_duration("Audio: 4:51 (separated vocals)", whole=True) == 291.0
    assert pipeline.reported_duration("Audio: 1:00", whole=False) is None  # only the range's length
    assert pipeline.reported_duration("Audio:    cached /x/y.webm", whole=True) is None


def test_run_reports_the_length_when_info_gave_none(fake, tmp_path):
    events = []
    Runner(lambda kind, value: events.append((kind, value)), command=fake).run(Options("URL"))
    assert ("duration", 125.0) in events
    events.clear()
    Runner(lambda kind, value: events.append((kind, value)), command=fake).run(Options("URL", start=60.0))
    assert not [e for e in events if e[0] == "duration"]


def test_old_tools_without_info_get_a_clear_message():
    assert "older than this window" in pipeline.explain_info_error(
        "transcribe: error: unrecognized arguments: --info")
    assert pipeline.explain_info_error("x: error: private") == "x: error: private"
