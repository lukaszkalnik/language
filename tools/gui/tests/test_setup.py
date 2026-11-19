import sys
from pathlib import Path

from gui import pipeline, setup


def test_needed_only_for_the_packaged_app(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(setup, "data_dir", lambda: tmp_path)
    assert setup.needed() is False  # from source: nothing to install
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert setup.needed() is True
    setup.marker().write_text('{"version": "0.0.1"}')
    assert setup.needed() is True  # an update reinstalls
    setup.marker().write_text(f'{{"version": "{setup.__version__}"}}')
    assert setup.needed() is False


def test_commands_install_every_tool_into_our_folder(tmp_path: Path):
    steps = setup.commands("uv", tmp_path)
    installs = [cmd[-1] for what, cmd in steps if what.startswith("Installing") and "tools" in cmd[-1]]
    assert installs == [str(tmp_path / "tools" / t) for t in setup.TOOLS]
    assert any(setup.DENO in cmd for _, cmd in steps)
    assert steps[-1][0] == "Fetching ffmpeg"


def test_tool_env_puts_our_bin_first(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(setup, "data_dir", lambda: tmp_path)
    env = setup.tool_env({"PATH": "/usr/bin"})
    assert env["PATH"].split(setup.os.pathsep)[0] == str(tmp_path / "bin")
    assert env["UV_TOOL_BIN_DIR"] == str(tmp_path / "bin")


def test_install_ffmpeg_copies_an_executable(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(setup, "data_dir", lambda: tmp_path)
    src = tmp_path / "ffmpeg-macos-aarch64-v9.0"
    src.write_bytes(b"bin")
    target = setup.install_ffmpeg(str(src))
    assert target.name == f"ffmpeg{setup.EXE}" and target.read_bytes() == b"bin"


def test_translator_prefers_ollama_with_the_model():
    assert pipeline.translator(["gemma4:12b", "llama3:8b"]) == "llm"
    assert pipeline.translator(["llama3:8b"]) == "nllb"
    assert pipeline.translator([]) == "nllb"
    assert pipeline.translator(None) == "nllb"  # Ollama not running


def test_ollama_models_none_when_not_running():
    assert pipeline.ollama_models("127.0.0.1:9", timeout=0.5) is None


def test_plan_passes_the_translator():
    steps = pipeline.plan(pipeline.Options("URL", "speech", "de"), "nllb")
    assert steps[1][1][-2:] == ["--translator", "nllb"]
