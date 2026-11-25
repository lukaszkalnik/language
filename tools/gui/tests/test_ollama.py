import pytest

from gui import ollama
from gui.ollama import Status


def test_models_none_when_not_running():
    assert ollama.models("127.0.0.1:9", timeout=0.5) is None


def test_has_model():
    assert ollama.has_model(["gemma4:12b", "llama3:8b"])
    assert ollama.has_model(["gemma4:12b-it-q4_K_M"])
    assert not ollama.has_model(["llama3:8b"])


@pytest.mark.parametrize(("running", "installed", "started", "expected"), [
    (["gemma4:12b"], None, None, Status.READY),
    (["llama3:8b"], None, None, Status.NO_MODEL),
    (None, None, None, Status.NOT_INSTALLED),
    (None, "/opt/homebrew/bin/ollama", None, Status.NOT_STARTED),
    (None, "/opt/homebrew/bin/ollama", ["gemma4:12b"], Status.READY),  # installed, not running: started
])
def test_status_starts_an_installed_ollama(monkeypatch, running, installed, started, expected):
    calls = []
    monkeypatch.setattr(ollama, "models", lambda: running)
    monkeypatch.setattr(ollama, "binary", lambda: installed)
    monkeypatch.setattr(ollama, "start", lambda command: calls.append(command) or started)
    assert ollama.status() is expected
    assert calls == ([installed] if running is None and installed else [])


def test_start_gives_up_when_the_server_exits(tmp_path):
    fake = tmp_path / "ollama"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    assert ollama.start(str(fake), timeout=5) is None
    ollama.stop()


def test_tip_says_what_is_missing():
    assert ollama.tip(Status.READY) is None
    assert "ollama pull gemma4:12b" in (ollama.tip(Status.NO_MODEL) or "")
    assert "open the Ollama app" in (ollama.tip(Status.NOT_STARTED) or "")
    assert "https://ollama.com" in (ollama.tip(Status.NOT_INSTALLED) or "")
