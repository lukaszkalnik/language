from translate import cli, llm


def test_ollama_not_running_gives_a_hint(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")  # nothing listens on the discard port
    src = tmp_path / "in.txt"
    src.write_text("Hello\n", encoding="utf-8")
    assert cli.main([str(src), "--to", "de"]) == 1
    err = capsys.readouterr().err
    assert "Ollama is not running" in err and "ollama pull" in err and "--translator nllb" in err
    assert err.count("\n") == 1


def test_model_not_pulled_hint(monkeypatch):
    monkeypatch.setattr(llm, "_request", lambda path, payload=None, timeout=0: {"models": [{"name": "llama3:latest"}]})
    try:
        llm.check("gemma4:12b")
    except Exception as exc:
        assert "ollama pull gemma4:12b" in str(exc) and "llama3:latest" in str(exc)
    else:
        raise AssertionError("expected an error")


def test_unknown_language(tmp_path, capsys):
    src = tmp_path / "in.txt"
    src.write_text("Hello\n", encoding="utf-8")
    assert cli.main([str(src), "--to", "klingon"]) == 2
    assert "unknown language" in capsys.readouterr().err
