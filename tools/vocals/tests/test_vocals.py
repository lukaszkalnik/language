import io
import sys
from pathlib import Path

import numpy as np
import pytest

from ingest import audio, cache
from ingest.source import Source
from vocals import cli, separate
from vocals.separate import cache_tag, normalise_stem, output_name


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path):
    cache.set_cache_root(tmp_path / "cache")
    yield
    cache.set_cache_root(None)


def _src(kind="file", **kw):
    defaults = dict(kind=kind, path=Path("/music/My Song (live).mp3"), source_id="file-abc", start=0.0, end=None)
    defaults.update(kw)
    return Source(**defaults)


def test_stems():
    assert normalise_stem("vocals", "htdemucs_ft") == "vocals"
    assert normalise_stem("Instrumental", "htdemucs") == "no_vocals"
    assert normalise_stem("guitar", "htdemucs_6s") == "guitar"
    with pytest.raises(ValueError, match="no stem 'guitar'"):
        normalise_stem("guitar", "htdemucs_ft")


def test_cache_tag_includes_settings_that_change_the_result():
    assert cache_tag("htdemucs_ft") == "htdemucs_ft"
    assert cache_tag("htdemucs_ft", 3) == "htdemucs_ft-shifts3"
    src = _src()
    assert separate.stem_path(src, "htdemucs", "vocals") != separate.stem_path(src, "htdemucs_ft", "vocals")
    ranged = _src(start=30.0, end=60.0)
    assert separate.stem_path(src, "htdemucs", "vocals") != separate.stem_path(ranged, "htdemucs", "vocals")


def test_default_output_names():
    assert output_name(_src(), "vocals") == "My Song _live.vocals.wav"
    assert output_name(_src(), "no_vocals") == "My Song _live.instrumental.wav"
    from ingest.youtube import VideoInfo

    yt = _src("youtube", source_id="abcdefghijk", info=VideoInfo("abcdefghijk", "Artist - Title / Official", "Ch", 200.0))
    assert output_name(yt, "vocals") == "Artist - Title _ Official.vocals.wav"


def _fake_separation(monkeypatch, tmp_path):
    """Pretend Demucs ran: put a cached stem in place so no model is needed."""
    wav = tmp_path / "in.wav"
    audio.write_wav(wav, np.zeros(800, np.float32), 8000)
    stem = tmp_path / "stem.wav"
    audio.write_wav(stem, np.full(800, 0.25, np.float32), 8000)
    monkeypatch.setattr(cli, "separate", lambda src, *a, **k: {"vocals": stem, "no_vocals": stem})
    return wav, stem


def test_terminal_stdout_writes_a_file_not_binary(monkeypatch, tmp_path, capsys):
    if not audio.shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    wav, stem = _fake_separation(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert cli.main([str(wav)]) == 0
    out = capsys.readouterr()
    assert out.out == ""  # nothing binary on the terminal
    assert "in.vocals.wav" in out.err
    assert (tmp_path / "in.vocals.wav").read_bytes() == stem.read_bytes()


def test_piped_stdout_gets_the_wav_stream(monkeypatch, tmp_path):
    if not audio.shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    wav, stem = _fake_separation(monkeypatch, tmp_path)
    buf = io.BytesIO()
    fake_stdout = io.TextIOWrapper(buf)
    fake_stdout.isatty = lambda: False
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    assert cli.main([str(wav), "--stem", "instrumental"]) == 0
    fake_stdout.flush()
    assert buf.getvalue() == stem.read_bytes()


def test_bad_stem_is_one_error_line(monkeypatch, tmp_path, capsys):
    assert cli.main([str(tmp_path / "x.mp3"), "--stem", "piano"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("vocals: error:") and err.count("\n") == 1


def test_cached_section_stems_get_their_place_in_the_recording():
    src = _src(start=60.0, end=120.0)
    tag = cache_tag("htdemucs_ft", 1)
    for stem in [*separate.MODELS["htdemucs_ft"].stems, "no_vocals"]:  # cached by an older version
        path = separate.stem_path(src, tag, stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        audio.write_wav(path, np.zeros(800, dtype=np.float32), 8000)
    paths = separate.separate(src, "htdemucs_ft")
    assert audio.wav_origin(paths["vocals"]) == 60.0
    separate.separate(src, "htdemucs_ft")  # tagged once, not again
    assert list(audio.read_wav_tags(paths["vocals"])) == [audio.ORIGIN_TAG]
