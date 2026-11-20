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


def _hf_file(root: Path, repo: str, name: str, text: str = "") -> None:
    commit = "0123456789abcdef0123456789abcdef01234567"
    folder = root / ("models--" + repo.replace("/", "--"))
    (folder / "refs").mkdir(parents=True, exist_ok=True)
    (folder / "refs" / "main").write_text(commit)
    (folder / "snapshots" / commit).mkdir(parents=True, exist_ok=True)
    (folder / "snapshots" / commit / name).write_text(text)


def test_demucs_cached_needs_the_bag_and_all_its_weights(tmp_path):
    spec = separate.MODELS["htdemucs_ft"]
    hf = str(tmp_path / "hf")
    assert not separate.demucs_cached(spec, hf)
    _hf_file(tmp_path / "hf", spec.hf_repo, "htdemucs_ft.yaml", "models: [aaa, bbb]\n")
    _hf_file(tmp_path / "hf", spec.hf_repo, "aaa.safetensors")
    assert not separate.demucs_cached(spec, hf)  # interrupted download: bbb is missing
    _hf_file(tmp_path / "hf", spec.hf_repo, "bbb.safetensors")
    assert separate.demucs_cached(spec, hf)


def test_demucs_cached_rejects_a_broken_bag_file(tmp_path):
    spec = separate.MODELS["htdemucs"]
    _hf_file(tmp_path / "hf", spec.hf_repo, "htdemucs.yaml", "not: [a bag")
    assert not separate.demucs_cached(spec, str(tmp_path / "hf"))


def _tiny_demucs_in_cache(hf_root: Path, spec) -> None:
    """A real (tiny, untrained) Demucs bag in a fake HF cache, laid out as Demucs 4.1 reads it."""
    import json

    from demucs.hdemucs import HDemucs
    from safetensors.torch import save_file

    sources = list(spec.stems)
    model = HDemucs(channels=4, sources=sources)
    _hf_file(hf_root, spec.hf_repo, f"{spec.name}.yaml", "models: [tiny0000]\n")
    _hf_file(hf_root, spec.hf_repo, "tiny0000.safetensors")
    path = next(hf_root.glob("models--*/snapshots/*/tiny0000.safetensors"))
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path), metadata={
        "klass": "demucs.hdemucs.HDemucs", "args": "[]",
        "kwargs": json.dumps({"channels": 4, "sources": sources}),
    })


LOAD = """
import sys
from demucs.api import Separator
from ingest.models import load_cached_first
from vocals.separate import MODELS, demucs_cached
spec = MODELS["htdemucs"]
make = lambda: Separator(model="htdemucs", device="cpu", progress=False)
if sys.argv[1] == "plain":
    make()
else:
    load_cached_first(make, demucs_cached(spec))
print("loaded")
"""


@pytest.mark.parametrize("how", ["plain", "cached_first"])
def test_cached_demucs_loads_without_a_connection(tmp_path, how):
    import subprocess

    from ingest.testing import ConnectionTrap

    spec = separate.MODELS["htdemucs"]
    _tiny_demucs_in_cache(tmp_path / "hf", spec)
    with ConnectionTrap() as trap:
        env = trap.env()
        env["HF_HUB_CACHE"] = str(tmp_path / "hf")
        env["TORCH_HOME"] = str(tmp_path / "torch")  # where Demucs' legacy fallback would save
        env["HF_HUB_ETAG_TIMEOUT"] = "2"
        proc = subprocess.run([sys.executable, "-c", LOAD, how], env=env, capture_output=True, text=True,
                              timeout=120)
    if how == "plain":
        # What Demucs does on its own: ask the Hub although everything is cached, and when that
        # fails, download the model again from Meta's legacy server.
        assert any(b"huggingface.co" in r for r in trap.requests), trap.requests
        assert any(b"dl.fbaipublicfiles.com" in r for r in trap.requests), trap.requests
    else:
        assert "loaded" in proc.stdout, proc.stderr
        assert trap.requests == []
