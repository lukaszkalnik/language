import pytest

from ingest.models import hf_is_cached, hub_offline, load_cached_first

REPO = "someone/some-model"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def hf_cache(tmp_path):
    """A Hugging Face cache holding one file of REPO, as hf_hub_download leaves it."""
    repo = tmp_path / "models--someone--some-model"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(COMMIT)
    snapshot = repo / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    return tmp_path


def test_hf_is_cached_in_a_given_cache_dir(hf_cache):
    assert hf_is_cached(REPO, "config.json", cache_dir=str(hf_cache))
    assert hf_is_cached(REPO, cache_dir=str(hf_cache))
    assert not hf_is_cached("someone/other", "config.json", cache_dir=str(hf_cache))


def test_hub_offline_serves_cached_files_without_asking(hf_cache):
    from huggingface_hub import constants, hf_hub_download

    before = constants.HF_HUB_OFFLINE
    with hub_offline():
        assert constants.HF_HUB_OFFLINE is True
        path = hf_hub_download(REPO, "config.json", cache_dir=str(hf_cache))
        assert path.endswith("config.json")
        with pytest.raises(Exception, match="(?i)offline|local|cache"):
            hf_hub_download(REPO, "model.safetensors", cache_dir=str(hf_cache))
    assert constants.HF_HUB_OFFLINE is before


def test_hub_offline_restores_the_flag_after_an_error():
    from huggingface_hub import constants

    before = constants.HF_HUB_OFFLINE
    with pytest.raises(RuntimeError), hub_offline():
        raise RuntimeError("boom")
    assert constants.HF_HUB_OFFLINE is before


def _recording_load(*results):
    """A loader that records whether the Hub was offline on each call and returns `results` in turn."""
    from huggingface_hub import constants

    calls: list[bool] = []
    queue = list(results)

    def load():
        calls.append(constants.HF_HUB_OFFLINE)
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    return load, calls


def test_load_cached_first_stays_offline_when_cached():
    load, calls = _recording_load("model")
    assert load_cached_first(load, cached=True) == "model"
    assert calls == [True]


def test_load_cached_first_goes_online_if_a_file_is_missing():
    load, calls = _recording_load(OSError("missing file"), "model")
    assert load_cached_first(load, cached=True) == "model"
    assert calls == [True, False]


def test_load_cached_first_goes_online_on_none():
    load, calls = _recording_load(None, "model")
    assert load_cached_first(load, cached=True) == "model"
    assert calls == [True, False]


def test_load_cached_first_not_cached_loads_once_online():
    load, calls = _recording_load("model")
    assert load_cached_first(load, cached=False) == "model"
    assert calls == [False]


def test_connection_trap_catches_hub_requests(tmp_path):
    import subprocess
    import sys

    from ingest.testing import ConnectionTrap

    code = ("from huggingface_hub import hf_hub_download\n"
            "try:\n"
            f"    hf_hub_download('someone/some-model', 'config.json', cache_dir={str(tmp_path)!r})\n"
            "except Exception as exc:\n"
            "    print(type(exc).__name__)\n")
    with ConnectionTrap() as trap:
        env = trap.env()
        env["HF_HUB_ETAG_TIMEOUT"] = "2"
        subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert any(b"huggingface.co" in r for r in trap.requests), trap.requests


def test_noise_wav_differs_each_time(tmp_path):
    from ingest.testing import write_noise_wav

    a = write_noise_wav(tmp_path / "a.wav", seconds=0.5).read_bytes()
    b = write_noise_wav(tmp_path / "b.wav", seconds=0.5).read_bytes()
    assert a != b and len(a) == len(b) == 44 + 16000
