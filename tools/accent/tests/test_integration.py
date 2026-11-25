"""Real model and real YouTube download; not run by default:

    uv run --package accent pytest -m integration tools/accent
"""

import json

import pytest

from accent import cli
from ingest.testing import skip_if_youtube_refuses_ci

pytestmark = pytest.mark.integration


def run_json(capsys, *args):
    code = cli.main(["--json", *args])
    out = capsys.readouterr()
    skip_if_youtube_refuses_ci(out.err)
    assert code == 0, out.err
    return json.loads(out.out), out.err


def test_ecapa_on_model_card_sample(capsys):
    from huggingface_hub import hf_hub_download

    sample = hf_hub_download("Jzuluaga/accent-id-commonaccent_ecapa", "data/england_1.wav")
    result, err = run_json(capsys, sample)
    assert result["top"][0]["label"] == "england"
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert "statistical guess" in err


def test_short_public_video(capsys, tmp_path):
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo", 19 s
    result, err = run_json(capsys, "--cache-dir", str(tmp_path), url)
    assert result["source"]["id"] == "jNQXAC9IVRw"
    assert "Video:" in err
    assert len(result["top"]) == 3
    cached, _ = run_json(capsys, "--cache-dir", str(tmp_path), url)
    assert cached["source"]["from_cache"] is True


@pytest.mark.parametrize("model", ["ecapa", "xlsr"])
def test_cached_model_needs_no_network(tmp_path, model):
    from ingest.testing import run_twice_second_trapped, write_noise_wav

    wav = write_noise_wav(tmp_path / "noise.wav")
    run, requests = run_twice_second_trapped("accent.cli", str(wav), "--model", model, "--no-cache", "--json")
    assert run.returncode == 0, run.stderr
    assert requests == []
