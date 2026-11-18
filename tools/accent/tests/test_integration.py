"""Real model and real YouTube download; not run by default:

    uv run --package accent pytest -m integration tools/accent
"""

import json

import pytest

from accent import cli

pytestmark = pytest.mark.integration


def run_json(capsys, *args):
    code = cli.main(["--json", *args])
    out = capsys.readouterr()
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
