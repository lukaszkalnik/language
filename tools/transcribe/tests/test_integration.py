"""Real Whisper / VAD models on a short public clip; not run by default:

    uv run --package transcribe pytest -m integration tools/transcribe
"""

import json
import subprocess
import sys

import pytest

from ingest.testing import skip_if_youtube_refuses_ci

pytestmark = pytest.mark.integration


def test_short_video_speech_json(tmp_path):
    cmd = [sys.executable, "-m", "transcribe.cli", "https://www.youtube.com/watch?v=jNQXAC9IVRw",
           "--model", "turbo", "--json", "--cache-dir", str(tmp_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    skip_if_youtube_refuses_ci(proc.stderr)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["language"] == "en" and data["mode"] == "speech"
    assert "elephant" in " ".join(s["text"] for s in data["segments"]).lower()
    assert "Mode: speech" in proc.stderr


def test_breton_uses_the_breton_whisper():
    # Wikitongues, "Yann speaking Breton" (Breton subtitles: "Setu, penaos 'm eus bet desket brezhoneg?").
    # The default cache keeps the converted model between runs; --no-cache transcribes anew.
    cmd = [sys.executable, "-m", "transcribe.cli", "https://youtu.be/IyBNVrLQ2-A", "--end", "0:30",
           "--language", "br", "--json", "--no-cache"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    skip_if_youtube_refuses_ci(proc.stderr)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["language"] == "br" and data["model"].endswith(":whisper-br")
    text = " ".join(s["text"] for s in data["segments"]).lower()
    assert "brezhoneg" in text and "zud" in text


def test_pyannote_reads_telemetry_switch():
    """No network needed, but importing pyannote takes seconds, so it is kept out of the unit tests."""
    import os

    code = ("from transcribe.diarize import no_pyannote_telemetry\n"
            "no_pyannote_telemetry()\n"
            "from pyannote.audio.telemetry.metrics import is_metrics_enabled\n"
            "print(is_metrics_enabled())\n")
    env = {k: v for k, v in os.environ.items() if k != "PYANNOTE_METRICS_ENABLED"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "False", out.stderr


@pytest.mark.parametrize("speakers", [False, True], ids=["plain", "speakers"])
def test_cached_models_need_no_network(speakers):
    from huggingface_hub import hf_hub_download

    from ingest.testing import run_twice_second_trapped

    speech = hf_hub_download("Jzuluaga/accent-id-commonaccent_ecapa", "data/england_1.wav")
    args = [speech, "--mode", "speech", "--model", "turbo", "--json", "--no-cache"]
    run, requests = run_twice_second_trapped("transcribe.cli", *args, *(["--speakers"] if speakers else []))
    assert run.returncode == 0, run.stderr
    assert requests == []
