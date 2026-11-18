"""Real Whisper / VAD models on a short public clip; not run by default:

    uv run --package transcribe pytest -m integration tools/transcribe
"""

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


def test_short_video_speech_json(tmp_path):
    cmd = [sys.executable, "-m", "transcribe.cli", "https://www.youtube.com/watch?v=jNQXAC9IVRw",
           "--model", "turbo", "--json", "--cache-dir", str(tmp_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["language"] == "en" and data["mode"] == "speech"
    assert "elephant" in " ".join(s["text"] for s in data["segments"]).lower()
    assert "Mode: speech" in proc.stderr
