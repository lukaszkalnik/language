"""Real Demucs model on a short public clip; not run by default:

    uv run --package vocals pytest -m integration tools/vocals
"""

import subprocess
import sys

import pytest

from ingest import audio

pytestmark = pytest.mark.integration


def test_short_video_vocals(tmp_path):
    out = tmp_path / "v.wav"
    cmd = [sys.executable, "-m", "vocals.cli", "https://www.youtube.com/watch?v=jNQXAC9IVRw",
           "--model", "htdemucs", "--cache-dir", str(tmp_path / "c"), "-o", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    info = audio.probe(out)
    assert info.sample_rate in (44100, 48000) and info.duration > 15
    assert audio.read_wav_tags(out)["ISFT"] == "vocals/htdemucs/vocals"
