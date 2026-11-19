import io
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from ingest import audio, cache, source
from ingest.errors import AudioError

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path):
    cache.set_cache_root(tmp_path / "cache")
    yield
    cache.set_cache_root(None)


# --- stdout-is-a-terminal behaviour of audio tools -----------------------------

def test_terminal_stdout_writes_default_file():
    assert source.audio_destination(None, stdout_isatty=True, default_name="song.vocals.wav") == Path("song.vocals.wav")


def test_piped_stdout_streams_wav():
    assert source.audio_destination(None, stdout_isatty=False, default_name="x.wav") is None


def test_explicit_output_wins():
    assert source.audio_destination("out.wav", stdout_isatty=True, default_name="x.wav") == Path("out.wav")
    assert source.audio_destination("out.wav", stdout_isatty=False, default_name="x.wav") == Path("out.wav")


def test_dash_output_forces_stdout_even_on_terminal():
    assert source.audio_destination("-", stdout_isatty=True, default_name="x.wav") is None


def _sine(seconds=1.0, sr=8000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_emit_wav_to_stream_and_file(tmp_path, capsys):
    wav = tmp_path / "in.wav"
    audio.write_wav(wav, _sine(), 8000)
    buf = io.BytesIO()
    source.emit_wav(wav, None, stdout=buf)
    assert buf.getvalue() == wav.read_bytes()

    dest = tmp_path / "out" / "copy.wav"
    source.emit_wav(wav, dest)
    assert dest.read_bytes() == wav.read_bytes()
    assert str(dest) in capsys.readouterr().err


# --- inputs -----------------------------------------------------------------------

def test_missing_file(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    with pytest.raises(AudioError, match="file not found"):
        source.open_input(str(tmp_path / "nope.mp3"))


@needs_ffmpeg
def test_stdin_wav_is_cached_by_content(tmp_path):
    wav = tmp_path / "in.wav"
    audio.write_wav(wav, _sine(), 8000)
    data = wav.read_bytes()
    src = source.open_input("-", stdin=io.BytesIO(data))
    assert src.kind == "stdin"
    assert src.path.read_bytes() == data
    again = source.open_input("-", stdin=io.BytesIO(data))
    assert again.source_id == src.source_id


@needs_ffmpeg
def test_empty_stdin_is_an_error():
    with pytest.raises(AudioError, match="nothing on stdin"):
        source.open_input("-", stdin=io.BytesIO(b""))


@needs_ffmpeg
def test_file_source_range_and_cached_wav(tmp_path):
    wav = tmp_path / "tone.wav"
    audio.write_wav(wav, np.stack([_sine(3.0, 44100)] * 2, axis=1), 44100)
    src = source.open_input(str(wav), start=1.0, end=2.0)
    assert src.kind == "file" and src.source_id.startswith("file-")
    out = src.cached_wav(16000, 1)
    with wave.open(str(out)) as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1
        assert abs(wf.getnframes() - 16000) < 400
    assert src.key("x") in out.name or out.name.startswith(src.key())


@needs_ffmpeg
def test_to_wav_keeps_source_rate_when_not_asked(tmp_path):
    wav = tmp_path / "tone.wav"
    audio.write_wav(wav, np.stack([_sine(1.0, 48000)] * 2, axis=1), 48000)
    out = audio.to_wav(wav, tmp_path / "out.wav")
    info = audio.probe(out)
    assert info.sample_rate == 48000 and info.channels == 2


def test_wav_reader_slices(tmp_path):
    wav = tmp_path / "tone.wav"
    samples = _sine(2.0, 8000)
    audio.write_wav(wav, samples, 8000)
    with audio.WavReader(wav) as reader:
        assert reader.duration == pytest.approx(2.0)
        part = reader.read(0.5, 1.0)
        assert len(part) == 4000
        np.testing.assert_allclose(part, samples[4000:8000], atol=1e-4)
        assert len(reader.read(1.9, None)) == 800
        assert len(reader.read(3.0, 4.0)) == 0


def test_ffmpeg_missing_gives_install_hint(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    with pytest.raises(AudioError, match="ffmpeg was not found on PATH. Install"):
        audio.require_ffmpeg()


def test_wav_tags_round_trip_and_stay_readable(tmp_path):
    wav = tmp_path / "v.wav"
    samples = _sine(1.0, 8000)
    audio.write_wav(wav, samples, 8000)
    audio.add_wav_tags(wav, {"ISFT": "vocals/htdemucs_ft/vocals", "INAM": "Ünïcode title"})
    assert audio.read_wav_tags(wav) == {"ISFT": "vocals/htdemucs_ft/vocals", "INAM": "Ünïcode title"}
    with audio.WavReader(wav) as reader:  # the extra chunk does not disturb readers
        assert len(reader.read()) == len(samples)


def test_wav_tags_of_untagged_and_non_wav(tmp_path):
    wav = tmp_path / "plain.wav"
    audio.write_wav(wav, _sine(0.1, 8000), 8000)
    assert audio.read_wav_tags(wav) == {}
    other = tmp_path / "x.mp3"
    other.write_bytes(b"ID3....")
    assert audio.read_wav_tags(other) == {}


@needs_ffmpeg
def test_input_info_of_a_file(tmp_path):
    wav = tmp_path / "tone.wav"
    audio.write_wav(wav, _sine(2.0), 8000)
    info = source.input_info(str(wav))
    assert info["type"] == "file" and info["title"] == "tone.wav"
    assert abs(info["duration"] - 2.0) < 0.05


def test_input_info_missing_file():
    with pytest.raises(AudioError, match="file not found"):
        source.input_info("/no/such/file.mp3")


def test_input_info_of_a_video_uses_metadata_only(monkeypatch):
    from ingest import youtube

    calls = []

    def fake(url, cache_dir, use_cache=True, ipv4=False):
        calls.append(url)
        return youtube.VideoInfo("dQw4w9WgXcQ", "Title", "Channel", 212.0)

    monkeypatch.setattr(youtube, "video_info", fake)
    info = source.input_info("https://youtu.be/dQw4w9WgXcQ")
    assert info == {"type": "youtube", "id": "dQw4w9WgXcQ", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "title": "Title", "channel": "Channel", "duration": 212.0}
    assert calls == ["https://youtu.be/dQw4w9WgXcQ"]


@needs_ffmpeg
def test_section_wav_keeps_its_place_in_the_recording(tmp_path):
    wav = tmp_path / "section.wav"
    audio.write_wav(wav, _sine(2.0), 8000)
    assert source.open_input(str(wav)).timeline_start == 0.0
    audio.add_wav_tags(wav, {audio.ORIGIN_TAG: "60.000"})
    src = source.open_input(str(wav), start=0.5)
    assert (src.origin, src.timeline_start) == (60.0, 60.5)
    assert src.describe()["range"]["origin"] == 60.0
    with open(wav, "rb") as fh:
        piped = source.open_input("-", stdin=fh)
    assert piped.timeline_start == 60.0


def test_wav_origin_ignores_bad_values(tmp_path):
    wav = tmp_path / "x.wav"
    audio.write_wav(wav, _sine(0.1), 8000)
    audio.add_wav_tags(wav, {audio.ORIGIN_TAG: "soon"})
    assert audio.wav_origin(wav) == 0.0
