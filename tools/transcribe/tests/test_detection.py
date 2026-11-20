import numpy as np
import pytest

from transcribe import mode
from transcribe.diarize import cluster, embedding_windows, windows_to_turns
from transcribe.langdetect import aggregate, pick_windows
from transcribe.vad import merge_regions, plan_pieces

SR = 16_000
rng = np.random.default_rng(0)


def tone(freq, seconds, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return amp * (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t))


def song_like(seconds=20):
    notes = [220, 247, 262, 294, 330, 294, 262, 247]
    sig = np.concatenate([tone(notes[i % len(notes)], 0.6) for i in range(int(seconds / 0.6))])
    return sig + 0.05 * rng.standard_normal(len(sig))  # continuous band underneath


def speech_like(seconds=20):
    parts = []
    while sum(len(p) for p in parts) < seconds * SR:
        syl = np.linspace(rng.uniform(110, 180), rng.uniform(110, 180), int(0.12 * SR))
        phase = 2 * np.pi * np.cumsum(syl) / SR
        parts.append(0.3 * np.sin(phase) * np.hanning(len(phase)))
        if rng.random() < 0.25:
            parts.append(np.zeros(int(rng.uniform(0.3, 0.8) * SR)))
    return np.concatenate(parts)


def test_song_like_audio_is_song():
    d = mode.decide(mode.features(song_like()))
    assert d.mode == "song", d.reason


def test_speech_like_audio_is_speech():
    d = mode.decide(mode.features(speech_like()))
    assert d.mode == "speech", d.reason


def test_separated_vocals_push_towards_song():
    f = mode.ModeFeatures(voiced_ratio=0.8, sustained_ratio=0.15, silence_ratio=0.3, seconds=20)
    assert mode.decide(f).mode == "speech"
    assert mode.decide(f, separated=True).mode == "song"


def test_mode_sample_windows_spread_and_skip_intro():
    wins = mode.sample_windows(600)
    assert len(wins) == 6 and wins[0][0] >= 25 and wins[-1][1] <= 600
    assert mode.sample_windows(20) == [(0.0, 20)]


# --- language detection segments ---------------------------------------------------

def test_windows_are_spread_across_the_file_and_start_at_voice():
    voice = [(60.0 + i * 50, 100.0 + i * 50) for i in range(20)]  # instrumental intro: first 60 s silent
    wins = pick_windows(1100, voice, n=8)
    assert len(wins) >= 6
    assert wins[0][0] >= 60  # never the instrumental intro
    assert wins[-1][0] > 700  # reaches the end of the file
    assert all(b - a == 30 for a, b in wins)
    assert all(wins[i][1] <= wins[i + 1][0] for i in range(len(wins) - 1))  # no overlap


def test_windows_short_file_and_little_voice():
    assert pick_windows(20, [(2, 10)]) == [(0.0, 20)]
    assert len(pick_windows(600, [(300, 302)])) == 1  # still one window, at the only voice


def test_aggregate_main_and_other_languages():
    results = [{"en": 0.9, "de": 0.1}] * 5 + [{"pl": 0.85, "en": 0.15}] * 2
    rep = aggregate(results)
    assert rep.main == "en"
    assert rep.others() == ["pl"]
    assert rep.multilingual


def test_aggregate_ignores_noise_languages():
    results = [{"fr": 0.95, "it": 0.03, "es": 0.02}] * 6
    rep = aggregate(results)
    assert rep.main == "fr" and not rep.multilingual


# --- VAD piece planning --------------------------------------------------------------

def test_merge_regions():
    assert merge_regions([(0, 1), (1.2, 2), (5, 6)], 0.5) == [(0, 2), (5, 6)]


def test_plan_pieces_respects_max_length_and_pauses():
    regions = [(i * 4.0, i * 4.0 + 3.0) for i in range(20)]  # 3 s speech, 1 s pause
    pieces = plan_pieces(regions, max_len=28, pad=0)
    assert all(b - a <= 28 for a, b in pieces)
    starts = {a for a, _ in regions}
    assert all(a in starts for a, _ in pieces)  # every cut is in a pause


def test_plan_pieces_splits_long_regions_and_pads():
    pieces = plan_pieces([(10, 80)], max_len=28, pad=0.5, duration=80)
    assert len(pieces) == 3 and pieces[0][0] == 9.5 and pieces[-1][1] == 80


# --- ECAPA fallback clustering (no model: synthetic embeddings) ------------------------

def test_cluster_two_speakers():
    a = rng.normal(0, 0.05, (20, 16)) + np.eye(16)[0]
    b = rng.normal(0, 0.05, (15, 16)) + np.eye(16)[1]
    labels = cluster(np.vstack([a, b]), np.ones(35))
    assert len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1 and labels[0] != labels[20]
    forced = cluster(np.vstack([a, b]), np.ones(35), num_speakers=2)
    assert len(set(forced)) == 2


def test_cluster_merges_tiny_clusters():
    a = rng.normal(0, 0.02, (50, 8)) + np.eye(8)[0]
    stray = np.eye(8)[[3]]
    labels = cluster(np.vstack([a, stray]), np.ones(51))
    assert len(set(labels)) == 1


def test_embedding_windows_and_turns():
    wins = embedding_windows([(0, 5), (10, 10.5), (20, 21.5)])
    assert wins[0] == (0, 2.0) and (20, 21.5) in wins and all(not (a == 10) for a, _ in wins)
    turns = windows_to_turns([(0, 2), (1, 3), (2, 4), (10, 12)], np.array([0, 0, 1, 1]))
    assert turns[0][2] == "S0" and turns[1][2] == "S1"
    assert turns[0][1] == pytest.approx(turns[1][0])


def test_long_region_is_cut_at_the_quietest_moment():
    from transcribe.vad import quietest_point, split_region

    sig = np.concatenate([tone(200, 20), np.zeros(SR // 5), tone(200, 15)])  # pause at 20.0-20.2 s
    cut = quietest_point(sig[int(16 * SR):int(28 * SR)], 16.0)
    assert 20.0 <= cut <= 20.2
    pieces = split_region(0, 35.2, 28, cut_at=lambda lo, hi: quietest_point(sig[int(lo * SR):int(hi * SR)], lo))
    assert len(pieces) == 2 and 20.0 <= pieces[0][1] <= 20.2 and pieces[1] == (pieces[0][1], 35.2)


def test_split_region_without_cut_finder_is_even_and_bounded():
    from transcribe.vad import split_region

    assert split_region(0, 60, 28) == [(0, 20), (20, 40), (40, 60)]
    weird = split_region(0, 60, 28, cut_at=lambda lo, hi: -5)  # a bad cut is clamped into range
    assert all(b - a <= 28 for a, b in weird) and weird[-1][1] == 60



def test_pyannote_telemetry_is_off_unless_the_user_turned_it_on(monkeypatch):
    import os

    from transcribe.diarize import no_pyannote_telemetry

    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)
    no_pyannote_telemetry()
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "false"
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "true")
    no_pyannote_telemetry()
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "true"
