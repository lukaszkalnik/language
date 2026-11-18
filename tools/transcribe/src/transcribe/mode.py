"""Guess whether a recording is a song or speech, from the audio alone.

Singing holds pitched notes (long runs of frames with a stable pitch); speech glides and
pauses. A music bed keeps the signal continuous with almost no silent frames, whereas speech
recordings (and separated vocals) have gaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 16_000
FRAME = 640  # 40 ms
HOP = 320  # 20 ms
MIN_LAG, MAX_LAG = SR // 1000, SR // 70  # 70-1000 Hz
VOICED_CORR = 0.5
STABLE_SEMITONES = 0.7  # pitch change between frames below this counts as "held"
MIN_NOTE_SECONDS = 0.28


@dataclass
class ModeFeatures:
    voiced_ratio: float  # voiced frames / non-silent frames
    sustained_ratio: float  # voiced time inside held notes / voiced time
    silence_ratio: float  # silent frames / all frames
    seconds: float


@dataclass
class ModeDecision:
    mode: str  # "song" | "speech"
    reason: str
    features: ModeFeatures


def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME:
        return np.zeros((0, FRAME), dtype=np.float32)
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def pitch_track(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame (pitch Hz or 0, voicing strength, level dBFS) via normalised autocorrelation."""
    frames = _frames(x.astype(np.float32))
    if len(frames) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = np.mean(frames ** 2, axis=1)
    with np.errstate(divide="ignore"):
        level = 10 * np.log10(energy + 1e-12)
    win = frames * np.hanning(FRAME)[None, :]
    spec = np.fft.rfft(win, n=2 * FRAME, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, : MAX_LAG + 1]
    ac = ac / np.maximum(ac[:, :1], 1e-12)
    seg = ac[:, MIN_LAG: MAX_LAG + 1]
    lag = np.argmax(seg, axis=1) + MIN_LAG
    strength = seg[np.arange(len(seg)), lag - MIN_LAG]
    pitch = np.where(strength > VOICED_CORR, SR / lag, 0.0)
    return pitch, strength, level


def features(x: np.ndarray) -> ModeFeatures:
    pitch, _strength, level = pitch_track(x)
    if len(level) == 0:
        return ModeFeatures(0.0, 0.0, 1.0, 0.0)
    loud = level > max(np.percentile(level, 95) - 35.0, -60.0)
    voiced = (pitch > 0) & loud
    silence_ratio = 1.0 - float(loud.mean())
    voiced_ratio = float(voiced.sum()) / max(1, int(loud.sum()))

    # Held notes: runs of voiced frames whose pitch barely moves.
    min_run = int(MIN_NOTE_SECONDS * SR / HOP)
    semis = np.zeros(len(pitch))
    ok = voiced[1:] & voiced[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        semis[1:] = np.where(ok, np.abs(12 * np.log2(pitch[1:] / np.where(pitch[:-1] > 0, pitch[:-1], 1))), 99)
    stable = voiced & (semis < STABLE_SEMITONES)
    held = 0
    run = 0
    for flag in np.append(stable, False):
        if flag:
            run += 1
        else:
            if run >= min_run:
                held += run
            run = 0
    sustained_ratio = held / max(1, int(voiced.sum()))
    return ModeFeatures(voiced_ratio, sustained_ratio, silence_ratio, len(x) / SR)


def decide(f: ModeFeatures, separated: bool = False) -> ModeDecision:
    """Combine the features into a decision with a human-readable reason."""
    score = 0.0
    reasons = []
    if f.sustained_ratio >= 0.22:
        score += 2
        reasons.append(f"sustained pitched notes ({f.sustained_ratio:.0%} of voiced time)")
    elif f.sustained_ratio >= 0.12:
        score += 1
        reasons.append(f"some held notes ({f.sustained_ratio:.0%})")
    else:
        reasons.append(f"few held notes ({f.sustained_ratio:.0%})")
    if f.silence_ratio < 0.04:
        score += 1
        reasons.append("continuous music-like background")
    elif f.silence_ratio > 0.18 and not separated:
        score -= 1
        reasons.append(f"pauses between phrases ({f.silence_ratio:.0%} silent)")
    if separated:
        score += 1
        reasons.append("input is separated vocals")
    mode = "song" if score >= 2 else "speech"
    return ModeDecision(mode, ", ".join(reasons), f)


def sample_windows(duration: float, n: int = 6, window: float = 20.0) -> list[tuple[float, float]]:
    """Up to n windows spread over the file (skipping the very start, often an intro or jingle)."""
    if duration <= window * 1.5:
        return [(0.0, duration)]
    begin = min(duration * 0.05, 30.0)
    span = duration - begin - window
    count = max(1, min(n, int(span // window) + 1))
    step = span / max(1, count - 1) if count > 1 else 0
    return [(begin + i * step, begin + i * step + window) for i in range(count)]
