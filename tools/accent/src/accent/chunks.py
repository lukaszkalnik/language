"""Split 16 kHz mono audio into ~20 s chunks, skip silent ones, trim silence inside the rest."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ingest.audio import rms_dbfs

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 20.0
MIN_TAIL_SECONDS = 5.0  # a shorter tail joins the previous chunk
SILENCE_DBFS = -45.0  # chunks and 30 ms frames below this are silence
FRAME_SECONDS = 0.03
SPEECH_PAD_SECONDS = 0.2  # keep this much either side of speech so word edges survive
MIN_SPEECH_SECONDS = 1.0  # a chunk with less speech than this is skipped


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    rms_dbfs: float
    silent: bool
    speech_seconds: float = 0.0
    speech_mask: np.ndarray = field(default=None, repr=False, compare=False)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def speech(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        a, b = int(round(self.start * sample_rate)), int(round(self.end * sample_rate))
        piece = samples[a:b]
        return piece if self.speech_mask is None else piece[self.speech_mask]


def chunk_bounds(n_samples: int, sample_rate: int = SAMPLE_RATE, chunk_seconds: float = CHUNK_SECONDS,
                 min_tail_seconds: float = MIN_TAIL_SECONDS) -> list[tuple[int, int]]:
    if n_samples <= 0:
        return []
    size = int(round(chunk_seconds * sample_rate))
    min_tail = int(round(min_tail_seconds * sample_rate))
    bounds = [(s, min(s + size, n_samples)) for s in range(0, n_samples, size)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < min_tail:
        last = bounds.pop()
        bounds[-1] = (bounds[-1][0], last[1])
    return bounds


def speech_mask(samples: np.ndarray, sample_rate: int = SAMPLE_RATE, silence_dbfs: float = SILENCE_DBFS,
                frame_seconds: float = FRAME_SECONDS, pad_seconds: float = SPEECH_PAD_SECONDS) -> np.ndarray:
    """Per-sample mask: True where a frame is above the energy threshold, padded both sides."""
    n = len(samples)
    if n == 0:
        return np.zeros(0, dtype=bool)
    frame = max(1, int(round(frame_seconds * sample_rate)))
    n_frames = -(-n // frame)
    padded = np.zeros(n_frames * frame, dtype=np.float64)
    padded[:n] = samples
    power = np.mean(np.square(padded.reshape(n_frames, frame)), axis=1)
    with np.errstate(divide="ignore"):
        level = 10 * np.log10(power)
    active = level >= silence_dbfs
    reach = int(round(pad_seconds / frame_seconds))
    if reach and active.any():
        active = np.convolve(active, np.ones(2 * reach + 1), mode="same") > 0
    return np.repeat(active, frame)[:n]


def make_chunks(samples: np.ndarray, sample_rate: int = SAMPLE_RATE, chunk_seconds: float = CHUNK_SECONDS,
                silence_dbfs: float = SILENCE_DBFS, min_speech_seconds: float = MIN_SPEECH_SECONDS) -> list[Chunk]:
    mask = speech_mask(samples, sample_rate, silence_dbfs)
    chunks = []
    for i, (a, b) in enumerate(chunk_bounds(len(samples), sample_rate, chunk_seconds)):
        level = rms_dbfs(samples[a:b])
        keep = mask[a:b]
        speech = float(keep.sum()) / sample_rate
        chunks.append(Chunk(i, a / sample_rate, b / sample_rate, level,
                            bool(level < silence_dbfs or speech < min_speech_seconds), speech, keep))
    return chunks
