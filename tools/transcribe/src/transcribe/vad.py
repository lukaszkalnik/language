"""Voice activity detection (Silero) and planning of the pieces Whisper transcribes."""

from __future__ import annotations

import numpy as np

SR = 16_000
BLOCK_SECONDS = 600.0  # VAD runs block by block: flat memory on long files


def merge_regions(regions: list[tuple[float, float]], max_gap: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(regions):
        if out and a - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def quietest_point(samples: np.ndarray, offset: float, frame: float = 0.05) -> float:
    """Time (seconds, offset added) of the quietest `frame` in samples: the best place to cut."""
    n = int(frame * SR)
    usable = len(samples) // n * n
    if usable == 0:
        return offset + len(samples) / SR / 2
    power = np.mean(samples[:usable].reshape(-1, n) ** 2, axis=1)
    return offset + (int(np.argmin(power)) + 0.5) * frame


def split_region(a: float, b: float, max_len: float, cut_at=None) -> list[tuple[float, float]]:
    """Split a region longer than max_len. cut_at(lo, hi) picks a cut (e.g. the quietest moment)
    in the last 40% of each piece; without it the region is split evenly."""
    if cut_at is None:
        parts = int(np.ceil((b - a) / max_len))
        step = (b - a) / parts
        return [(a + i * step, a + (i + 1) * step) for i in range(parts)]
    out, pos = [], a
    while b - pos > max_len:
        t = cut_at(pos + 0.6 * max_len, pos + max_len)
        t = min(max(t, pos + 0.6 * max_len), pos + max_len)
        out.append((pos, t))
        pos = t
    out.append((pos, b))
    return out


def plan_pieces(regions: list[tuple[float, float]], max_len: float = 28.0, max_gap: float = 1.5,
                pad: float = 0.25, duration: float | None = None, cut_at=None) -> list[tuple[float, float]]:
    """Group voice regions into pieces of at most max_len seconds, cut in pauses where possible.

    Neighbouring regions closer than max_gap join while the piece stays short enough; a region
    longer than max_len is split at its quietest moments (cut_at), so Whisper never sees a word
    cut in half (it tends to invent a continuation there). Each piece gets `pad` seconds either side.
    """
    pieces: list[tuple[float, float]] = []
    cur: list[float] | None = None
    for a, b in sorted(regions):
        if b - a > max_len:
            if cur:
                pieces.append((cur[0], cur[1]))
                cur = None
            pieces += split_region(a, b, max_len, cut_at)
            continue
        if cur and a - cur[1] <= max_gap and b - cur[0] <= max_len:
            cur[1] = b
        else:
            if cur:
                pieces.append((cur[0], cur[1]))
            cur = [a, b]
    if cur:
        pieces.append((cur[0], cur[1]))
    end = duration if duration is not None else float("inf")
    return [(max(0.0, a - pad), min(end, b + pad)) for a, b in pieces]


class SileroVAD:
    def __init__(self, threshold: float = 0.5):
        import warnings

        warnings.filterwarnings("ignore")  # torch.jit deprecation notice from the bundled model
        from silero_vad import load_silero_vad

        self.model = load_silero_vad()  # bundled with the package (~2 MB), no download
        self.threshold = threshold

    def regions(self, reader, progress=None) -> list[tuple[float, float]]:
        """Voice regions (seconds) of a 16 kHz mono WavReader, processed in blocks."""
        import torch
        from silero_vad import get_speech_timestamps

        out: list[tuple[float, float]] = []
        t = 0.0
        while t < reader.duration:
            block = reader.read(t, t + BLOCK_SECONDS)
            if block.size:
                stamps = get_speech_timestamps(
                    torch.from_numpy(np.ascontiguousarray(block)), self.model, sampling_rate=SR,
                    threshold=self.threshold, min_silence_duration_ms=300, speech_pad_ms=120,
                    min_speech_duration_ms=200, return_seconds=True,
                )
                out += [(t + s["start"], t + s["end"]) for s in stamps]
            t += BLOCK_SECONDS
            if progress:
                progress(min(t, reader.duration), reader.duration)
        return merge_regions(out, 0.05)


def energy_regions(reader, floor_dbfs: float = -50.0, below_peak_db: float = 38.0, frame: float = 0.05,
                   max_gap: float = 0.6, min_len: float = 0.3) -> list[tuple[float, float]]:
    """Voice regions of *separated vocals* by level alone.

    Silero is trained on speech and misses belted or sustained singing; a vocal stem is
    near-silent wherever nobody sings, so its energy is a better guide there.
    """
    levels = []
    t = 0.0
    while t < reader.duration:
        block = reader.read(t, t + BLOCK_SECONDS)
        n = int(frame * SR)
        usable = len(block) // n * n
        if usable:
            power = np.mean(block[:usable].reshape(-1, n) ** 2, axis=1)
            levels.append(10 * np.log10(power + 1e-12))
        t += BLOCK_SECONDS
    if not levels:
        return []
    level = np.concatenate(levels)
    threshold = max(floor_dbfs, float(np.percentile(level, 99)) - below_peak_db)
    active = level > threshold
    regions, start = [], None
    for i, on in enumerate(np.append(active, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            regions.append((start * frame, i * frame))
            start = None
    regions = merge_regions(regions, max_gap)
    return [(a, b) for a, b in regions if b - a >= min_len]
