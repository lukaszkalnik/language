"""Language detection on several voiced windows spread across the file."""

from __future__ import annotations

from dataclasses import dataclass, field

from ingest.timeparse import format_seconds

WINDOW = 30.0  # Whisper's context
MIN_VOICE = 4.0  # a window needs this much voice to be useful
OTHER_SHARE = 0.12  # a language with at least this share of the evidence is reported


def _coverage(a: float, b: float, voice: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in voice)


def pick_windows(duration: float, voice: list[tuple[float, float]], n: int = 8,
                 window: float = WINDOW) -> list[tuple[float, float]]:
    """Choose up to n non-overlapping windows spread across the file, each starting at voice.

    Song intros are often instrumental and podcasts start with jingles, so the windows are
    spread over the whole duration and snapped to where someone actually sings or speaks.
    """
    if duration <= 0:
        return []
    if not voice:
        voice = [(0.0, duration)]
    if duration <= window:
        return [(0.0, duration)] if _coverage(0.0, duration, voice) > 0 else []
    targets = [duration * (i + 0.5) / n for i in range(n)]
    starts = sorted({max(0.0, min(a, duration - window)) for a, _ in voice})
    chosen: list[tuple[float, float]] = []
    for t in targets:
        best = None
        for s in starts:
            if any(not (s + window <= c0 or s >= c1) for c0, c1 in chosen):
                continue
            cov = _coverage(s, s + window, voice)
            if cov < MIN_VOICE:
                continue
            score = abs(s - t) - cov * 2  # near the target, with lots of voice
            if best is None or score < best[0]:
                best = (score, s)
        if best is not None and abs(best[1] - t) <= duration / n:
            chosen.append((best[1], best[1] + window))
    if not chosen:  # very little voice: use the window with the most of it
        s = max(starts, key=lambda x: _coverage(x, x + window, voice))
        chosen.append((s, min(duration, s + window)))
    return sorted(chosen)


@dataclass
class LanguageReport:
    main: str
    languages: list[dict] = field(default_factory=list)  # [{"code", "share", "windows"}], by share

    @property
    def multilingual(self) -> bool:
        return len(self.languages) > 1

    def others(self) -> list[str]:
        return [lang["code"] for lang in self.languages[1:]]


def aggregate(results: list[dict[str, float]], weights: list[float] | None = None) -> LanguageReport:
    """Combine per-window language probabilities into a main language plus any others found."""
    if not results:
        raise ValueError("no language detection results")
    weights = weights or [1.0] * len(results)
    total: dict[str, float] = {}
    wins: dict[str, int] = {}
    for probs, w in zip(results, weights, strict=True):
        top = max(probs, key=probs.get)
        wins[top] = wins.get(top, 0) + 1
        for code, p in probs.items():
            total[code] = total.get(code, 0.0) + p * w
    norm = sum(total.values()) or 1.0
    shares = sorted(((c, v / norm) for c, v in total.items()), key=lambda x: -x[1])
    main = shares[0][0]
    langs = [{"code": main, "share": round(shares[0][1], 3), "windows": wins.get(main, 0)}]
    for code, share in shares[1:]:
        # Report a language when it is a real part of the evidence, not just noise in one window.
        if share >= OTHER_SHARE or (wins.get(code, 0) >= 2 and share >= 0.05):
            langs.append({"code": code, "share": round(share, 3), "windows": wins.get(code, 0)})
    return LanguageReport(main, langs)


def mixed_warning(chosen: str, report: LanguageReport, windows: list[tuple[float, float]],
                  results: list[dict[str, float]], offset: float = 0.0) -> str | None:
    """A warning when a chosen language meets a recording whose windows disagree on the language.

    One detected language that differs from the chosen one is the expected case (a rare
    language taken for a common one) and gives no warning; several suggest a mixed recording,
    whose other passages would come out as nonsense in the chosen language.
    """
    if not report.multilingual:
        return None
    tops = [max(p, key=p.get) for p in results]
    parts = []
    for lang in report.languages:
        times = [format_seconds(a + offset) for (a, _), top in zip(windows, tops, strict=True) if top == lang["code"]]
        parts.append(f"{lang['code']} {lang['share']:.0%}" + (f" at {', '.join(times)}" if times else ""))
    return (f"Warning: other languages may be spoken (detected {'; '.join(parts)}, in {len(windows)} "
            f"window(s)): everything is transcribed as {chosen}; without --language each passage "
            "gets its own language")
