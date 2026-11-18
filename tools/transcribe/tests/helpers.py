from transcribe.segments import Segment, Word


def words(text: str, start: float = 0.0, step: float = 0.3, gaps: dict[int, float] | None = None, prob: float = 0.9):
    """Words of `text`, each `step` long; gaps[i] adds a pause before word i."""
    out, t = [], start
    for i, w in enumerate(text.split()):
        t += (gaps or {}).get(i, 0.0)
        out.append(Word(t, t + step * 0.9, (" " if i else "") + w, prob))
        t += step
    return out


def seg(text: str, start: float = 0.0, end: float | None = None, **kw) -> Segment:
    ws = kw.pop("words", None)
    if end is None:
        end = start + max(1.0, 0.3 * len(text.split()))
    return Segment(start=start, end=end, text=text, words=ws or [], **kw)
