"""Transcript data model shared by every stage (and by translate, through --json)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    language: str | None = None
    confidence: float = 1.0  # 0..1: mean word probability (or exp(avg_logprob) without words)
    low_confidence: bool = False
    speaker: str | None = None
    paragraph: int | None = None
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self, words: bool = False) -> dict:
        data = asdict(self)
        if not words:
            data.pop("words")
        else:
            data["words"] = [asdict(w) for w in self.words]
        for key in ("start", "end"):
            data[key] = round(data[key], 3)
        data["confidence"] = round(self.confidence, 3)
        for key in ("avg_logprob", "no_speech_prob", "compression_ratio"):
            data[key] = round(data[key], 3)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Segment:
        words = [Word(**w) for w in data.get("words") or []]
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "words"}
        return cls(**known, words=words)
