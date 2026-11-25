"""LLM backend: a local model through Ollama's HTTP API (no Python client needed)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from ingest.errors import ModelError

from . import languages
from .chunks import Chunk

DEFAULT_MODEL = "gemma4:12b"
LINE_RE = re.compile(r"^\s*\[?(\d+)\]?\s*[|:.)\]-]\s?(.*)$")


def host() -> str:
    value = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value.rstrip("/").replace("://0.0.0.0", "://127.0.0.1")


NOT_RUNNING = (
    "Ollama is not running at {host}. Install it from https://ollama.com (macOS: brew install ollama), "
    "start it (open the app, or run `ollama serve`), then `ollama pull {model}`. "
    "Or translate offline with --translator nllb."
)
NOT_PULLED = "The model {model!r} is not available in Ollama. Run: ollama pull {model}  (or pick one with --llm-model; installed: {have})"


def _request(path: str, payload: dict | None = None, timeout: float = 900) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(host() + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _stream(path: str, payload: dict, on_text, timeout: float = 900) -> dict:
    """POST with streaming on; on_text(content so far) after every piece. Returns the last message."""
    data = json.dumps(dict(payload, stream=True)).encode("utf-8")
    req = urllib.request.Request(host() + path, data=data, headers={"Content-Type": "application/json"})
    content = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if not raw.strip():
                continue
            part = json.loads(raw.decode("utf-8"))
            piece = part.get("message", {}).get("content", "")
            if piece:
                content.append(piece)
                on_text("".join(content))
    return {"message": {"content": "".join(content)}}



def check(model: str) -> None:
    """Fail with a clear hint if Ollama is down or the model is not pulled."""
    try:
        tags = _request("/api/tags", timeout=5)
    except (urllib.error.URLError, OSError, TimeoutError):
        raise ModelError(NOT_RUNNING.format(host=host(), model=model)) from None
    names = [m.get("name", "") for m in tags.get("models", [])]
    wanted = model if ":" in model else model + ":latest"
    if wanted not in names and model not in names:
        raise ModelError(NOT_PULLED.format(model=model, have=", ".join(names) or "none"))


def build_messages(chunk: Chunk, source: str | None, target: str, song: bool) -> list[dict]:
    src = languages.name(source) if source else "the source language (detect it)"
    tgt = languages.name(target)
    rules = [
        f"You are a professional translator. Translate from {src} into {tgt}.",
        ("Translate faithfully, line by line. Each numbered input line becomes exactly one numbered output line "
         "with the same number: never merge, split, drop, reorder or add lines."),
        ("Keep the meaning, tone and register. Render idioms and wordplay with a natural equivalent where one exists, "
         "otherwise translate the meaning; do not add explanations."),
        "Keep names, brands and technical terms. If a line is already in the target language, repeat it unchanged.",
        ("Output only the translated lines in the form `N| translation`, one per input line. "
         "Do not repeat the original lines, do not add notes."),
    ]
    if song:
        rules.insert(2, "These are song lyrics, given in full so you can use the context. Give a faithful translation, "
                        "not a poetic, rhyming or singable rewrite. Lyrics are often old-fashioned, poetic or in "
                        "dialect (folk songs and ballads use old spellings, verb forms and pronouns): translate "
                        "them by what they meant then. Names of people and places stay names even when they look "
                        "like ordinary words or are written in lower case (e.g. 'Herr Mannelig' is a man's name).")
    if any(u.language and u.language != source for u in chunk.units):
        rules.insert(1, "Some lines may be in other languages; translate every line into the target language.")
    parts = []
    if chunk.context:
        ctx = "\n".join(f"- {u.text}" + (f"  =>  {u.translation}" if u.translation else "") for u in chunk.context)
        parts.append("Context: the lines just before this passage (for reference only, do not translate them again):\n" + ctx)
    parts.append("Translate:\n" + "\n".join(f"{i}| {u.text}" for i, u in enumerate(chunk.units, 1)))
    return [{"role": "system", "content": "\n".join(rules)}, {"role": "user", "content": "\n\n".join(parts)}]


def parse_reply(text: str, expected: int, sources: list[str] | None = None) -> dict[int, str]:
    """Numbered lines from the model's reply -> {number: translation}.

    Models sometimes echo each source line before its translation ("3| original" then
    "3| translation"); when a number appears more than once, the last line that differs from
    the source wins.
    """
    candidates: dict[int, list[str]] = {}
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if m and 1 <= int(m.group(1)) <= expected:
            candidates.setdefault(int(m.group(1)), []).append(m.group(2).strip())
    out: dict[int, str] = {}
    for n, options in candidates.items():
        source = sources[n - 1].strip() if sources and n <= len(sources) else None
        different = [o for o in options if o != source]
        out[n] = (different or options)[-1]
    if not out and expected == 1 and text.strip():
        out[1] = text.strip().splitlines()[0]
    return out


class OllamaTranslator:
    name = "llm"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.on_progress = None  # called with the fraction of a chunk translated so far
        check(model)

    def _chat(self, messages: list[dict], expected_chars: int = 0) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False, "think": False,
                   "options": {"temperature": 0.2, "num_ctx": 8192}}
        on_progress = self.on_progress

        def send():
            if on_progress is None:
                return _request("/api/chat", payload)
            # translations are roughly as long as their source; capped, since some languages run longer
            return _stream("/api/chat", payload,
                           lambda text: on_progress(min(0.99, len(text) / max(1, expected_chars))))

        try:
            reply = send()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if "think" in body.lower():  # a model without a thinking switch
                payload.pop("think")
                reply = send()
            else:
                raise ModelError(f"Ollama error {exc.code}: {body[:200]}") from None
        except (urllib.error.URLError, OSError):
            raise ModelError(NOT_RUNNING.format(host=host(), model=self.model)) from None
        return reply.get("message", {}).get("content", "")

    def translate(self, chunk: Chunk, source: str | None, target: str, song: bool = False) -> list[str]:
        n = len(chunk.units)
        texts = [u.text for u in chunk.units]
        chars = sum(len(t) + 4 for t in texts)  # plus the "12: " line numbers
        got = parse_reply(self._chat(build_messages(chunk, source, target, song), chars), n, texts)
        missing = [i for i in range(1, n + 1) if i not in got]
        if missing and len(missing) < n:
            # Ask again for just the lines that went missing, keeping the rest of the chunk as context.
            retry = Chunk([chunk.units[i - 1] for i in missing], chunk.context + [u for i, u in enumerate(chunk.units, 1) if i in got])
            again = parse_reply(self._chat(build_messages(retry, source, target, song)), len(missing),
                                [u.text for u in retry.units])
            for j, i in enumerate(missing, 1):
                if j in again:
                    got[i] = again[j]
        elif missing:
            got = {}
            for i, unit in enumerate(chunk.units, 1):  # line by line as a last resort
                one = parse_reply(self._chat(build_messages(Chunk([unit], chunk.context), source, target, song)), 1, [unit.text])
                got[i] = one.get(1, "")
        return [got.get(i, "") for i in range(1, n + 1)]
