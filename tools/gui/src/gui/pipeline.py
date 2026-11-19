"""What the window runs: the command-line tools as subprocesses, chained step by step.

Nothing here imports Tk or a model library, so it is unit-tested without a display.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import setup

# Target languages offered in the selector (a subset of translate's ISO 639-1 codes).
LANGUAGES = {
    "en": "English", "de": "German", "pl": "Polish", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "sv": "Swedish", "da": "Danish",
    "no": "Norwegian", "fi": "Finnish", "cs": "Czech", "sk": "Slovak", "uk": "Ukrainian",
    "ru": "Russian", "hu": "Hungarian", "ro": "Romanian", "el": "Greek", "tr": "Turkish",
    "ar": "Arabic", "he": "Hebrew", "hi": "Hindi", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "vi": "Vietnamese", "id": "Indonesian",
}

_LANGUAGE_LINE = re.compile(r"^Language:\s*([a-z]{2,3})\b(?:\s*\((\d+%)\))?(?:;\s*also found:\s*(.*))?",
                            re.MULTILINE)
# Names for languages Whisper detects beyond the translation targets.
EXTRA_NAMES = {
    "af": "Afrikaans", "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "br": "Breton",
    "bs": "Bosnian", "ca": "Catalan", "cy": "Welsh", "et": "Estonian", "eu": "Basque", "fa": "Persian",
    "fo": "Faroese", "ga": "Irish", "gl": "Galician", "hr": "Croatian", "hy": "Armenian", "is": "Icelandic",
    "ka": "Georgian", "kk": "Kazakh", "la": "Latin", "lb": "Luxembourgish", "lt": "Lithuanian",
    "lv": "Latvian", "mk": "Macedonian", "ms": "Malay", "mt": "Maltese", "oc": "Occitan", "sl": "Slovenian",
    "sq": "Albanian", "sr": "Serbian", "sw": "Swahili", "ta": "Tamil", "th": "Thai", "tl": "Tagalog",
    "ur": "Urdu", "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese",
}


LLM_MODEL = "gemma4:12b"  # translate's default Ollama model
NLLB_NOTE = ("Translated offline with NLLB. For better translations install Ollama from "
             "https://ollama.com and download the {model} model in it.")


def ollama_models(host: str | None = None, timeout: float = 2) -> list[str] | None:
    """Models in a running Ollama, or None when it is not running."""
    host = host or os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    url = (host if host.startswith("http") else f"http://{host}") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return [m.get("name", "") for m in json.load(resp).get("models", [])]
    except (OSError, ValueError):
        return None


def translator(models: list[str] | None) -> str:
    """The local LLM when Ollama runs with translate's model, else NLLB (no extra install)."""
    if models and any(m == LLM_MODEL or m.startswith(LLM_MODEL + "-") for m in models):
        return "llm"
    return "nllb"


class Cancelled(Exception):
    pass


class StepFailed(Exception):
    pass


@dataclass
class Options:
    url: str
    kind: str = "speech"  # "speech" or "song"
    translate_to: str | None = None
    accent: bool = False


@dataclass
class Result:
    transcript: str = ""
    language: str = ""  # as describe_language() puts it
    translation: str = ""
    accent: str = ""
    notes: list[str] = field(default_factory=list)


def repo_root() -> Path | None:
    """The workspace checkout this file lives in, if it is one (for `uv run --package`)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "tools").is_dir():
            return parent
    return None


def tool_command(name: str, which: Callable[[str], str | None] = shutil.which,
                 root: Path | None = None) -> list[str]:
    """How to start a tool: installed by the app's setup, on PATH, else from the workspace with uv."""
    own = setup.bin_dir() / f"{name}{setup.EXE}"
    if which is shutil.which and own.is_file():
        return [str(own)]
    found = which(name)
    if found:
        return [found]
    root = root if root is not None else repo_root()
    uv = which("uv")
    if uv and root is not None and (root / "tools" / name).is_dir():
        return [uv, "run", "--project", str(root), "--package", name, name]
    raise StepFailed(f"{name} is not installed; run: uv tool install --editable ./tools/{name}")


def language_name(code: str) -> str:
    return LANGUAGES.get(code) or EXTRA_NAMES.get(code) or code


def detected_language(stderr: str) -> str | None:
    """The main language transcribe reports on stderr ("Language: en (97%)")."""
    m = _LANGUAGE_LINE.search(stderr)
    return m.group(1) if m else None


def describe_language(line: str) -> str | None:
    """A transcribe language line for people: "English (en, 97%); also German (de, 3%)"."""
    m = _LANGUAGE_LINE.search(line)
    if not m:
        return None
    code, share, others = m.groups()
    text = f"{language_name(code)} ({code}{', ' + share if share else ''})"
    if others:
        named = []
        for part in others.split(","):
            o = re.match(r"\s*([a-z]{2,3})\s*(?:\((\d+%)\))?", part)
            if o:
                named.append(f"{language_name(o.group(1))} ({o.group(1)}{', ' + o.group(2) if o.group(2) else ''})")
        if named:
            text += "; also " + ", ".join(named)
    return text


def last_error(stderr: str) -> str:
    """The tool's one `error:` line, or its last line of output."""
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "error:" in ln:
            return ln
    return lines[-1] if lines else "failed without a message"


def plan(opts: Options, translator: str = "llm") -> list[tuple[str, list[str]]]:
    """The steps (tool, arguments) for the options. `{audio}` is the transcribed/analysed audio."""
    steps: list[tuple[str, list[str]]] = []
    if opts.kind == "song":
        steps.append(("vocals", [opts.url, "-o", "{vocals}"]))
        steps.append(("transcribe", ["{vocals}", "--mode", "song"]))
    else:
        steps.append(("transcribe", [opts.url, "--mode", "speech"]))
    if opts.translate_to:
        steps.append(("translate", ["-", "--to", opts.translate_to, "--width", "120",
                                    "--translator", translator]))
    if opts.accent:
        steps.append(("accent", ["{vocals}" if opts.kind == "song" else opts.url]))
    return steps


STEP_TITLES = {
    "vocals": "Separating the vocals",
    "transcribe": "Transcribing",
    "translate": "Translating",
    "accent": "Recognising the accent",
}
_PROGRESS_LINE = re.compile(r"^@progress\s+([0-9.]+)\s*(.*)$")
_TQDM_LINE = re.compile(r"^(.*?)\s*(\d{1,3})%\|")  # tqdm bars, e.g. model downloads


def parse_line(line: str) -> tuple[str, object] | None:
    """One stderr line as an event: ("progress", (fraction, label or None)) or ("status", text)."""
    text = line.strip()
    if not text:
        return None
    m = _PROGRESS_LINE.match(text)
    if m:
        return "progress", (min(1.0, float(m.group(1))), m.group(2) or None)
    m = _TQDM_LINE.match(text)
    if m:
        label = m.group(1).rstrip(":").strip()
        return "progress", (int(m.group(2)) / 100, f"Downloading {label}" if "." in label else (label or None))
    return "status", text


class Runner:
    """Runs a plan, one process at a time, reporting events; `cancel()` stops it.

    on_event gets ("step", (index, count, title)), ("status", text) for a stderr line, and
    ("progress", (fraction, label)) from the tools' `@progress` lines and tqdm bars.
    """

    def __init__(self, on_event: Callable[[str, object], None] = lambda kind, value: None,
                 command: Callable[[str], list[str]] = tool_command):
        self.on_event = on_event
        self.command = command
        self._proc: subprocess.Popen | None = None
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def _run(self, tool: str, args: list[str], stdin: str | None = None) -> tuple[str, str]:
        if self._cancelled:
            raise Cancelled
        env = setup.tool_env()
        env["LANGUAGE_PROGRESS"] = "1"
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(
            self.command(tool) + args, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", env=env, creationflags=flags,
        )
        if stdin is not None:
            def feed():
                try:
                    self._proc.stdin.write(stdin)
                    self._proc.stdin.close()
                except OSError:
                    pass
            threading.Thread(target=feed, daemon=True).start()
        out_chunks: list[str] = []
        reader = threading.Thread(target=lambda: out_chunks.append(self._proc.stdout.read()), daemon=True)
        reader.start()
        err_lines = []
        for line in self._proc.stderr:  # text mode turns progress bars' \r into line breaks
            event = parse_line(line)
            if event is None:
                continue
            if event[0] == "status":
                err_lines.append(line)
                found = describe_language(event[1]) if tool == "transcribe" else None
                if found:
                    self.on_event("language", found)
            self.on_event(*event)
        self._proc.wait()
        reader.join()
        stderr = "".join(err_lines)
        if self._cancelled:
            raise Cancelled
        if self._proc.returncode != 0:
            raise StepFailed(last_error(stderr))
        return "".join(out_chunks), stderr

    def run(self, opts: Options) -> Result:
        result = Result()
        language = None
        with tempfile.TemporaryDirectory(prefix="language-gui-") as tmp:
            vocals = str(Path(tmp) / "vocals.wav")
            backend = "llm"
            if opts.translate_to:
                backend = translator(ollama_models())
                if backend == "nllb":
                    result.notes.append(NLLB_NOTE.format(model=LLM_MODEL))
            steps = plan(opts, backend)
            for index, (tool, args) in enumerate(steps, 1):
                args = [vocals if a == "{vocals}" else a for a in args]
                self.on_event("step", (index, len(steps), STEP_TITLES.get(tool, tool)))
                if tool == "transcribe":
                    result.transcript, err = self._run(tool, args)
                    language = detected_language(err)
                    result.language = describe_language(err) or ""
                    if not result.transcript.strip():
                        result.notes.append("No speech or singing found.")
                        break
                elif tool == "translate":
                    result.translation, _ = self._run(tool, args, stdin=result.transcript)
                elif tool == "accent":
                    if language and language != "en":
                        name = language_name(language)
                        result.notes.append(f"Accent recognition skipped: works for English only, this is {name}.")
                        continue
                    result.accent, _ = self._run(tool, args)
                else:
                    self._run(tool, args)
        return result


def format_result(result: Result) -> str:
    parts = []
    if result.language:
        parts.append("LANGUAGE\n" + result.language)
    if result.accent.strip():
        parts.append("ACCENT\n" + result.accent.strip())
    if result.translation.strip():
        parts.append("TRANSLATION\n" + result.translation.rstrip())
    elif result.transcript.strip():
        parts.append("TRANSCRIPT\n" + result.transcript.rstrip())
    if result.notes:
        parts.append("\n".join(result.notes))
    return "\n\n".join(parts) + "\n"
