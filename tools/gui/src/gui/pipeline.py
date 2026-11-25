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
from .timerange import range_args

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
# Names for the other languages Whisper knows (it detects and transcribes all of WHISPER_CODES).
EXTRA_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "as": "Assamese", "az": "Azerbaijani", "ba": "Bashkir",
    "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan", "br": "Breton",
    "bs": "Bosnian", "ca": "Catalan", "cy": "Welsh", "et": "Estonian", "eu": "Basque", "fa": "Persian",
    "fo": "Faroese", "ga": "Irish", "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "haw": "Hawaiian",
    "hr": "Croatian", "ht": "Haitian Creole", "hy": "Armenian", "is": "Icelandic", "jw": "Javanese",
    "ka": "Georgian", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "la": "Latin", "lb": "Luxembourgish",
    "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy", "mi": "Maori",
    "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Burmese", "ne": "Nepali", "nn": "Norwegian Nynorsk", "oc": "Occitan", "pa": "Punjabi",
    "ps": "Pashto", "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala", "sl": "Slovenian", "sn": "Shona",
    "so": "Somali", "sq": "Albanian", "sr": "Serbian", "su": "Sundanese", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "tg": "Tajik", "th": "Thai", "tk": "Turkmen", "tl": "Tagalog", "tt": "Tatar",
    "ur": "Urdu", "uz": "Uzbek", "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese",
}
# The languages transcribe can be told to use (`--language`): Whisper large-v3's.
WHISPER_CODES = (
    "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr "
    "bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn "
    "yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su "
    "yue"
).split()
AUTOMATIC = "Automatic"


def spoken_choices() -> list[str]:
    """The spoken-language selector's values: Automatic, then every language by name."""
    return [AUTOMATIC] + sorted((f"{language_name(c)} ({c})" for c in WHISPER_CODES), key=str.casefold)


def choice_code(choice: str) -> str | None:
    """The code in a selector value ("Breton (br)" -> "br"); None for Automatic or anything else."""
    m = re.fullmatch(r".*\(([a-z]{2,3})\)", choice.strip())
    return m.group(1) if m and m.group(1) in WHISPER_CODES else None


def choice_for(code: str | None) -> str:
    return f"{language_name(code)} ({code})" if code in WHISPER_CODES else AUTOMATIC


def jump_choice(choices: list[str], current: str, letter: str) -> str | None:
    """Typing a letter on the selector: the next value starting with it, round to the first."""
    if len(letter) != 1 or not letter.isalpha():
        return None
    matches = [c for c in choices if c.casefold().startswith(letter.casefold())]
    if not matches:
        return None
    later = [c for c in matches if choices.index(c) > (choices.index(current) if current in choices else -1)]
    return later[0] if later and current in matches else matches[0]


def accent_possible(spoken: str | None) -> bool:
    """Accent recognition is for English: possible with English or automatic detection."""
    return spoken in (None, "en")


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
    translate_to: str | None = None  # translate right after transcribing
    accent: bool = False
    start: float | None = None  # None: from the beginning
    end: float | None = None  # None: to the end
    language: str | None = None  # the spoken language, chosen; None: detect it


@dataclass
class Result:
    transcript: str = ""
    language: str = ""  # as describe_language() puts it
    language_code: str | None = None  # transcribe's main language, passed on to translate
    translation: str = ""
    translated_to: str | None = None
    accent: str = ""
    notes: list[str] = field(default_factory=list)
    translation_notes: list[str] = field(default_factory=list)


def repo_root() -> Path | None:
    """The workspace checkout this file lives in, if it is one (for `uv run --package`)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "tools").is_dir():
            return parent
    return None


def tool_command(name: str, which: Callable[[str], str | None] = shutil.which,
                 root: Path | None = None) -> list[str]:
    """How to start a tool: installed by the app's setup, on PATH, else from the workspace with uv.

    The packaged app uses its own installation first. Run from source, PATH comes first: the
    app's folder may hold tools from an older release than this window.
    """
    own = setup.bin_dir() / f"{name}{setup.EXE}"
    mine = which is shutil.which and own.is_file()
    if mine and setup.frozen():
        return [str(own)]
    found = which(name)
    if found:
        return [found]
    if mine:
        return [str(own)]
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
    if "set with --language" in line[m.start():].splitlines()[0]:
        return f"{language_name(code)} ({code}, chosen)"
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


_MIXED_LINE = re.compile(r"^Warning: other languages may be spoken \(detected (.*), in (\d+) window\(s\)\)",
                         re.MULTILINE)


def mixed_language_note(stderr: str, chosen: str) -> str | None:
    """transcribe's warning that a chosen language meets a recording with several, for people."""
    m = _MIXED_LINE.search(stderr)
    if not m:
        return None
    found = re.sub(r"\b([a-z]{2,3}) (\d+%)", lambda o: f"{language_name(o.group(1))} {o.group(2)}", m.group(1))
    return (f"Other languages may be spoken: detection found {found} (in {m.group(2)} parts checked). "
            f"Everything was transcribed as {language_name(chosen)}; choose Automatic as the spoken "
            "language to give each passage its own.")


_DURATION_LINE = re.compile(r"^Duration:\s*(\d+(?::\d{2}){1,2})\s*$")  # a video's length (ingest)
_AUDIO_LINE = re.compile(r"^Audio:\s*(\d+(?::\d{2}){1,2})\b")  # transcribe: length of what it reads


def clock_seconds(text: str) -> float:
    seconds = 0
    for part in text.split(":"):
        seconds = seconds * 60 + int(part)
    return float(seconds)


def reported_duration(line: str, whole: bool) -> float | None:
    """The input's length from a tool's stderr line, for when `--info` gave none.

    `Duration:` is a video's full length; transcribe's `Audio:` is the full length only when
    the run covers the whole input (`whole`).
    """
    m = _DURATION_LINE.match(line.strip()) or (_AUDIO_LINE.match(line.strip()) if whole else None)
    return clock_seconds(m.group(1)) if m else None


def explain_info_error(message: str) -> str:
    if "unrecognized arguments: --info" in message:
        return ("the installed tools are older than this window (no --info); update them "
                "(uv tool install --editable ./tools/<tool>, or Set up in the app)")
    return message


def last_error(stderr: str) -> str:
    """The tool's one `error:` line, or its last line of output."""
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "error:" in ln:
            return ln
    return lines[-1] if lines else "failed without a message"


# Source languages translate knows besides the targets above (Breton: through French).
EXTRA_SOURCES = {"br"}


def translate_args(to: str, source: str | None = None, translator: str = "llm") -> list[str]:
    """translate on the transcript (stdin); `--from` only for a language translate knows."""
    args = ["-", "--to", to]
    if source and (source in LANGUAGES or source in EXTRA_SOURCES):
        args += ["--from", source]
    return args + ["--only-translation", "--translator", translator]


def plan(opts: Options, translator: str = "llm") -> list[tuple[str, list[str]]]:
    """The steps (tool, arguments) for the options. `{vocals}` is the separated vocals' file.

    A narrowed range goes to the step that reads the URL: for a song that is vocals, whose
    output already is just that part (and says where it begins, so timestamps stay right).
    Accent without a range keeps its default cap (the first 10 minutes of a video).
    """
    rng = range_args(opts.start, opts.end)
    language = ["--language", opts.language] if opts.language else []
    steps: list[tuple[str, list[str]]] = []
    if opts.kind == "song":
        steps.append(("vocals", [opts.url, *rng, "-o", "{vocals}"]))
        steps.append(("transcribe", ["{vocals}", "--mode", "song", *language]))
    else:
        steps.append(("transcribe", [opts.url, *rng, "--mode", "speech", *language]))
    if opts.translate_to:
        steps.append(("translate", translate_args(opts.translate_to, opts.language, translator)))
    if opts.accent and accent_possible(opts.language):
        steps.append(("accent", ["{vocals}"] if opts.kind == "song" else [opts.url, *rng]))
    return steps


INFO_TOOL = "transcribe"  # needed for every run anyway; --info loads no model


def looks_like_input(text: str) -> bool:
    """Worth asking for the length: a URL, or a path to an existing file."""
    text = text.strip()
    if not text:
        return False
    if re.match(r"^(https?://|(www\.|m\.|music\.)?youtube\.com/|youtu\.be/)", text, re.IGNORECASE):
        return True
    return Path(text).expanduser().is_file()


def parse_info(stdout: str) -> dict:
    """The JSON line `--info` prints; StepFailed when there is none."""
    for line in reversed(stdout.strip().splitlines()):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    raise StepFailed("no information about the input")


class InfoQuery:
    """`<tool> INPUT --info` in the background of the window: the title and duration."""

    def __init__(self, command: Callable[[str], list[str]] = tool_command):
        self.command = command
        self._proc: subprocess.Popen | None = None
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def run(self, url: str, timeout: float = 60) -> dict:
        if self._cancelled:
            raise Cancelled
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(
            self.command(INFO_TOOL) + [url, "--info"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", env=setup.tool_env(),
            creationflags=flags,
        )
        try:
            out, err = self._proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            raise StepFailed("no answer about the input") from None
        if self._cancelled:
            raise Cancelled
        if self._proc.returncode != 0:
            raise StepFailed(explain_info_error(last_error(err)))
        return parse_info(out)


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
        self._whole = False  # the run covers the whole input (its length can be read off)

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
                length = reported_duration(event[1], self._whole and tool == "transcribe")
                if length:
                    self.on_event("duration", length)
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
        self._whole = opts.start is None and opts.end is None
        with tempfile.TemporaryDirectory(prefix="language-gui-") as tmp:
            vocals = str(Path(tmp) / "vocals.wav")
            backend = translator(ollama_models()) if opts.translate_to else "llm"
            steps = plan(opts, backend)
            for index, (tool, args) in enumerate(steps, 1):
                args = [vocals if a == "{vocals}" else a for a in args]
                self.on_event("step", (index, len(steps), STEP_TITLES.get(tool, tool)))
                if tool == "transcribe":
                    result.transcript, err = self._run(tool, args)
                    language = detected_language(err)
                    result.language_code = language
                    result.language = describe_language(err) or ""
                    mixed = mixed_language_note(err, opts.language) if opts.language else None
                    if mixed:
                        result.notes.append(mixed)
                    if not result.transcript.strip():
                        result.notes.append("No speech or singing found.")
                        break
                elif tool == "translate":
                    self._translate(result, opts.translate_to, backend)
                elif tool == "accent":
                    if language and language != "en":
                        name = language_name(language)
                        result.notes.append(f"Accent recognition skipped: works for English only, this is {name}.")
                        continue
                    result.accent, _ = self._run(tool, args)
                else:
                    self._run(tool, args)
        return result

    def translate(self, result: Result, to: str) -> Result:
        """Translate a finished run's transcript (again), without transcribing again."""
        if not result.transcript.strip():
            raise StepFailed("nothing to translate yet: run a transcription first")
        backend = translator(ollama_models())
        self.on_event("step", (1, 1, STEP_TITLES["translate"]))
        self._translate(result, to, backend)
        return result

    def _translate(self, result: Result, to: str, backend: str) -> None:
        translation, _ = self._run("translate", translate_args(to, result.language_code, backend),
                                   stdin=result.transcript)
        result.translation, result.translated_to = translation, to
        result.translation_notes = [NLLB_NOTE.format(model=LLM_MODEL)] if backend == "nllb" else []


def format_summary(result: Result) -> str:
    """What is found about the recording: language, accent and notes."""
    parts = []
    if result.language:
        parts.append("LANGUAGE\n" + result.language)
    if result.accent.strip():
        parts.append("ACCENT\n" + result.accent.strip())
    if result.notes:
        parts.append("\n".join(result.notes))
    return "\n\n".join(parts) + ("\n" if parts else "")


def format_transcript(result: Result) -> str:
    return result.transcript.rstrip() + "\n" if result.transcript.strip() else ""


def format_translation(result: Result) -> str:
    if not result.translation.strip():
        return ""
    text = result.translation.rstrip() + "\n"
    if result.translation_notes:
        text += "\n" + "\n".join(result.translation_notes) + "\n"
    return text


def format_result(result: Result) -> str:
    """Everything as one text (Copy all): summary, transcript, then the translation."""
    parts = [format_summary(result).rstrip()]
    if result.transcript.strip():
        parts.append("TRANSCRIPT\n" + format_transcript(result).rstrip())
    if result.translation.strip():
        name = language_name(result.translated_to) if result.translated_to else ""
        parts.append(f"TRANSLATION{' (' + name + ')' if name else ''}\n" + format_translation(result).rstrip())
    return "\n\n".join(p for p in parts if p) + "\n"
