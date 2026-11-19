# language

Four small local command-line tools for analysing speech and songs, built to be piped together.
Everything runs on your machine; no cloud APIs.

| Tool | What it does |
| --- | --- |
| [`accent`](tools/accent) | Guesses the English accent of a speaker (top 3 of 16 CommonAccent labels, SpeechBrain) |
| [`vocals`](tools/vocals) | Separates the vocals (or another stem) from music with Demucs, at the source's sample rate |
| [`transcribe`](tools/transcribe) | Detects the language and transcribes speech or singing with Whisper large-v3 (MLX on Apple Silicon); lyrics, paragraphs, speakers, LRC/SRT/VTT/Markdown/JSON |
| [`translate`](tools/translate) | Translates text or transcribe JSON line by line with a local LLM (Ollama) or NLLB-200 |
| [`gui`](tools/gui) | `language-gui`: a desktop window over the tools — paste a URL, pick song/speech, translation, accent |

They share [`ingest`](packages/ingest): files in any format ffmpeg reads, YouTube URLs
(audio only; with `--start`/`--end` only that section is downloaded, unless the whole
audio is faster to fetch — see [ingest](packages/ingest)), WAV streams on stdin,
time ranges, and one download/result cache.

```bash
vocals "https://www.youtube.com/watch?v=..." | transcribe - | translate --to de
vocals song.mp3 -o vocals.wav
transcribe vocals.wav --lrc > song.lrc
transcribe podcast.mp3 --speakers --md > podcast.md
transcribe podcast.mp3 --speakers --json | translate --to pl --json
accent "https://www.youtube.com/watch?v=QXX-uwQ-VGw"
```

## Conventions (all tools)

- Data goes to stdout; progress, video title, notes and warnings go to stderr, so pipes stay clean.
- `-` as input reads a WAV stream from stdin.
- Audio between tools is a WAV stream. An audio tool never writes binary to a terminal: it
  writes `-o FILE` (or `<name>.vocals.wav`) and prints the path on stderr.
- Text between tools is plain text, or JSON with timestamps with `--json`; `translate` accepts both.
- Failures exit non-zero with one `error:` line, no stack trace (`LANGUAGE_DEBUG=1` shows it).
- `--start` / `--end` take seconds, `mm:ss` or `hh:mm:ss`; `--no-cache` recomputes;
  `--cache-dir` (or `LANGUAGE_CACHE`) moves the cache (default `~/.cache/language`).
- Downloads use the system's network settings (IPv6 and IPv4). At start each tool checks that
  IPv6 works (one connection attempt, at most ~1.5 s); if only IPv4 connects, as on a router
  advertising a stale IPv6 prefix, it uses IPv4 and says so. `--ipv4` (or `LANGUAGE_IPV4=1`)
  forces IPv4; `LANGUAGE_IPV4=0` skips the check.
- `--info` prints the input's title and duration as JSON and exits, without downloading the
  audio or loading a model (YouTube metadata only; files through ffprobe).
- The download size is printed the first time a model is fetched.

## Install

Needs macOS on Apple Silicon (Linux and Windows should work, with faster-whisper instead of
MLX), Python 3.11+, [uv](https://docs.astral.sh/uv/), and ffmpeg on `PATH`
(`brew install ffmpeg`, `sudo apt install ffmpeg`, `winget install ffmpeg`). YouTube downloads
also need a JavaScript runtime (Node.js, Deno or Bun) for yt-dlp. `translate`'s default backend
needs [Ollama](https://ollama.com) with a model pulled (`ollama pull gemma4:12b`).

Install each tool in its own environment, so one tool's models never come with another's:

```bash
uv tool install --editable ./tools/accent
uv tool install --editable ./tools/vocals
uv tool install --editable ./tools/transcribe
uv tool install --editable ./tools/translate
uv tool install --editable ./tools/gui        # optional window: language-gui
```

Or work inside the workspace: `uv sync --package transcribe` installs only what transcribe
needs, and `uv run --package transcribe transcribe ...` runs it.

Dependencies are pinned, except yt-dlp: YouTube changes often, and on extractor errors the
tools suggest upgrading it (`uv tool upgrade --all`, or `uv lock --upgrade-package yt-dlp`).

## Development

```bash
uv run pytest                 # all unit tests: no network, no models
uv run ruff check .
uv run --package transcribe pytest -m integration tools/transcribe   # real models, short clips
```

| Path | |
| --- | --- |
| `packages/ingest/` | shared library: inputs, YouTube, ranges, cache, CLI conventions |
| `tools/accent/` | accent recognition |
| `tools/vocals/` | vocal separation |
| `tools/transcribe/` | language detection and transcription |
| `tools/translate/` | translation |
| `tools/gui/` | desktop window (Tkinter, runs the tools as subprocesses) |
| `docs/tasks/` | task concepts and specifications (`NNNN-short-title.md`), ordered in [ROADMAP.md](ROADMAP.md) |
