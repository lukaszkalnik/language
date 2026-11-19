# gui

A small desktop window (macOS, Windows, Linux) over the tools. Paste a YouTube URL or a file
path, choose the options, press **Run**, and read the result.

```bash
uv tool install --editable ./tools/gui
language-gui
```

Or download the app (no Python, uv or terminal needed) from the
[releases](https://github.com/lukaszkalnik/language/releases): `Language-…-macos-arm64.dmg` or
`Language-…-windows-x64.exe`. Install steps for users are in
[RELEASE_NOTES.md](packaging/RELEASE_NOTES.md).

| Control | Runs |
| --- | --- |
| URL | after typing stops, `transcribe URL --info` in the background: title and length only, nothing is downloaded |
| Range | a two-handle slider over that length (drag, click the track, arrow keys; Shift: 10 s) and two time fields (`m:ss`, `h:mm:ss`). Narrowed, it adds `--start`/`--end`; otherwise the whole file is used. If the length cannot be read (older tools, a live stream), the slider stays off until a run reports it |
| Speech | `transcribe URL [--start --end] --mode speech` |
| Song | `vocals URL [--start --end] -o vocals.wav`, then `transcribe vocals.wav --mode song` (the vocals WAV says where it begins, so timestamps match the video) |
| Accent recognition | `accent` on the URL with the range (speech) or on the separated vocals (song); without a range a video is capped at its first 10 minutes; skipped with a note when transcribe detects a language other than English |
| Translate to … · **Translate** | the transcript already shown, piped into `translate --to XX --from <detected> --only-translation`; no new transcription. Translating again replaces the translation. Through Ollama when it runs with `gemma4:12b`, otherwise offline with NLLB and a note saying how to get the better one |
| also right after each run | the same translation as the last step of **Run** |

Only **Run** downloads and processes; with a range only that part is fetched (unless fetching the
whole audio is faster, see [ingest](../../packages/ingest)). The result shows the language and
accent on top, the transcript and the translation side by side in two panes.

- The result panes can be scrolled, selected and copied (⌘C / Ctrl+C, right-click, **Copy all**
  for everything), not edited.
- The status line shows the running tool's latest stderr line (progress, language, mode);
  **Stop** kills the running tool. A failing tool's `error:` line is shown in the result box.
- Standard library only (Tkinter): each tool runs as a subprocess in its own environment, so the
  window never loads a model. The packaged app uses the tools its setup installed; from source a
  tool is found on `PATH` (`uv tool install --editable ./tools/<tool>`),
  otherwise run from this workspace with `uv run --package <tool>`.
- Tkinter comes with the python.org and uv-managed Pythons; on Linux it may need
  `sudo apt install python3-tk`.

## The packaged app

[`packaging/build.py`](packaging/build.py) builds it with PyInstaller (the
[release workflow](../../.github/workflows/release.yml) runs it on macOS and Windows for a
`v*` tag and publishes the files as a GitHub release). The app contains the window, a `uv`
binary and the workspace sources. On first start (and after an update) it shows a **Set up**
step that, with that uv and into the app's own folder
(`~/Library/Application Support/Language`, `%LOCALAPPDATA%\Language`):

1. copies the sources and runs `uv tool install` for transcribe, vocals, accent and translate
   (uv fetches Python 3.12 itself);
2. installs Deno (yt-dlp's JavaScript runtime for YouTube) from its PyPI package;
3. copies ffmpeg out of the `imageio-ffmpeg` wheel.

Tools are then run from that folder's `bin`, which also goes first on `PATH`, since apps
started from the Finder do not see Homebrew. Models download on first use of each tool; none is
gated (`--speakers`, the only feature that wants a Hugging Face token, is not in the window).
The app is not signed or notarized, so macOS Gatekeeper and Windows SmartScreen ask once.

```bash
uv run --no-project --with "pyinstaller>=6.11,<7" --with pillow python tools/gui/packaging/build.py
uv run --no-project --with pillow python tools/gui/packaging/make_icon.py   # redraw the icon
```
