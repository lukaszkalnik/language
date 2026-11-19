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

| Option | Runs |
| --- | --- |
| Speech | `transcribe URL --mode speech` |
| Song | `vocals URL -o vocals.wav`, then `transcribe vocals.wav --mode song` |
| Translate to … | the transcript piped into `translate --to XX` (original and translation side by side): through Ollama when it runs with `gemma4:12b`, otherwise offline with NLLB and a note saying how to get the better one |
| Accent recognition | `accent` on the URL (speech) or the separated vocals (song); skipped with a note when transcribe detects a language other than English |

- The result box can be scrolled, selected and copied (⌘C / Ctrl+C, right-click, **Copy all**),
  not edited.
- The status line shows the running tool's latest stderr line (progress, language, mode);
  **Stop** kills the running tool. A failing tool's `error:` line is shown in the result box.
- Standard library only (Tkinter): each tool runs as a subprocess in its own environment, so the
  window never loads a model. A tool is found on `PATH` (`uv tool install --editable ./tools/<tool>`),
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
