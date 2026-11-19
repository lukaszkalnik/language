# gui

A small desktop window (macOS, Windows, Linux) over the tools. Paste a YouTube URL or a file
path, choose the options, press **Run**, and read the result.

```bash
uv tool install --editable ./tools/gui
language-gui
```

| Option | Runs |
| --- | --- |
| Speech | `transcribe URL --mode speech` |
| Song | `vocals URL -o vocals.wav`, then `transcribe vocals.wav --mode song` |
| Translate to … | the transcript piped into `translate --to XX` (original and translation side by side) |
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
