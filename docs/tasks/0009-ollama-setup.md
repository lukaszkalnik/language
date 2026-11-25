# 0009 Ollama installed from the window

Status: specified

## Problem
The window translates with Ollama (gemma4:12b) when it can, and otherwise falls back to NLLB,
which translates noticeably worse. Since b5ca532 it starts an Ollama that is installed but not
running. When Ollama or its model is missing, though, all it does is show a tip with a URL and
an `ollama pull` command. People using the packaged app should not need a terminal or a second
installer to get the better translations.

## Proposal
When Ollama translation is not possible, the window offers to set it up, and does so after
one click:

- **Model missing** (Ollama is there, gemma4:12b is not): the note above the transcript reads
  "Better translations need Ollama's model (about N GB, once)" with a **Download model** button.
  The download runs like any other step: it shows on the progress bar with size and
  percentage, Stop cancels it, and Ollama resumes a cancelled download next time.
- **Ollama missing**: the button reads **Install Ollama**. It installs Ollama into the app's
  own data folder (like the tools in first-run setup), without admin rights and without
  touching an Ollama the user installs elsewhere, then downloads the model as above.
- When it is done, the window says so and translates the current transcript again with Ollama
  (the button turns into "Translate again with Ollama" if a run is in progress meanwhile).
- Nothing is downloaded without that click. Declining keeps today's NLLB behaviour, and the
  offer stays in the note so the user can take it up later.
- First-run setup mentions the option in its intro (replacing "not needed if you use Ollama")
  but does not install it: the download is large and only needed for translation.

## Details
- **Model download**: Ollama's HTTP API `POST /api/pull` with `{"model": "gemma4:12b",
  "stream": true}` streams JSON lines with `status`, `total` and `completed`, enough for the
  progress bar; no `ollama` command is needed. Cancelling closes the connection; Ollama keeps
  the downloaded layers and resumes. Check the model's size at implementation time and show it
  before the click.
- **Installing Ollama** (`ollama.py`, the window's standard library only):
  - macOS: the stand-alone build from Ollama's GitHub releases (`ollama-darwin.tgz`: the
    `ollama` command and its libraries, no menu-bar app), unpacked into
    `<data folder>/ollama/`. Checked against the release's `sha256sum.txt` before unpacking;
    the version is pinned in the source like `DENO`, and updated deliberately.
  - Windows: the matching `ollama-windows-amd64.zip` (or `-arm64`), the same way.
  - Linux: `ollama-linux-amd64.tgz` (or `-arm64`), the same way.
  - `ollama.binary()` also looks in that folder (after PATH and the usual places, so a
    user's own Ollama wins). The server started from there uses the default models folder
    (`~/.ollama/models`), so a later own install of Ollama finds the model already there.
- **Memory**: gemma4:12b needs roughly 8–10 GB of RAM while it translates. Below 16 GB of
  memory the offer says so and suggests staying with NLLB; transcription is finished before
  translation starts, so Whisper and the LLM never run at the same time.
- **Disk**: check the free space on the models folder's volume before downloading and refuse
  with a clear message if it is short.
- **Files**: `tools/gui/src/gui/ollama.py` (install, pull with progress), `pipeline.py` (a
  step that pulls), `app.py` (the button in the notes; `App` is at its size limits, so the note
  with its button goes into its own widget module, like `selector.py`), `setup.py` (intro
  text), CHANGELOG.
- **Open questions**:
  - Stand-alone command only (proposed), or the full Ollama app with its menu-bar icon,
    which the user would then manage themselves? The app needs moving to Applications and
    starts at login; the command is invisible and stops with the window.
  - Should the button also appear after a run without translation, or only once NLLB has
    actually translated?
  - Does the packaged app's code signing (task 0003, parked) matter for running a downloaded
    `ollama` binary? macOS quarantines files only when they are downloaded by a browser, so a
    download over `urllib` should run; check on a clean Mac.

## Done when
- Unit tests (no network): progress parsing of `/api/pull` lines, checksum refusal, the
  binary search order including the data folder, the offer per status.
- A real run on a Mac without Ollama: Install Ollama, the model downloads with progress,
  Stop and a second click resume it, the transcript is translated with Ollama, closing the
  window stops the server, and an Ollama installed later with Homebrew is preferred and sees
  the model.
- The setup intro and CHANGELOG describe it.
