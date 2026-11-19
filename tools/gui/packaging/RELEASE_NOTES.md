Paste a YouTube link (or pick a file path), choose **Speech** or **Song**, optionally a time
**Range** and **Accent recognition**, press **Run**; then **Translate** the transcript into any
listed language. Everything runs on your computer.

## New in 0.3.0

- **Range**: the length of a video or file appears on a slider as soon as it is pasted (nothing
  is downloaded yet). Pick a part, e.g. 1:00–2:00 of a two-hour podcast, and only that part is
  downloaded and processed; timestamps still match the video.
- **Translate afterwards**: translate the transcript shown into a language, then into another,
  without transcribing again. Transcript and translation are shown side by side.
- The tools are set up again on the first start after the update (a few minutes).

## Download

- **Mac** (Apple Silicon: M1 or newer): `Language-…-macos-arm64.dmg`
- **Windows** (64-bit): `Language-…-windows-x64.exe`

## Install

**Mac**
1. Open the `.dmg` and drag **Language** onto **Applications**.
2. Open Language from Applications. macOS says it cannot check the app for malware (it is not
   signed with an Apple developer certificate). Click **Done**, then open
   **System Settings → Privacy & Security**, scroll down and click **Open Anyway** next to
   “Language was blocked”. You only need to do this once.

**Windows**
1. Double-click `Language-…-windows-x64.exe`. If Windows shows “Windows protected your PC”, click
   **More info → Run anyway** (the app is not code-signed).
2. Optional: right-click the file → **Pin to Start**, or keep it on the desktop.

## First start

The app installs its tools once (about 2–3 GB, 10–30 minutes, needs internet) — press **Set up**
and wait. Then, the first time you use each feature, its model is downloaded:
transcription 3.1 GB, song vocals 0.3 GB, accent 0.1 GB, offline translation 2.5 GB.
No account or token is needed.

**Better translations (optional):** install [Ollama](https://ollama.com), open it and download
the `gemma4:12b` model (7.7 GB). Language uses it automatically when Ollama is running;
otherwise it translates offline with NLLB, which is simpler and less accurate.

On Windows everything runs on the processor, so transcription takes about as long as the audio
(or longer); on a Mac it is several times faster.
