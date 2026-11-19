# Changelog

What changed for people using Language, newest release first. Each release lists the most
important changes first. The section for a release opens its GitHub release notes.

## Unreleased

- **Songs no longer lose their first lines.** On a song with an instrumental intro, the
  transcription could start with a made-up subtitle credit (e.g. "Svensktextning.nu") and skip
  the first sung lines. It now starts listening where the singing starts.
- **Better translations of old or poetic lyrics:** names stay names ("Herr Mannelig" is no
  longer translated as "Sir manly"), and old-fashioned words are translated by their old meaning.
- The welcome screen before setup says what gets installed in plain words (the four tools by
  what they do: transcription, song vocals, accent recognition, translation) and lists each
  model's download size on its own line.
- Lists in the window keep their indent when a line wraps, so they are easier to read.

## 0.3.1

(0.3.0 was tagged but never published: its build stopped at a failing test.)


- **Work on just the part you need.** As soon as you paste a link or a file, its length appears
  on a new **Range** slider (nothing is downloaded yet). Pick, say, 1:00–2:00 of a two-hour
  podcast and only that minute is downloaded and processed: about half a minute instead of the
  whole file. Drag the handles, use the arrow keys, or type times like `1:00`.
- **Translate after reading.** Translation is now a separate **Translate** button for the
  transcript already shown. Translate into German, then into Polish, without transcribing
  again. Each translation takes about as long as before, but the minutes of transcribing are
  no longer repeated. The old "translate right after the run" choice is still there as a
  checkbox.
- **Transcript and translation side by side**, in two panes, with the language and accent
  above them. Text is now in a normal proportional font, slightly larger than the buttons.
- Timestamps of a song section match the video (a line sung at 1:15 says 1:15, not 0:15).
- After updating, the app sets up its tools again on the first start (a few minutes; models
  already downloaded are kept).

## 0.2.0

- First release of the desktop app for Mac and Windows: paste a YouTube link or a file,
  choose speech or song, translation and accent recognition, and read the result. Setup
  installs everything on the first start; nothing is uploaded.
