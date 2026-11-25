# Changelog

What changed for people using Language, newest release first. Each release lists the most
important changes first. The section for a release opens its GitHub release notes.

## Unreleased

- **Breton.** Choose Breton as the spoken language (`--language br` on the command line) and
  it is transcribed with a Whisper trained on Breton, which gets about three in four words
  right where the general model got one in five. The first Breton run downloads 967 MB once.
  Choose it rather than leaving the language on Automatic: detection takes Breton for French
  most of the time. Breton is translated into French with a model trained for that (1.9 GB,
  downloaded once), and from French on into the language you chose. Breton into French works
  without Ollama.
- **Choose the spoken language in the window.** A new "Spoken language" selector next to
  Speech/Song: leave it on Automatic, or pick the language when detection gets it wrong, as it
  often does with rare languages, short clips and songs. The transcript is then made in that
  language (Swedish with its own model) and translated from it. The window remembers your choice,
  and greys out accent recognition while a language other than English is chosen. On the command
  line this is `transcribe --language xx`, as before.
- **A warning for mixed languages.** With a chosen language, the recording is still checked: if
  other languages seem to be spoken too, you get a note saying which and where, since everything
  is transcribed in the chosen one. Choose Automatic for such recordings: it gives each passage
  its own language.
- **Much better Swedish.** Swedish songs and speech are now transcribed with KB-Whisper, a
  Whisper the National Library of Sweden trained on 50,000 hours of Swedish. In our test songs
  most misheard words are now right, and a chorus that used to go missing is back. The
  first Swedish run downloads 3.2 GB once. On a Mac it takes about twice as long as before and
  needs about 6 GB of memory. To use the old model, pass `--model large-v3`.
- **No more stalls on a bad network.** Song vocals, accent recognition and speaker labels no
  longer check online for model updates once the model is downloaded, so a flaky connection
  can no longer hold them up for minutes, and they work with no connection at all.
  Transcription and translation already did. Song vocals also no longer download the whole
  model again when the check fails.
- **Fewer gaps in songs.** Singing that the first pass skipped is now transcribed again on its own.
- Mac: fixed a fault in timing that could put a stray "!" into a line and switch off the checks
  for made-up text and the retries on uncertain passages.
- A short line that starts a new sung line is no longer glued to the end of the line before it.
- **Songs no longer lose their first lines.** On a song with an instrumental intro, the
  transcription could start with a made-up subtitle credit (e.g. "Svensktextning.nu") and skip
  the first sung lines. It now starts listening where the singing starts.
- "Thanks for watching" lines that Whisper makes up in Swedish, Danish, Norwegian, German,
  Polish, French, Spanish and Italian are removed, as the English ones already were.
- **Better translations of old or poetic lyrics:** names stay names ("Herr Mannelig" is no
  longer translated as "Sir manly"), and old-fashioned words are translated by their old meaning.
- The welcome screen before setup says what gets installed in plain words (the four tools by
  what they do: transcription, song vocals, accent recognition, translation) and lists each
  model's download size on its own line.
- Lists in the window keep their indent when a line wraps, so they are easier to read.
- Speaker labels no longer send usage data to pyannote's servers (pyannote.audio does that by
  default). To allow it, set `PYANNOTE_METRICS_ENABLED=true`.

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
