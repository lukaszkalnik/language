# 0006 Spoken language in the window

Status: done (2026-11-20, on main; not released yet)

## Problem
transcribe detects the language with Whisper large-v3 on up to 8 windows across the file. For
rare languages (Breton, Cornish, …), short clips and songs, it can pick the wrong one and then
transcribes as that language, which ruins the transcript and the translation. On the
command line `transcribe --language xx` gets around this, but the window cannot set it.

## Proposal
A **Spoken language** selector next to Speech/Song, default **Automatic**.

- **Automatic** keeps today's behaviour: detection, and a label per passage when several
  languages are found.
- **A chosen language** is passed as `--language xx` to transcribe (speech, and song after
  `vocals`), selects that language's own model where there is one (`--model auto`: KB-Whisper
  for Swedish, later Breton, task 0001), and is the source language for translation.
- **Several languages**: if the recording seems to contain other languages besides the chosen
  one, the run warns (see Details), since everything is transcribed as the chosen language.
- The window **remembers** the selection between starts.
- **Accent recognition** is greyed out while a language other than English is chosen (with
  Automatic it stays available and is skipped with the existing note when the language turns out
  not to be English).

## Details
- **Values**: "Automatic" first, then every language transcribe can be told to use, by name with
  the code ("Breton (br)"), sorted by name. The window is standard-library only, so the list is a
  table in `gui/pipeline.py`: Whisper's 99 languages (the existing `LANGUAGES` + `EXTRA_NAMES`
  cover about 65; complete them), plus any language with its own backend. In a long readonly
  `ttk.Combobox`, typing a letter should jump to the first matching name (bind `<KeyPress>`).
- **Remembering**: a small `settings.json` in the window's data directory (`setup.data_dir()`,
  next to `installed.json`), written when the selection changes; a missing, unreadable or
  unknown value falls back to Automatic. Read in one place so later settings (translation
  target, Speech/Song) can join it.
- **Found panel**: "Language: Breton (br, chosen)" instead of a detected share, so it is clear
  nothing was decided by detection.
- **Translation**: the chosen code is `--from` for translate (now passed only for languages
  translate knows; keep that rule, and leave detection to translate for the rest). "Translate
  afterwards" uses the language of the run shown, not the selector's current value.
- **Accent**: `Checkbutton` state follows the selector (disabled for anything but English and
  Automatic); the tick is kept and applies again when switching back.

### Several languages in one recording
With `--language` transcribe now skips detection, so it cannot notice a second language;
passages in it come out as nonsense in the chosen language (or Whisper translates them).
- **Keep detecting**: with `--language`, transcribe still runs the window detection (no full
  transcription, only the language token per window; measure its cost on a long file, expected a
  few seconds) as a check, not to decide the language.
- **When to warn**: the detected languages disagree between windows, i.e. the report has more
  than one language (`LanguageReport.multilingual`). A single detected language that differs
  from the chosen one is the expected case (that is why one is chosen: Breton taken for French)
  and gives no warning.
- **Warning**: one line on stderr, e.g. `warning: other languages may be spoken (fr 40%, de 25%
  in 3 of 8 windows): everything is transcribed as br; use automatic detection to label each
  passage`, ideally with the windows' times. The window shows it in the found panel's notes.
  The `--json` metadata records the detection result alongside the chosen language.
- Song mode: the check runs on the separated vocals, as detection does now.
- False alarms are likely for rare languages (Breton taken for French in some windows and Welsh
  in others); the wording says "may" for that reason. Measure on the Breton clips of task 0001.
- Only a warning: a chosen language is always used for the whole recording. No mode that keeps
  the chosen language as the main one and labels other passages separately; for mixed
  recordings, Automatic does that.

## Done when
- Choosing a language transcribes in it (speech and song), with its own model where there is
  one; the found panel says it was chosen; translation uses it as the source; Automatic behaves
  as before.
- The selection survives closing and reopening the window.
- Accent recognition is greyed out for non-English choices.
- A recording with two languages (e.g. an English–German interview) and a chosen language gives
  the warning on stderr and in the window; a single-language one does not.
- Unit tests: step plans with and without a chosen language, settings load/save (missing and
  broken file), the warning rule; READMEs (gui, transcribe) and CHANGELOG under Unreleased.

## Outcome
- transcribe: with `--language`, detection still runs as a check (large-v3 loads only when the
  windows' result is not cached); when its windows disagree, one `Warning: other languages may
  be spoken (detected en 60% at 1:00, 3:00; de 40% at 2:00, in 3 window(s)) …` line, times in
  the original recording. `--json` adds `detected_languages`.
- Window: "Spoken language" combobox (Automatic + Whisper's 100 languages, letter jumps),
  saved in `settings.json` (`gui/settings.py`); "Accent (English only)" greyed out for other
  choices; the found panel says "Breton (br, chosen)"; the warning becomes a note in the result.
  Run and Stop are packed first so they stay visible; minimum width 760 px.
- Checked: unit tests (plans, warning rule and note, settings, letter jumps, lazy detection);
  the window under Xvfb at default and minimum size, the choice surviving a restart.
- Not checked with real models in this container: the cost of the extra detection on a long
  file, and false alarms on real Breton clips (task 0001's measurements).
