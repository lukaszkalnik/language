# 0004 Window: time range and translating afterwards

Status: done

## Problem
The window always processes the whole video or file, which for a long video means a long
download and run when only a minute is wanted (the tools already take `--start` / `--end`).
Translation can only be chosen before a run, so translating the same transcript into another
language, or deciding to translate after reading it, means transcribing again.

## Proposal
1. **Time range**: a slider with two handles (start and end) under the URL field. Once a URL or
   file is entered, its duration is fetched and becomes the slider's full range; both handles
   start at the ends (whole file). The selected times are shown as `mm:ss` (`h:mm:ss` for long
   files) and can also be typed. The range applies to every step of the run.
2. **Translate afterwards**: translation becomes its own action. After a run, a language selector
   and a **Translate** button translate the transcript already shown, without transcribing again;
   translating again into another language replaces the translation. The transcript stays
   visible beside or above the translation. The pre-run "Translate to" option can stay as a
   shortcut that runs the same action right after transcribing.

## Details
- **Duration**: the window is standard-library only, so it asks a tool. Add `--info` to the
  shared input arguments (`ingest.cli.add_input_args`): print the source's title and duration
  as JSON on stdout and exit before any model is loaded (YouTube: yt-dlp metadata only, no
  download; files: ffprobe/ffmpeg). The window calls it from the lightest tool, in the
  background, debounced while the URL is typed, with an animated progress bar meanwhile; until it
  answers (or if it fails, e.g. a live stream) the slider is disabled and the whole file is used.
- **Range slider**: Tk has no two-handle slider; draw one on a `Canvas` (track, selected span,
  two draggable handles, keyboard arrows on the focused handle, minimum span e.g. 1 s), plus two
  entry fields kept in sync. Snap to whole seconds.
- **Passing the range**: `--start` / `--end` only when narrowed (so caches of whole-file runs are
  reused). Song: `vocals` gets the range and `transcribe` works on the separated part. Accent:
  with a range its 10-minute YouTube cap does not matter; without one, pass `--full` only if the
  user asked for the whole video (open: keep the cap as the default?).
- **Timestamps**: with a range, transcribe's timestamps (`--json`, LRC/SRT) are relative to the
  range start or absolute; check and keep them absolute so they match the video.
- **Translate afterwards**: keep the last run's transcript (and detected language, passed as
  `--from`) in the window; the translate step is the existing `translate` run with the transcript
  on stdin, so the Ollama/NLLB choice and progress work as now. Disable Translate while a run is
  going and when there is no transcript.
- Result layout: sections TRANSCRIPT and TRANSLATION both shown (today the side-by-side
  translation replaces the transcript); consider two panes or tabs.

## Done when
- Pasting a YouTube URL shows its length on the slider within a few seconds; choosing 1:00–2:00
  downloads and processes only that minute (speech and song), and the output timestamps match
  the video.
- After a run, Translate → German and then Translate → Polish both work without re-running
  transcribe.
- Unit tests for the slider's value logic (snapping, minimum span, typed times), `--info`, and
  the window's step plans; READMEs updated.

## Outcome
- `--info` in every tool's input arguments (ingest), metadata only; the window asks `transcribe`.
- Range slider (`gui/rangeslider.py`, value logic in `gui/timerange.py`); a narrowed range goes
  to the step that reads the URL. When `--info` fails, the length is taken from the run's own
  `Duration:` / `Audio:` lines, so the slider works from the second run.
- Song timestamps stay absolute: `vocals` tags a section's WAV with where it begins (`IOFS`),
  and `open_input` adds it back (transcribe and accent).
- Accent keeps its 10-minute cap without a range (no `--full` from the window).
- Translate is its own action with `--from` and `--only-translation`; transcript and translation
  are shown in two panes.
- Checked with a real run: an 11:40 video's length showed in 3 s; 1:00–2:00 transcribed in 50 s;
  German, then Polish, without transcribing again.
