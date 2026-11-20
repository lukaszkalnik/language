# 0001 Breton support

Status: concept

## Problem
Whisper large-v3 nominally knows Breton (brezhoneg, `br`), but transcribes it poorly, and the
translation backends have no confirmed Breton coverage. Language detection is the first hurdle:
Whisper saw little Breton in training and will likely call it French (or Welsh, Cornish), then
transcribe it as that language. The same happens with other rare languages and with short or
sung clips, and the window has no way around it: `transcribe --language xx` exists, but only on
the command line.

## Proposal
1. **Spoken language in the window** (first, useful on its own for every rare language): a
   "Spoken language" selector next to Speech/Song, default **Automatic**. Choosing a language
   skips detection and passes `--language xx` to transcribe; the same code is the source for
   translation. Automatic keeps today's behaviour.
2. **Breton transcription** with a model trained for it, picked through transcribe's
   language → model table (`--model auto`, built for Swedish in 0005) when the language is `br`,
   chosen or detected.
3. **Breton translation** Breton ↔ French (other languages through French). Breton appears in
   the "Translate to" list once a backend covers it.

## Details

### 1. Spoken language selector
- **Values**: "Automatic" first, then every language transcribe can be told to use, by name with
  the code ("Breton (br)"), sorted by name. The window is standard-library only, so the list is a
  table in `gui/pipeline.py`: Whisper's 99 languages (the existing `LANGUAGES` + `EXTRA_NAMES`
  cover about 65; complete them) plus any language with its own backend. A long readonly
  `ttk.Combobox` is tedious: let typing a letter jump to the first matching name (bind
  `<KeyPress>`), or put the languages used recently at the top.
- **Run**: speech gets `--language xx` on its transcribe step; song gets it on the transcribe
  step after `vocals`. Detection is skipped, which also saves its up to 8 Whisper windows.
- **Found panel**: shows "Language: Breton (br, chosen)" instead of a detected share, so it is
  clear nothing was detected.
- **Translation**: the chosen code is `--from` for translate (now `--from` is passed only for
  languages translate knows; keep that rule, and fall back to translate's own detection for the
  rest). "Translate afterwards" uses the language of the run shown, not the selector's current
  value, so changing the selector after a run does not mislabel the transcript.
- **Accent**: unchanged (English only); with a non-English language chosen, untick and disable
  "Accent recognition", or note on stderr that it was skipped. Open: which.
- **Model downloads**: choosing Swedish (or later Breton) selects the dedicated model on the
  first run; the window already announces model download sizes.
- Open: remember the last choice between starts? Default stays Automatic each start unless it
  proves annoying for someone who mostly listens to one rare language.
- Could be split into its own task if Breton waits; it does not depend on any Breton work.

### 2. Breton transcription
- **Measure first**: run several Breton clips (Common Voice `br`, a YouTube video in Breton)
  through transcribe and record what it detects and how `--language br` with large-v3 does.
  Keep URLs and ranges in the task's notes, as in 0005.
- **Candidates**: wav2vec2 XLS-R CTC models fine-tuned on Common Voice Breton, e.g.
  [`infinitejoy/wav2vec2-large-xls-r-300m-breton`](https://huggingface.co/infinitejoy/wav2vec2-large-xls-r-300m-breton)
  (card: WER 0.4455), `Marxav/wav2vec2-large-xlsr-53-breton`,
  `mrm8488/wav2vec2-large-xlsr-53-breton`; a Whisper fine-tune
  [`anuragshas/whisper-large-v2-br`](https://huggingface.co/anuragshas/whisper-large-v2-br)
  (would fit the existing Whisper path and conversion, like KB-Whisper).
  The paper [Speech-to-Text for Breton](https://www.researchgate.net/publication/386137301_Speech-to-Text_for_Breton)
  found wav2vec2 with a KenLM decoder clearly better than Whisper. Compare on the measured clips.
- A CTC model is a new backend: no punctuation and only character timings, so segments, lines,
  paragraphs and the formats (`--json`, `--srt`, LRC) need a mapping (cut at voice-activity
  pauses, which transcribe already finds). Its dependencies (transformers, pyctcdecode, kenlm)
  belong to transcribe only.
- Check each model's licence and size; pin revisions as for KB-Whisper.
- **Detection second opinion** (lower priority once the selector exists): when Whisper says
  `fr`, `cy` or `kw`, ask a language-ID model: SpeechBrain VoxLingua107 ECAPA
  (`speechbrain/lang-id-voxlingua107-ecapa`; SpeechBrain is already used by accent) or Meta
  MMS-LID (`facebook/mms-lid-*`). Verify both list Breton. Only worth it if the measurements show
  Breton is regularly misdetected and the selector is not enough.

### 3. Breton translation
- [`amurienne/kellag-m2m100`](https://huggingface.co/amurienne/kellag-m2m100-v0.2) (br→fr),
  [`amurienne/gallek-m2m100`](https://huggingface.co/amurienne/gallek-m2m100-v0.2) (fr→br),
  [`lgrobol/m2m100_418M_br_fr`](https://huggingface.co/lgrobol/m2m100_418M_br_fr).
- Open: NLLB-200's Breton coverage; whether the Ollama model handles Breton acceptably (then no
  new backend is needed).

## Done when
- In the window, choosing a spoken language transcribes in that language without detection
  (speech and song), the found panel says it was chosen, and translation uses it as source;
  Automatic behaves as before. Unit tests for the step plans with and without a chosen language.
- A Breton clip (Common Voice `br`, or a YouTube video in Breton), with Breton chosen or
  detected, transcribes with the new backend and translates to French and English.
- Unit tests for the new code; READMEs list the models with sizes and licences; CHANGELOG under
  Unreleased.
