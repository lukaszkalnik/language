# 0001 Breton support

Status: concept

## Problem
Whisper large-v3 nominally knows Breton (brezhoneg, `br`), but transcribes it poorly, and the
translation backends have no confirmed Breton coverage. Language detection is the first hurdle:
Whisper saw little Breton in training and will likely call it French (or Welsh, Cornish), then
transcribe it as that language. Choosing the language in the window is task
[0006](0006-spoken-language-selector.md); this task builds on it.

## Proposal
1. **Breton transcription** with a model trained for it, picked through transcribe's
   language → model table (`--model auto`, built for Swedish in 0005) when the language is `br`,
   chosen or detected.
2. **Breton translation** Breton ↔ French (other languages through French). Breton appears in
   the "Translate to" list once a backend covers it.

## Details

### 1. Breton transcription
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
- **Detection second opinion** (lower priority once the language can be chosen, 0006): when Whisper says
  `fr`, `cy` or `kw`, ask a language-ID model: SpeechBrain VoxLingua107 ECAPA
  (`speechbrain/lang-id-voxlingua107-ecapa`; SpeechBrain is already used by accent) or Meta
  MMS-LID (`facebook/mms-lid-*`). Verify both list Breton. Only worth it if the measurements show
  Breton is regularly misdetected and choosing it in the window (0006) is not enough.

### 2. Breton translation
- [`amurienne/kellag-m2m100`](https://huggingface.co/amurienne/kellag-m2m100-v0.2) (br→fr),
  [`amurienne/gallek-m2m100`](https://huggingface.co/amurienne/gallek-m2m100-v0.2) (fr→br),
  [`lgrobol/m2m100_418M_br_fr`](https://huggingface.co/lgrobol/m2m100_418M_br_fr).
- Open: NLLB-200's Breton coverage; whether the Ollama model handles Breton acceptably (then no
  new backend is needed).

## Done when
- A Breton clip (Common Voice `br`, or a YouTube video in Breton), with Breton chosen or
  detected, transcribes with the new backend and translates to French and English.
- Unit tests for the new code; READMEs list the models with sizes and licences; CHANGELOG under
  Unreleased.
