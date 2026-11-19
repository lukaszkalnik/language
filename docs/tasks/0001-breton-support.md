# 0001 Breton support

Status: concept

## Problem
Whisper large-v3 nominally knows Breton (brezhoneg, `br`), but transcribes it poorly, and the
translation backends have no confirmed Breton coverage.

## Proposal
Transcribe Breton with a model trained for it, and translate Breton ↔ French (other languages
through French). The window lists Breton where it is supported.

## Details
- **Speech recognition**: wav2vec2 XLS-R CTC models fine-tuned on Common Voice Breton, e.g.
  [`infinitejoy/wav2vec2-large-xls-r-300m-breton`](https://huggingface.co/infinitejoy/wav2vec2-large-xls-r-300m-breton)
  (card: WER 0.4455), `Marxav/wav2vec2-large-xlsr-53-breton`,
  `mrm8488/wav2vec2-large-xlsr-53-breton`; a Whisper fine-tune
  [`anuragshas/whisper-large-v2-br`](https://huggingface.co/anuragshas/whisper-large-v2-br).
  The paper [Speech-to-Text for Breton](https://www.researchgate.net/publication/386137301_Speech-to-Text_for_Breton)
  found wav2vec2 with a KenLM decoder clearly better than Whisper. Compare on real clips.
- CTC output has no punctuation and only character timings: segments, lines and the existing
  formats (`--json`, `--srt`, paragraphs) need a sensible mapping.
- **Translation**: [`amurienne/kellag-m2m100`](https://huggingface.co/amurienne/kellag-m2m100-v0.2) (br→fr),
  [`amurienne/gallek-m2m100`](https://huggingface.co/amurienne/gallek-m2m100-v0.2) (fr→br),
  [`lgrobol/m2m100_418M_br_fr`](https://huggingface.co/lgrobol/m2m100_418M_br_fr).
  Open: NLLB-200's Breton coverage; whether the Ollama model handles Breton acceptably.
- Check each model's licence and size.

## Done when
A Breton clip (Common Voice `br`, or a YouTube video in Breton) transcribes with the new backend
and translates to French and English; unit tests for the new code; READMEs list the models.
