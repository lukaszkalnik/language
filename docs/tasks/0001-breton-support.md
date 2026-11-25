# 0001 Breton support

Status: done (2026-11-25, commits d283f6f to 3f67201 on main; not released yet)

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

## Result (2026-11-25)

Built:
- **Transcription**: `--model auto` uses
  [`gweltou/whisper-small-br_260315`](https://huggingface.co/gweltou/whisper-small-br_260315)
  (MIT, Whisper small, ~80 h of Breton from the Anaouder project: Brezhoweb, dialect archives,
  Common Voice…) at commit `32f4d9e` for `br`. It fits the existing Whisper path: converted to
  MLX on first use (float16 is fine), and to CTranslate2 for faster-whisper with ctranslate2's
  converter (transformers added to transcribe off Apple Silicon only; checked on CPU here).
  The fine-tunes are now one table in `whisper.py`. No CTC backend was needed.
- **Translation**: Breton lines go into French with
  [`lgrobol/m2m100_418M_br_fr`](https://huggingface.co/lgrobol/m2m100_418M_br_fr) (MIT, 1.9 GB,
  commit `f4b23bd`), then on from French with the chosen backend; fillers ("euh") are dropped
  first. Into French no backend is set up, so it runs without Ollama.
- **Window**: the chosen spoken language is translate's `--from` (Breton included).

### Transcription measurements
80 random clips of Common Voice 22 `br` test (`Bretagne/common_voice_22.0_br`, read speech, 4:46
in all) and four stretches of natural speech with Breton subtitles as reference: Wikitongues
*Yann speaking Breton* (youtu.be/IyBNVrLQ2-A, whole 2:52) and three Brezhoweb videos from 2026,
after the fine-tunes' training data: *Bevañ war enez Eusa* (youtu.be/7iGkrB70Ers, 1:00–4:00),
*Charlie Grall* (youtu.be/ODK24KWSal0, 1:00–4:00), *Begerel* (youtu.be/h6xQ1sYMa1g, 0:00–3:00).
mlx-whisper with the repo's decoding settings; WER/CER after lower-casing and removing punctuation.

| Model | CV WER / CER | Natural speech WER / CER | Time on CV |
| --- | --- | --- | --- |
| large-v3, `--language br` | 81% / 32% | not measured (stopped: memory) | 658 s |
| large-v3, detection | 107% / 66% (detected `br` in 2 of 80 clips, `fr` in 53) | – | 758 s |
| `Bretagne/whisper-large-v3-turbo-…breton` (no licence) | 23% / 17% | 44% / 31% | 315 s |
| **`gweltou/whisper-small-br_260315`** | **25% / 11%** | **39% / 23%** | **239 s** |

The natural-speech WER overstates the errors: most differences are elisions in the subtitles
('omp/omp), fillers ("euh") and numbers in words ("seizh kant" for 700). The old
`anuragshas/whisper-large-v2-br` (card WER 38) was not tried: it ships only a pickle.

### Translation measurements
The three Brezhoweb videos have Breton and French subtitles with the same timings: 317 pieces
of three cues each, Breton → French, scored against the French subtitles.

| Model | chrF | BLEU | Time |
| --- | --- | --- | --- |
| **`lgrobol/m2m100_418M_br_fr`** | **50.4** | **23.9** | 396 s (CPU) |
| `amurienne/kellag-m2m100-v0.2` (MIT) | 43.0 | 20.2 | 405 s; drops half sentences |
| Ollama `gemma4:12b`, directly from Breton | 31.5 | 10.9 | 2011 s; mostly guesses |

NLLB-200 has no Breton.

### Open
- **Detection**: Whisper takes Breton for French; choosing Breton (0006) is needed. A second
  opinion from a language-ID model (VoxLingua107 lists Breton) was not built.
- **Mixed-language warning**: on Yann's clip, chosen `br`, detection found br 53%, oc 16%, fr
  16% and warned. As expected in 0006 this will be common for Breton; Breton interviews often
  do switch to French, so the warning stays.
- **Into Breton** (`amurienne/gallek-m2m100`, fr→br, MIT) is not built; translate refuses
  `--to br`, and Breton is not in the window's target list.
- Not checked by eye in the window itself; the window's steps are unit-tested.
