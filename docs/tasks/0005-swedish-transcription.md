# 0005 Better Swedish transcription

Status: done (2026-11-20, commits from 140b9c4 on main; not released yet)

## Problem
Whisper large-v3 transcribes Swedish songs with many mishearings and sometimes loses whole
passages. Tested on the first two minutes of three songs (separated vocals, song mode):

| Song | Singing covered | Notes |
| --- | --- | --- |
| Veronica Maggio, *Jag kommer* | 84% | mostly right; mishearings ("losing the sky") |
| Håkan Hellström, *Känn ingen sorg för mig Göteborg* | 51–54% | intro chant and the 1:36–1:54 chorus missing; first lines differ from run to run |
| Garmarna, *Herr Mannelig* (old Swedish) | 93% | many words misheard ("fagerungers vän" for "fager ungersven") |

The Håkan chorus comes out right when 1:36–2:00 is transcribed on its own; inside the whole run
Whisper writes a made-up "Tack för att du tittade…" at 1:54 and continues after it, losing the
lines before. Turning off Whisper's no-speech skip and shorter clips (one per phrase) did not help.

## Proposal
1. **A Swedish model**: when the detected (or chosen) language is Swedish, transcribe with a
   Whisper fine-tuned on Swedish instead of large-v3.
2. **Second pass over lost singing**: after transcribing, find sung stretches (from the vocal
   track's loudness) that no line covers, and transcribe each again on its own.

## Details
- **Candidate model**: KBLab's KB-Whisper (National Library of Sweden), e.g.
  [`KBLab/kb-whisper-large`](https://huggingface.co/KBLab/kb-whisper-large), trained on a large
  amount of Swedish speech, with smaller sizes too. To check: licence, download size, whether an
  MLX conversion exists (or can be made with `mlx_whisper`'s converter) and a CTranslate2 one for
  faster-whisper, and how it does on singing, not only speech.
- **Measure before and after** on the songs above and a few spoken clips (news, podcast): lines
  right / misheard / missing, and coverage of the singing. Keep the clips' URLs and ranges in
  the test notes so runs can be repeated.
- **Model per language**: transcribe already detects the language before transcribing, so the
  choice can follow it (`--language sv` or detection). A small table language → model also
  serves 0001 (Breton). The window needs nothing new; the first Swedish run announces the
  download size like any model.
- **Second pass**: reuse the cached vocals and voice regions; a stretch counts as lost when no
  kept line overlaps most of it. Cost: one extra Whisper call per lost stretch, usually few.
- **Run-to-run differences**: Whisper retries unsure windows with randomness (temperature
  fallback), so the same file can come out differently. Check whether a fixed seed (MLX) makes
  results repeatable.
- Old Swedish (ballads, folk songs) will stay hard for any model; not a goal by itself.

## Done when
- On the songs above, fewer misheard lines and no missing chorus, measured against the table.
- Spoken Swedish no worse than with large-v3.
- READMEs and CHANGELOG updated; the model's size and licence in the transcribe README.

## Result (2026-11-20)

Built: `--model auto` (default) uses KB-Whisper large for Swedish (`KBLab/kb-whisper-large` at
commit `d5d5984`, the "standard" style; Apache-2.0). On MLX it is converted from KBLab's
transformers weights on first use (`transcribe/convert.py`). faster-whisper uses the CTranslate2
copy in the same repo. The second pass over lost singing runs in song mode on separated vocals and
keeps only lines with confidence ≥ 0.3.

Findings:
- mlx-whisper 0.4.3's timestamp rules have bugs (`transcribe/mlx_fixes.py`). KB-Whisper hit them
  in almost every window: a stray "!" (token 0), NaN log-probs, collapsed timestamps.
- KB-Whisper overflows in float16 on MLX (word timings all 0), so it runs in float32, about 2×
  slower than large-v3 (a 2-minute song: ~75–130 s against ~35 s).
- The weights correlate 0.76–0.99 with large-v3's, so the conversion is right. Its timestamps are
  whole seconds. Word timings use the default alignment heads, like large-v3 on MLX.
- Repeatable: with `mx.random.seed(0)`, two runs from scratch give identical JSON (Håkan, with
  both models). The 36–52% spread of large-v3 on Håkan came from the random fallback.
- "strict" revision (more verbatim), on the same songs: coverage 85/63/93%; a few words better
  ("betsel"), a few worse ("vill" for "väl"). On a conversation it keeps more fillers ("alltså",
  "så"). Not better overall, and it has no CTranslate2 copy: we stay with standard.
- faster-whisper path checked on CPU on this Mac (Ekot 0:00–1:00 with `--device cpu`): downloads
  the pinned CTranslate2 copy (3.1 GB) and transcribes like the MLX path.

Songs, first two minutes, separated vocals, song mode (MLX). Coverage = share of the vocal
stem's sung time under a line:

| Song | large-v3 | KB-Whisper + second pass |
| --- | --- | --- |
| Maggio, *Jag kommer* (youtu.be/VjZwz3yxS9I, 0:00–2:00) | 84% | 85%; "Lucy in the sky", "Natten är ung", "jag är kär", "taxitrappen" now right |
| Håkan, *Känn ingen sorg…* (youtu.be/-fusSPk4j0k, 0:00–2:00) | 36–52% | 59%; chorus 1:39–1:55 present, "Göteborg" right |
| Garmarna, *Herr Mannelig* (youtu.be/X_BpmQqErks, 0:00–1:56) | 94% | 92%; "upprann", "Bergatrollet", "Heder vill jag giva", "uti munnen" right; "Fagerungers vän" still wrong |

Speech (speech mode; words in the transcript, then the differences):

| Clip | large-v3 | KB-Whisper |
| --- | --- | --- |
| Ekot, *Kvinnoregistret* (youtu.be/J4AWnq4MfNE, 0:00–2:00) | 164 | 165; "dömd" (not "drömd"); one likely mistake ("Henning Holm" for "henne") |
| SVT, *Fyra scenarion* (youtu.be/7vw4ucXPW9U, 0:00–2:00) | 256 | 262; "Tidöpartierna", "Ulf Kristersson"; spoken words large-v3 left out |
| Enkelt Svenska Podcast (youtu.be/uPh-bkVxbWE, 1:00–4:00) | 321 | large-v3 skipped a passage of six sentences that KB-Whisper has |
| Framgångspodden (youtu.be/qmenxr1hOXs, 10:00–13:00) | – | same content; leaves out fillers ("ju", "så", "uppenbarligen"), the "standard" style condenses |

Not solved: Håkan's intro chant (0:00–0:13). On its own, KB-Whisper writes a guess with
confidence 0.06 ("Michael Jackson"), now dropped, and large-v3 writes a subtitle credit. It is
probably too unclear on the separated stem.
