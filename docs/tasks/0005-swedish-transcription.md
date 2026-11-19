# 0005 Better Swedish transcription

Status: concept

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
