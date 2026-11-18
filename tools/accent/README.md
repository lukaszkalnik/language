# accent

Guesses the English accent of the speaker in a file, a YouTube video or a WAV stream, using
SpeechBrain's pretrained [CommonAccent](https://arxiv.org/abs/2305.18283) classifiers.

```bash
accent interview.m4a
accent "https://www.youtube.com/watch?v=QXX-uwQ-VGw"           # first 10 minutes
accent "https://youtu.be/QXX-uwQ-VGw" --full                  # whole video
accent talk.mp4 --start 1:30 --end 4:00 -v                    # range + per-chunk table
vocals song.mp3 | accent -                                    # WAV on stdin
accent talk.mp4 --json --model xlsr
```

The top 3 accents go to stdout; video info, progress and the caveat note go to stderr.

| Option | Meaning |
| --- | --- |
| `--model ecapa\|xlsr` | `ecapa` (default) or the larger wav2vec2 XLSR model |
| `--start`, `--end` | range (for YouTube only that section is downloaded) |
| `--full` | YouTube without a range: no 10-minute cap |
| `-v`, `--verbose` | per-chunk table |
| `--json` | JSON with all probabilities and chunks |
| `--top N` | how many accents to list (default 3) |
| `--device auto\|cpu\|mps\|cuda` | `auto` = CUDA, then Apple MPS, then CPU |

## Models (model cards checked 2026-11-18)

| `--model` | Hugging Face ID | Download |
| --- | --- | --- |
| `ecapa` | [`Jzuluaga/accent-id-commonaccent_ecapa`](https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa) | ~85 MB |
| `xlsr` | [`Jzuluaga/accent-id-commonaccent_xlsr-en-english`](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english) | ~1.3 GB + `facebook/wav2vec2-large-xlsr-53` ~1.3 GB |

The cards say only `pip install speechbrain` (no version) and use the old
`speechbrain.pretrained` imports; with SpeechBrain 1.x these live in `speechbrain.inference`.
SpeechBrain is pinned to 1.1.2. On Windows models are copied instead of symlinked.

Both models share 16 labels (in a different order): african, australia, bermuda, canada,
england, hongkong, indian, ireland, malaysia, newzealand, philippines, scotland, singapore,
southatlandtic (sic), us, wales.

The ECAPA head outputs cosine similarities; they are turned into probabilities with
`softmax(30 × cosine)`, the margin-softmax scale the model was trained with. Expect very
confident numbers: they reflect training, not real certainty.

## How it works

Audio is resampled to 16 kHz mono, split into 20 s chunks (a tail under 5 s joins the previous
chunk), chunks below -45 dBFS are skipped, silent 30 ms frames inside chunks are trimmed, each
chunk is classified, and the probabilities are averaged weighted by speech time.
