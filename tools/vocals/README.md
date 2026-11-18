# vocals

Separates the vocals from the music with [Demucs](https://github.com/adefossez/demucs) and writes
a WAV of the vocals only, at the source's sample rate.

```bash
vocals song.mp3 -o vocals.wav
vocals "https://www.youtube.com/watch?v=..." | transcribe -
vocals song.flac --stem instrumental -o karaoke.flac
vocals song.mp3 --model htdemucs            # ~3x faster, slightly lower quality
```

- Piped or redirected stdout gets a WAV stream; a terminal never gets binary: the result is
  written to `-o FILE` or `<name>.vocals.wav`, and the path is printed on stderr.
- All stems are cached in the shared cache (keyed by input, range and model), so a second run,
  or asking for another `--stem` of the same song, is instant. `--no-cache` recomputes.
- The output WAV carries a RIFF INFO tag (`ISFT=vocals/<model>/<stem>`) so `transcribe` knows
  its input is already separated.

## Models

| `--model` | Weights (Hugging Face) | Stems | Notes |
| --- | --- | --- | --- |
| `htdemucs_ft` (default) | `adefossez/HTDemucs-ft`, 336 MB | drums, bass, other, vocals | best Demucs quality; ~0.6x real time on an M1 (MPS) |
| `htdemucs` | `adefossez/HTDemucs`, 84 MB | same | ~5x real time on an M1 |
| `htdemucs_6s` | `adefossez/HTDemucs-6s`, 55 MB | + guitar, piano | |
| `mdx_extra` | `adefossez/Demucs-mdx_extra`, 669 MB | same as 4-stem | older hybrid model |

`instrumental` is every stem except vocals, summed. Demucs 4.1.0 runs on Apple MPS (falls back
to CPU if an MPS op fails), CUDA or CPU.

Why Demucs: RoFormer models (via `audio-separator`) score higher on vocal SDR, but the current
`audio-separator` release failed to import / list models in a clean environment here, while
Demucs 4.1.0 installs cleanly, runs on MPS and is maintained again (weights moved to the HF Hub).
