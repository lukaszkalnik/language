# transcribe

Detects the language and transcribes speech or singing with Whisper large-v3, locally
(Swedish with KB-Whisper, a Whisper trained on Swedish).
Works on separated vocals from `vocals`, on full songs, and on podcasts, interviews and lectures.

```bash
transcribe vocals.wav --lrc > song.lrc
vocals "https://youtu.be/..." | transcribe - | translate --to de
transcribe podcast.mp3 --speakers --md > podcast.md
transcribe podcast.mp3 --speakers --names "Anna,Ben" --json | translate --to pl --json
transcribe lecture.m4a --start 10:00 --end 25:00 --srt > part.srt
```

The transcript goes to stdout; mode, languages, progress and warnings go to stderr.

## What it does

1. **Voice activity** with Silero VAD (block by block, flat memory). On separated vocals the
   vocal stem's level is used instead, because Silero (trained on speech) misses sung notes.
2. **Mode** (`--mode auto|song|speech`): held, pitched notes and a continuous music bed mean
   *song*; gliding pitch with pauses means *speech*. The choice and the reason go to stderr. Song
   mode on a full, unseparated mix prints a hint to run `vocals` first.
3. **Language** on up to 8 voiced 30 s windows spread across the whole file (song intros are
   often instrumental, podcasts start with jingles). The main language and any others found go to
   stderr and `--json`. `--language xx` skips this. In speech mode a file that switches language
   is transcribed passage by passage with each passage's own language.
4. **Transcription** with Whisper: MLX (`mlx-whisper`) on Apple Silicon, `faster-whisper`
   elsewhere. `--model auto` (default) picks the language's own model if there is one
   (Swedish: KB-Whisper) and large-v3 otherwise; `--model large-v3`, `turbo` or `kb-whisper`
   choose one. A file transcribed passage by passage in several languages keeps large-v3.
5. **Hallucination control**: VAD, `condition_on_previous_text=False`, temperature fallback,
   segments without voice / known filler phrases ("Thanks for watching", subtitle credits) /
   music-only text dropped, repetition loops collapsed, and low-confidence lines marked `[?]`
   rather than hidden (`--no-marks` to remove the marks).
6. **Long files**: speech is cut into ≤ 28 s pieces at pauses, transcribed piece by piece with
   progress and ETA on stderr, and saved every ~10 minutes of audio in the shared cache, so an
   interrupted run resumes where it stopped. Only the current piece is in memory.
7. **Speakers** (`--speakers`, speech mode): see below.

## Output

| Flag | Output |
| --- | --- |
| (none) | song: one sung line per line; speech: paragraphs (at pauses, speaker or topic changes) |
| `--lrc` | LRC lyrics (song mode) |
| `--srt`, `--vtt` | subtitles |
| `--json` | source, mode, languages, speakers, and per segment: start, end, text, language, confidence, low_confidence, speaker, paragraph (`--words` adds word timings) |
| `--md` | Markdown transcript with a timestamp every 5 minutes (speech mode) |

## Speaker labels

`--speakers` uses [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
(pyannote.audio 4.1). Its weights are gated, so it needs a one-time setup:

1. Create a free account at huggingface.co and accept the conditions on the model page.
2. Create a read token at https://huggingface.co/settings/tokens.
3. Run `hf auth login` once (or set `HF_TOKEN`).

Without that, transcribe prints these steps and falls back to clustering SpeechBrain ECAPA
speaker embeddings (`speechbrain/spkrec-ecapa-voxceleb`, not gated), which works well for clean
two-person conversations but is weaker on overlapping speech. `--num-speakers N` helps both.
Speakers are named "Speaker 1", "Speaker 2"... or by `--names "Anna,Ben"`, in order of first appearance.

## Models and downloads

| Model | Size | When |
| --- | --- | --- |
| `mlx-community/whisper-large-v3-mlx` | 3.1 GB | default on Apple Silicon |
| `mlx-community/whisper-large-v3-turbo` | 1.6 GB | `--model turbo` |
| [`KBLab/kb-whisper-large`](https://huggingface.co/KBLab/kb-whisper-large) (Apache-2.0) | 3.2 GB download, 3.1 GB kept | Swedish; on Apple Silicon converted to MLX on first use (runs in float32: about 6 GB of memory, ~2× slower than large-v3) |
| `Systran/faster-whisper-large-v3` | 3.1 GB | default elsewhere |
| `KBLab/kb-whisper-large` (CTranslate2 copy) | 3.1 GB | Swedish elsewhere |
| Silero VAD | 2 MB | bundled with the `silero-vad` package |
| `pyannote/speaker-diarization-community-1` | 34 MB | `--speakers` with a token |
| `speechbrain/spkrec-ecapa-voxceleb` | 83 MB | `--speakers` fallback |

The size is printed the first time a model is fetched.
