# translate

Translates plain text or `transcribe --json` output locally, line by line.

```bash
transcribe song.mp3 | translate --to de
transcribe podcast.mp3 --speakers --json | translate --to pl --json > podcast.pl.json
translate lyrics.txt --to en --only-translation
translate notes.txt --to de --translator nllb --from pl     # offline, no Ollama
```

- **Input** from a file or stdin (`-` or nothing). Plain text vs. transcribe JSON is detected
  automatically.
- **Output**: numbered rows with the original and the translation side by side (wrapped so the
  rows stay aligned); `--only-translation` for the translation alone (blank lines and
  `Speaker N:` labels kept); `--json` copies the input JSON and adds `translation` and
  `source_language` to every segment, so timestamps and speaker labels survive.
- **Source language**: `--from`, else per segment from the JSON (so a podcast that switches
  language is handled per segment), else detected offline with py3langid. Lines already in the
  target language are kept unchanged.
- **Long inputs** are sent in chunks of whole paragraphs (~2400 characters), each with the last
  few lines of the previous chunk and their translations as context, so meaning carries across
  chunk borders. Songs are sent whole so context, idioms and wordplay survive.

## Backends

| `--translator` | What | Notes |
| --- | --- | --- |
| `llm` (default) | a local model through [Ollama](https://ollama.com), `--llm-model` (default `gemma4:12b`, 7.7 GB) | asked for a faithful line-by-line translation, not a poetic rewrite; numbered lines keep alignment, missing lines are re-requested |
| `nllb` | Meta NLLB-200 via transformers, `--nllb-model 600m` (2.5 GB, default) or `1.3b` (5.5 GB) | offline fallback; sentence by sentence, no cross-line context |

### Breton

Neither backend knows Breton well (NLLB-200 has none; gemma4:12b mostly guesses), so Breton
lines are first translated into French with
[`lgrobol/m2m100_418M_br_fr`](https://huggingface.co/lgrobol/m2m100_418M_br_fr) (MIT, 1.9 GB,
downloaded on first use), an M2M100 trained on Breton-French text; the chosen backend then
takes the French on to the target language. Into French no backend is needed (no Ollama).
Hesitations such as "euh" are left out first. On Breton TV subtitles it scored chrF 50 against
their French subtitles, gemma4:12b directly 32. Translating *into* Breton is not supported.
`--json` names both steps, e.g. `m2m100-br-fr+llm:gemma4:12b`.

If Ollama is not running or the model is not pulled, translate says so and how to fix it:

```bash
brew install ollama      # or the app from https://ollama.com
ollama serve             # or open the app
ollama pull gemma4:12b   # 7.7 GB; gemma4:e4b (6.6 GB) needs less memory
```

On a network with broken IPv6, `ollama pull` can time out (its downloads try IPv6 first);
pulling on another network or with IPv6 disabled works around it.
