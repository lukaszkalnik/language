# 0002 No network checks for cached models

Status: in progress

## Problem
Libraries given a Hugging Face repo ID ask the Hub for the latest revision even when the model is
cached. On a network with broken IPv6 that request hangs for minutes (transcribe stalled about
ten minutes after "Mode: speech"). Transcribe's Whisper backends and translate's NLLB now load
cached models through `ingest.models.hf_local_path`; the other loads still ask.

## Proposal
Make every model load work offline once the model is cached, in all four tools. A first run
still downloads as before.

## Details
Found by reading the installed libraries (SpeechBrain 1.1.2, Demucs 4.1.0, pyannote.audio 4.1.0,
transformers 5.23, huggingface_hub 2.4):

- **SpeechBrain** `from_hparams(source=..., savedir=...)` returns a file that already exists in
  `savedir` without asking the Hub (`FetchConfig.allow_updates` is off). So accent's ECAPA and
  transcribe's ECAPA speaker model are already fine.
- **accent XLSR**: its yaml loads the base model `facebook/wav2vec2-large-xlsr-53` through
  transformers (`AutoConfig.from_pretrained(repo, cache_dir=...)`), which asks the Hub every time.
- **vocals (Demucs)**: `demucs.hf.get_hf_model` calls `hf_hub_download(repo, f"{name}.yaml")` and
  one per `.safetensors` without `local_files_only`, so every run asks. If that fails it silently
  falls back to Meta's legacy download server (dl.fbaipublicfiles.com).
- **transcribe speaker labels (pyannote)**: `Pipeline.from_pretrained` downloads each file with
  `hf_hub_download`, asking each time. pyannote also sends usage telemetry to
  `otel.pyannote.ai` by default (`PYANNOTE_METRICS_ENABLED`), another request at every load.

Approach:
- `ingest.models.hub_offline()`: a context manager that sets huggingface_hub's offline flag
  (`constants.HF_HUB_OFFLINE`, which transformers also reads) for the duration of a load. Hub
  calls then use the cache at once and never open a connection.
- Callers use it only when the model is cached; if the offline load fails anyway (a file
  missing after an interrupted download), they load again online.
- Demucs: cached means the bag's yaml and every `.safetensors` it lists are in the HF cache,
  checked before loading, so the AWS fallback never runs for a cached model.
- pyannote: set `PYANNOTE_METRICS_ENABLED=false` unless the user set it.
- Tests: unit tests for `hub_offline` and the cache checks, with a fake HF cache. Integration
  tests per tool: a second run with `HTTPS_PROXY`/`HTTP_PROXY` pointing at a local listener
  that counts connections; it must see none.

## Done when
With models cached and every connection refused (the integration tests above), every tool runs
without delay: accent (ecapa, xlsr), vocals, transcribe (with and without speaker labels),
translate (NLLB).
