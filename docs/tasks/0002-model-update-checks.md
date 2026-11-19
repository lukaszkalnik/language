# 0002 No network checks for cached models

Status: concept

## Problem
Libraries given a Hugging Face repo ID ask the Hub for the latest revision even when the model is
cached. On a network with broken IPv6 that request hangs for minutes (transcribe stalled about
ten minutes after "Mode: speech"). Transcribe's Whisper backends and translate's NLLB now load
cached models through `ingest.models.hf_local_path`; accent (SpeechBrain) and vocals (Demucs)
have not been checked.

## Proposal
Make every model load work offline once the model is cached, in all four tools.

## Details
- SpeechBrain `EncoderClassifier.from_hparams(source=..., savedir=...)` in `tools/accent`.
- Demucs 4.1 weights from the HF Hub in `tools/vocals`.
- Possibly a test that loads with `HF_HUB_OFFLINE=1` after a first run (integration).

## Done when
With the network off (or `HF_HUB_OFFLINE=1`) and models cached, every tool runs without delay.
