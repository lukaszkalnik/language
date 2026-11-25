# 0007 Faster and clearer updates

Status: concept

## Problem
After an update the app runs its setup again (`setup.needed()` compares the version in
`installed.json` with the app's). That is mostly reuse already: Python stays, PyTorch and the
other large dependencies come from uv's cache, Deno and ffmpeg too, and models are never
touched; only our own packages are reinstalled (`--reinstall-package` for each tool and
`ingest`). It takes a few minutes. But:

- **The window shows the first-start welcome** (`app.py`, `setup.INTRO`): "Before the first
  use", "about 2–3 GB", "10–30 minutes", and waits for **Set up**. For an update that is wrong
  and alarming.
- **Every tool is reinstalled**, even when its sources did not change between versions.
- **Fast updates depend on uv's cache** in its default place (`~/Library/Caches/uv`,
  `%LOCALAPPDATA%\uv\cache`). If the user or a disk-cleaning app empties it, the next update
  downloads the 2–3 GB again.

## Proposal
- An update gets its own short screen and starts on its own: "Updating the tools for version
  X (a few minutes; nothing large is downloaded again)", with the same progress line as setup.
  The first-start welcome stays for the first start only (no `installed.json`).
- Setup remembers a fingerprint of each tool's sources (its files and those of `ingest`, plus
  the lock entries it depends on) in `installed.json` and reinstalls only the tools whose
  fingerprint changed; Deno and ffmpeg only when their pinned versions changed. An update that
  changes only the window then needs no install step at all.

## Details
- Files: `tools/gui/src/gui/setup.py` (marker contents, `needed()`, `commands()`),
  `tools/gui/src/gui/app.py` (update screen), unit tests for which steps run for a given old
  marker.
- Old markers hold only `{"version": ...}`: treat them as "reinstall everything" once.
- Open: put uv's cache under the app's data folder (`UV_CACHE_DIR` in `tool_env()`) so it
  survives cache cleaners? Then deleting the app's folder also deletes the cache, and the
  folder grows by the cache's size (several GB, prunable with `uv cache prune`). Or leave it
  and accept the rare full download.
- Open: a failed update must leave the previous tools usable or say clearly that setup has to
  be run again (today a failure leaves the marker on the old version, so setup reruns).
- `RELEASE_NOTES.md` and the changelog's "after updating" line change accordingly.

## Done when
Updating from the previous release on a Mac shows the update screen, starts by itself and
finishes in under a minute when only the window changed; a version that changes one tool
reinstalls only that tool; a first start still shows the welcome. Unit tests cover the step
selection.
