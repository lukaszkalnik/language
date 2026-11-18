# Working on this repository

- Work directly on `main`; no feature branches.
- Commit after every atomic change (one logical change per commit, tests passing), and push.
- `uv run pytest` at the root runs all unit tests (no network, no models); keep it green.
  Integration tests: `uv run --package <tool> pytest -m integration tools/<tool>`.
- `uv run ruff check .` before committing.
- Conventions every tool follows: data on stdout, everything for humans on stderr;
  `-` reads a WAV stream from stdin; one `error:` line and a non-zero exit code on failure.
- Heavy model dependencies belong to the tool that needs them, never to `ingest` or the root.
