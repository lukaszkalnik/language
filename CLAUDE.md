# Working on this repository

- Work directly on `main`; no feature branches.
- Commit after every atomic change (one logical change per commit, tests passing), and push.
- `uv run pytest` at the root runs all unit tests (no network, no models); keep it green.
  Integration tests: `uv run --package <tool> pytest -m integration tools/<tool>`.
- `uv run ruff check .` before committing.
- Conventions every tool follows: data on stdout, everything for humans on stderr;
  `-` reads a WAV stream from stdin; one `error:` line and a non-zero exit code on failure.
- Heavy model dependencies belong to the tool that needs them, never to `ingest` or the root.
- Every release (a `v*` tag) gets a section `## <version>` in `CHANGELOG.md`, written for the
  people using the app: most important changes first, in plain words saying how each change
  affects them (what they can now do, what got faster and by how much, what they must do after
  updating), not which modules changed. The release workflow puts that section into the
  GitHub release notes and fails without it.
- Task concepts and specifications live in `docs/tasks/NNNN-short-title.md` (four-digit number,
  never reused); their order is in `ROADMAP.md`. See `docs/tasks/README.md`.
