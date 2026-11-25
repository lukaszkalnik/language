# Working on this repository

- Work directly on `main`; no feature branches.
- Commit after every atomic change (one logical change per commit: parts that only make sense together; separate fixes are separate commits; tests passing), and push.
- `uv run pytest` at the root runs all unit tests (no network, no models); keep it green.
  Integration tests: `uv run --package <tool> pytest -m integration tools/<tool>`.
- `uv run ruff check .` before committing.
- Conventions every tool follows: data on stdout, everything for humans on stderr;
  `-` reads a WAV stream from stdin; one `error:` line and a non-zero exit code on failure.
- Heavy model dependencies belong to the tool that needs them, never to `ingest` or the root.
- The user's Mac has 16 GB of memory: run one model-heavy process at a time (Whisper, Ollama,
  translation models, measurements), never several in parallel or in the background next to
  each other, and stop servers you started (e.g. `ollama serve`) when done.
- Releases happen only when the user asks for one; never offer or suggest a release after a change.
- Every release (a `v*` tag) gets a section `## <version>` in `CHANGELOG.md`, written for the
  people using the app: most important changes first, in plain words saying how each change
  affects them (what they can now do, what got faster and by how much, what they must do after
  updating), not which modules changed. The release workflow puts that section into the
  GitHub release notes and fails without it. Collect changes under `## Unreleased` meanwhile and
  rename it to the version when tagging.
- Task concepts and specifications live in `docs/tasks/NNNN-short-title.md` (four-digit number,
  never reused); their order is in `ROADMAP.md`. See `docs/tasks/README.md`.
- `ROADMAP.md` is one numbered list without sections, ordered by implementation time (done
  tasks in the order they were built, then the planned ones). A new task goes at the end, even
  when work on it starts at once, unless it fits better next to related tasks. Each entry
  starts with its status emoji (💡 concept, 📝 specified, 🚧 in progress, ⏸️ parked, ✅ done,
  ❌ dropped), kept in step with the task file's `Status:` line.
- When working through the roadmap, skip parked (⏸️) and dropped (❌) tasks.
- Architecture and code-quality rules: [docs/architecture.md](docs/architecture.md). When one of
  them contradicts sound domain modeling, do not bend the model to fit the rule: ask the user,
  and propose how to adapt the rule instead.
