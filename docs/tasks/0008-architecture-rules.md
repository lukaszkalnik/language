# 0008 Architecture and code-quality rules

Status: in progress

## Problem
The rules that keep the workspace healthy lived in people's heads and in a few lines of
`CLAUDE.md`. Release 0.4.0 failed because 15 unit tests imported `huggingface_hub`, `torch`,
`yaml` and `demucs`: the shared local `.venv` had them (every `uv sync --package <tool>`
installs into it), CI's root environment did not, so the tests passed locally and failed in the
release build. Nothing checked that the root unit tests run without model libraries, that tools
import only lazily what is heavy, or that tools stay independent of each other.

## Proposal
Write the rules down in one place ([docs/architecture.md](../architecture.md), linked from
`CLAUDE.md`) and enforce the ones a machine can check, inside the two commands everyone already
runs:

- `uv run pytest`
  - blocks every installed library that is not part of the root environment (the `dev` group
    and what it pulls in), so a unit test that needs a model library fails locally just as in
    CI, unless it skips itself (`pytest.importorskip`);
  - `tests/test_architecture.py` reads the sources and checks the dependency direction (tools
    use `ingest`, never each other; `ingest` uses no tool; the window imports no tool and
    no `ingest`) and that modules import heavy libraries only inside functions.
- `uv run ruff check .` with more rule sets (comprehensions, simplifications, implicit string
  concatenation, naming, ruff's own rules) whose findings were few and fixed in place.

## Details
- Files: `conftest.py` (the blocker), `tests/test_architecture.py`, `pyproject.toml` (ruff,
  `testpaths`), `docs/architecture.md`, `CLAUDE.md`.
- `LANGUAGE_TEST_MODEL_LIBS=1 uv run pytest` lifts the block to run the tests that need model
  libraries where they are installed; integration runs (`-m integration`) are not blocked.
- Left out on purpose, because they would change much of the existing code: `PTH` (pathlib
  instead of `open()`), `BLE` (the broad `except Exception` around model loads are deliberate
  fallbacks), `T20` (the CLIs print), `PL`, `TRY`, `EM`, `FBT`, `S`. They can be adopted later
  if the user wants them.

## Done when
- The rules are in `docs/architecture.md` and `CLAUDE.md` points to them.
- `uv run pytest` fails when a unit test or a module-level import needs a model library, or a
  tool imports another tool; `uv run ruff check .` runs the stricter rule sets; both are green.
