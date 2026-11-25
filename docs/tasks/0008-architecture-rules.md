# 0008 Architecture and code-quality rules

Status: done

## Problem
The rules that keep the workspace healthy lived in people's heads and in a few lines of
`CLAUDE.md`. Release 0.4.0 failed because 15 unit tests imported `huggingface_hub`, `torch`,
`yaml` and `demucs`: the shared local `.venv` had them (every `uv sync --package <tool>`
installs into it), CI's root environment did not, so the tests passed locally and failed in the
release build. Nothing checked that the root unit tests run without model libraries, that tools
import heavy libraries only lazily, or that tools stay independent of each other.

## Proposal
Write the rules down in one place ([docs/architecture.md](../architecture.md), linked from
`CLAUDE.md`) and enforce the ones a machine can check, inside the two commands everyone already
runs. Where existing code breaks a new limit, a baseline records it and may only shrink
(chosen by the user over refactoring everything at once).

- `uv run pytest`
  - the root `conftest.py` blocks every installed library outside the root environment (the
    `dev` group and what it pulls in), so a unit test that needs a model library fails locally
    just as in CI unless it skips itself (`pytest.importorskip`);
  - `tests/test_architecture.py`: tools use `ingest`, never each other; `ingest` uses no tool;
    the window's sources use the standard library only; modules import heavy libraries only
    inside functions;
  - `tests/test_code_size.py`: functions per module 20, parameters 5, lines per function 40,
    nesting 3, cyclomatic complexity 10 (ruff C901), attributes per class 7, fields per record
    12; 45 existing violations in `tests/code_size_baseline.json`;
  - `tests/test_types.py`: basedpyright (pyright's checker with its own Node) in `standard`
    mode, libraries outside the root environment stubbed as `Any` so the result is the same
    with or without them installed; 56 existing errors in `.basedpyright/baseline.json`.
- `uv run ruff check .` with more rule sets (`C4`, `SIM`, `PIE`, `PERF`, `ISC`, `N`, `TID`,
  `ICN`, `RUF`); their few findings were fixed in place.

## Details
- `LANGUAGE_TEST_MODEL_LIBS=1 uv run pytest` lifts the block to run the tests that need model
  libraries where they are installed; integration runs (`-m integration`) are not blocked.
- Left out on purpose, because they would change much of the existing code: ruff's `PTH`,
  `BLE`, `T20`, `PL`, `TRY`, `EM`, `FBT`, `S`.
- Rule for conflicts: when a rule contradicts sound domain modeling, ask the user and propose
  adapting the rule (`CLAUDE.md`, `docs/architecture.md`).
- Checked: with the blocker, the tests from before the 0.4.1 fix fail locally (14 failures)
  instead of passing; the type check reports the same 56 errors in the full `.venv` and in an
  environment with only the `dev` group.
- Commits: a6a05c0 (task), 85d9c06 (blocker), 7206ad8 (import rules), 8efc144 (ruff),
  44ac0ee (size limits), a5f11b2 (types), d800d63 (docs).

## Done when
- The rules are in `docs/architecture.md` and `CLAUDE.md` points to them.
- `uv run pytest` fails when a unit test or a module-level import needs a model library, a tool
  imports another tool, new code breaks a size limit, or a new type error appears;
  `uv run ruff check .` runs the stricter rule sets; both are green.
