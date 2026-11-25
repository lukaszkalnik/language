# Architecture and code-quality rules

How the code is organised and what every change must keep true. Rules marked **(checked)** are
enforced by `uv run pytest` or `uv run ruff check .`, so a broken rule fails the build; the
others are for review. Background: [task 0008](tasks/0008-architecture-rules.md).

**The domain model comes first.** When a rule here gets in the way of modelling the domain
soundly (a record that really has 14 fields, a function whose steps belong together), do not
bend the model to fit the rule: ask, and propose how to adapt the rule.

## Structure: packages by feature

```
packages/ingest      shared input handling: files, YouTube, stdin WAV, time ranges, cache, CLI helpers
tools/transcribe     language detection and transcription (Whisper)
tools/vocals         vocal separation (Demucs)
tools/accent         English accent recognition (SpeechBrain)
tools/translate      translation (Ollama LLM, NLLB, Breton via French)
tools/gui            the desktop window (Tk); runs the tools as subprocesses
```

- Each tool is one feature: its own package, its own command, its own dependencies in its
  `pyproject.toml`. Inside a tool, a module holds one part of that feature (`vad`, `diarize`,
  `whisper`, `hallucination`), not one technical layer (no `utils.py`, `helpers.py`, `models.py`
  catch-alls across features).
- What two tools need goes into `ingest`, and only when the second tool needs it.

## Dependency direction

- **(checked)** Tools depend on `ingest`, never on each other. `ingest` depends on no tool.
  (`tests/test_architecture.py`)
- **(checked)** The window's sources use the standard library only: they run in the app bundle
  and talk to the tools only by running their commands. Its tests may read a tool's data where
  that keeps the two in step; each such import is listed with its reason in
  `tests/test_architecture.py`.
- Heavy model dependencies (torch, transformers, Demucs, Whisper, pyannote, ...) belong to the
  tool that needs them, never to `ingest` or the root.

## Imports

- **(checked)** At module level a source or test file imports only the standard library, our
  own packages and the root environment's libraries (the `dev` group: numpy, scipy, py3langid).
  Everything else is imported inside the function that needs it, so every module loads in the
  root environment and a tool's `--help` starts fast. (`tests/test_architecture.py`)
- Relative imports within a package (`from . import vad`); no imports from a parent package
  with `..` **(checked: ruff TID)**.

## Size and complexity

**(checked)** by `tests/test_code_size.py`:

| Limit | Value |
|---|---|
| functions per module (sources only) | 20 |
| parameters per function (not counting `self`, `*args`, `**kwargs`) | 5 |
| lines per function | 40 |
| nesting of `if`/`for`/`while`/`with`/`try`/`match` in a function (`elif` is not deeper) | 3 |
| cyclomatic complexity (ruff's C901) | 10 |
| attributes per class | 7 |
| fields per record (dataclass, NamedTuple, TypedDict) | 12 |

Code from before these limits is listed in `tests/code_size_baseline.json` with the value it
had then. An entry may not get worse, and when it meets the limit (or improves) the test asks
for the entry to be removed (or lowered): the list only shrinks. Do not add entries; split the
code instead, or, if the domain says otherwise, ask (see above).

How to stay within them: give a group of parameters that travel together a name (a frozen
dataclass such as `Options`); return early instead of nesting; turn the body of a deep loop into
a function; split a module along the feature's parts.

## Types and domain modeling

- **(checked)** Every source and test passes basedpyright in `standard` mode
  (`tests/test_types.py`). Errors from before the check are in `.basedpyright/baseline.json`;
  new ones fail. After fixing old ones run `uv run python tests/test_types.py --writebaseline`,
  which removes them from the baseline. basedpyright sees only the root environment's
  libraries there (the tools' libraries are typed from typeshed's stubs or not at all), so the
  result does not depend on what is installed.
- Annotate every function's parameters and return type.
- Make illegal states unrepresentable:
  - a fixed set of choices is an `Enum` (or a `Literal[...]` for plain strings at a boundary
    such as argparse), never a free `str` compared against constants all over the code;
  - a value that is one of several shapes is a union of frozen dataclasses, one per case
    (Python's equivalent of a sealed type), handled with `match` and closed with
    `typing.assert_never`, so the checker reports a case nobody handles:

    ```python
    @dataclass(frozen=True)
    class FileInput:
        path: Path

    @dataclass(frozen=True)
    class YouTubeInput:
        video_id: str

    Input = FileInput | YouTubeInput

    def describe(source: Input) -> str:
        match source:
            case FileInput(path):
                return path.name
            case YouTubeInput(video_id):
                return f"youtube:{video_id}"
            case _:
                assert_never(source)
    ```
  - instead of several optional fields of which only some combinations make sense, one field
    holding one of those cases;
  - values are `@dataclass(frozen=True)` unless they must change.

## Errors

- A problem the user can fix raises a subclass of `ingest.errors.UserError` (`AudioError`,
  `YouTubeError`, `ModelError`, or a tool's own) with one readable sentence saying what to do.
- **(convention)** Every command runs its main through `ingest.cli.run`, which turns a
  `UserError` into one `<tool>: error: ...` line on stderr and exit code 1 (2 for bad arguments,
  130 when interrupted) instead of a traceback; `LANGUAGE_DEBUG=1` shows the traceback.
- Catch the narrowest exception that fits. A broad `except Exception` is only for a deliberate
  fallback or the edge of the program (a model load that may fail in many ways, the window
  showing an error) and carries a `# noqa: BLE001 - <why>` comment.
- Never swallow an error silently; turn it into a `UserError`, a warning on stderr, or re-raise.

## Command-line tools

- Data on stdout, everything for humans on stderr (status, progress, warnings), so commands
  can be piped. `-` reads a WAV stream from stdin.
- Progress for the window as `@progress <fraction> <label>` lines on stderr, only with
  `LANGUAGE_PROGRESS=1` (`ingest.cli.report_progress`). Under the same variable, output that is
  ready before the end goes out as `@partial "<JSON string>"` lines on stderr
  (`ingest.cli.report_partial`): a piece to show at once, which the final stdout replaces. Stdout
  itself is written only when done, so it stays the one finished result.
- Common input options come from `ingest.cli.add_input_args`; `--info` prints the input's title
  and duration as one JSON line.

## Naming

- **(checked: ruff N)** PEP 8: `snake_case` functions, variables and modules, `CapWords`
  classes, `UPPER_CASE` constants. Exceptions may go without an `Error` suffix when the name
  says what happened (`Cancelled`, `StepFailed`).
- Names say what something is in the domain (`voice_regions`, `lost_stretches`), not its type
  (`data`, `lst`, `result2`). Functions are verbs, values nouns, booleans read as a statement
  (`is_cached`, `capped`).
- A leading `_` marks what is private to its module.

## Testing

- `uv run pytest` at the root runs all unit tests: no network, no models, a few seconds.
- **(checked)** Unit tests run with the root environment only. The root `conftest.py` blocks
  every installed library outside it, because the shared `.venv` gathers every tool's libraries
  while CI's root environment does not have them. A test that needs one skips itself with
  `pytest.importorskip("torch")`. `LANGUAGE_TEST_MODEL_LIBS=1 uv run pytest` lifts the block to
  run those tests where the libraries are installed.
- Tests that need the network or a model are marked `@pytest.mark.integration` and run per tool:
  `uv run --package <tool> pytest -m integration tools/<tool>` (one model-heavy run at a time).
- A test checks behaviour through the public function, with small inputs built in the test
  (`ingest.testing` has helpers for WAV files and for catching network connections).
- A fixed bug gets a test that fails without the fix.

## Lint

**(checked)** `uv run ruff check .` with the rule sets `E`, `F`, `W`, `I`, `B`, `UP`, `C4`, `SIM`,
`PIE`, `PERF`, `ISC`, `N`, `TID`, `ICN`, `RUF`, line length 110 (see `pyproject.toml` for the few
ignored rules and why). Not adopted, as they would change much of the existing code: `PTH`,
`BLE`, `T20`, `PL`, `TRY`, `EM`, `FBT`, `S`.
