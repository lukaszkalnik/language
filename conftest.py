"""Root test setup: unit tests stay off the network and run with the root environment only.

- No IPv6/IPv4 probe at tool start (see ingest.cli.choose_ipv4).
- Model libraries are blocked. The shared `.venv` gathers every tool's libraries (each
  `uv sync --package <tool>` installs into it), but CI's root environment has only the `dev`
  group. So a unit test that needs torch or huggingface_hub would pass here and fail in the
  release build. Blocking every installed library outside the root environment makes it fail
  here too; such a test must skip itself with `pytest.importorskip`.
  `LANGUAGE_TEST_MODEL_LIBS=1` lifts the block; integration runs (`-m integration`) are
  never blocked.
"""

import os
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

os.environ.setdefault("LANGUAGE_IPV4", "0")

ROOT = Path(__file__).parent


def root_environment() -> set[str]:
    """Distributions the root environment has: its default groups, pytest, and what they require."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    groups = project["tool"]["uv"]["default-groups"]
    todo = [Requirement(r).name for g in groups for r in project["dependency-groups"][g]] + ["pytest"]
    members = [Path(m).name for m in project["tool"]["uv"]["workspace"]["members"]]
    seen = {canonicalize_name(m) for m in members}
    while todo:
        name = canonicalize_name(todo.pop())
        if name in seen:
            continue
        seen.add(name)
        try:
            requires = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        for spec in requires:
            req = Requirement(spec)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                todo.append(req.name)
    return seen


def blocked_modules() -> set[str]:
    """Top-level modules of installed distributions that the root environment does not have."""
    allowed = root_environment()
    ours = {path.stem for path in ROOT.iterdir()}  # e.g. `tests`, which some libraries ship too
    return {
        module
        for module, dists in metadata.packages_distributions().items()
        if not module.startswith("_")
        and module not in ours
        and not any(canonicalize_name(d) in allowed for d in dists)
    }


class ModelLibraryBlocker:
    def __init__(self, blocked: set[str]):
        self.blocked = blocked

    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in self.blocked:
            raise ModuleNotFoundError(
                f"No module named {name!r} (not in the root environment: unit tests must not need "
                f"it; skip with pytest.importorskip, see docs/architecture.md)",
                name=name,
            )
        return None


def pytest_configure(config):
    if os.environ.get("LANGUAGE_TEST_MODEL_LIBS") == "1":
        return
    if (config.option.markexpr or "").strip() != "not integration":
        return
    blocked = blocked_modules() - set(sys.modules)
    sys.meta_path.insert(0, ModelLibraryBlocker(blocked))


def pytest_terminal_summary(terminalreporter):
    """Say how many tests skipped for want of the tools' libraries, and how to run them."""
    reasons = [
        str(report.longrepr[2]) if isinstance(report.longrepr, tuple) else str(report.longrepr)
        for report in terminalreporter.stats.get("skipped", [])
    ]
    count = sum("not in the root environment" in r or "is not installed" in r for r in reasons)
    if count:
        terminalreporter.write_line(
            f"{count} test(s) skipped: they need the tools' libraries (torch, Demucs, ...). CI runs them; "
            "here: uv sync --all-packages --inexact && LANGUAGE_TEST_MODEL_LIBS=1 uv run pytest",
            yellow=True,
        )
