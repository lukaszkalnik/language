"""Type check (docs/architecture.md): basedpyright over all sources and tests.

The shared `.venv` may or may not hold the tools' libraries (torch, yt_dlp, ...), and CI's root
environment never does. So every installed library outside the root environment gets an empty
stub that types it as Any (in build/type-stubs, the stubPath in pyproject.toml), and the
result is the same everywhere. Errors from before the check are listed in
.basedpyright/baseline.json; new ones fail the test. After fixing old ones, drop them from the
baseline with: uv run python tests/test_types.py --writebaseline
"""

import ast
import shutil
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.utils import canonicalize_name

ROOT = Path(__file__).parent.parent
# Used by the standard library's own stubs: replacing them with Any would break those.
KEEP = {"typing_extensions", "mypy_extensions", "_typeshed"}
STUB = "from typing import Any\n\ndef __getattr__(name: str) -> Any: ...\n"


def libraries_to_stub(exe: str) -> set[str]:
    """Every library outside the root environment, installed or known to typeshed.

    Installed ones are what conftest blocks; typeshed's are the third-party stubs basedpyright
    brings (yt_dlp, yaml, ...), used when the library is not installed. Stubbing both the same
    way makes the result independent of what the environment holds.
    """
    sys.path.insert(0, str(ROOT))
    try:
        from conftest import blocked_modules, root_environment
    finally:
        sys.path.remove(str(ROOT))
    stubs_dirs = list(Path(sys.prefix).glob("lib*/python*/site-packages/basedpyright/dist/typeshed-fallback/stubs"))
    stubs_dirs += list(Path(sys.prefix).glob("Lib/site-packages/basedpyright/dist/typeshed-fallback/stubs"))
    assert stubs_dirs, f"typeshed stubs of basedpyright not found under {sys.prefix}"
    known = {
        entry.name.removesuffix(".pyi")
        for dist in stubs_dirs[0].iterdir()
        for entry in dist.iterdir()
        if entry.name not in ("METADATA.toml", "@tests")
    }
    allowed = root_environment()
    root_modules = {
        module
        for module, dists in metadata.packages_distributions().items()
        if any(canonicalize_name(d) in allowed for d in dists)
    }
    return (blocked_modules() | known) - root_modules - KEEP


def write_stubs(exe: str) -> None:
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["basedpyright"]
    stubs = ROOT / settings["stubPath"]
    shutil.rmtree(stubs, ignore_errors=True)
    libraries = libraries_to_stub(exe)
    # A stub package missing a submodule lets basedpyright find the real one, so every
    # submodule we import (mlx.core, yt_dlp.utils, ...) gets its own stub too.
    for module in libraries | {m for m in imported_modules() if m.partition(".")[0] in libraries}:
        package = stubs.joinpath(*module.split("."))
        package.mkdir(parents=True, exist_ok=True)  # exist_ok: macOS ignores case
        (package / "__init__.pyi").write_text(STUB)


def imported_modules() -> set[str]:
    """Every absolute module name our sources and tests import, with its parent packages."""
    found = set()
    for path in [*ROOT.glob("packages/*/**/*.py"), *ROOT.glob("tools/*/**/*.py")]:
        if ".venv" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                parts = name.split(".")
                found |= {".".join(parts[:i]) for i in range(1, len(parts) + 1)}
    return found


def basedpyright(*args: str) -> subprocess.CompletedProcess:
    exe = shutil.which("basedpyright", path=str(Path(sys.executable).parent))
    assert exe, "basedpyright is not installed: uv sync"
    write_stubs(exe)
    return subprocess.run(
        [exe, "-p", str(ROOT), "--pythonpath", sys.executable, *args],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )


def test_types():
    result = basedpyright()
    assert result.returncode == 0, (
        "type errors (see docs/architecture.md); the baseline lists only older ones:\n" + result.stdout[-6000:]
    )


if __name__ == "__main__":
    done = basedpyright(*sys.argv[1:])
    print(done.stdout + done.stderr)
    sys.exit(done.returncode)
