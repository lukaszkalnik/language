"""Type check (docs/architecture.md): basedpyright over all sources and tests.

CI's root environment has only the `dev` group, while the shared `.venv` here may also hold the
tools' libraries (torch, yt_dlp, ...), whose types would change the result. So when it does,
basedpyright gets an interpreter (build/type-env) that sees only the root environment's
libraries, and the result is the same everywhere. Errors from before the check are listed in
.basedpyright/baseline.json; new ones fail the test. After fixing old ones, drop them from the
baseline with: uv run python tests/test_types.py --writebaseline
"""

import os
import shutil
import subprocess
import sys
import sysconfig
import venv
from importlib import metadata
from pathlib import Path

from packaging.utils import canonicalize_name

ROOT = Path(__file__).parent.parent
TYPE_ENV = ROOT / "build" / "type-env"


def extra_libraries_installed() -> tuple[bool, set[str]]:
    """Whether this environment holds more than the root one, and the root one's distributions."""
    sys.path.insert(0, str(ROOT))
    try:
        from conftest import blocked_modules, root_environment
    finally:
        sys.path.remove(str(ROOT))
    return bool(blocked_modules()), root_environment()


def interpreter() -> str:
    """This Python if it holds only the root environment, else build/type-env linking just that."""
    extra, allowed = extra_libraries_installed()
    if not extra:
        return sys.executable
    if not (TYPE_ENV / "pyvenv.cfg").exists():
        venv.create(TYPE_ENV, with_pip=False, symlinks=os.name != "nt")
    site = Path(sysconfig.get_path("purelib", vars={"base": str(TYPE_ENV), "platbase": str(TYPE_ENV)}))
    for entry in site.iterdir():
        if entry.is_symlink():
            entry.unlink()
    here = Path(sysconfig.get_path("purelib"))
    for dist in metadata.distributions():
        if canonicalize_name(dist.metadata["Name"]) not in allowed:
            continue
        for top in {Path(str(f)).parts[0] for f in dist.files or [] if not str(f).startswith("..")}:
            if (here / top).exists() and not (site / top).exists():
                (site / top).symlink_to(here / top, target_is_directory=(here / top).is_dir())
    return str(TYPE_ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


def basedpyright(*args: str) -> subprocess.CompletedProcess:
    exe = shutil.which("basedpyright", path=str(Path(sys.executable).parent))
    assert exe, "basedpyright is not installed: uv sync"
    return subprocess.run(
        [exe, "-p", str(ROOT), "--pythonpath", interpreter(), *args],
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
