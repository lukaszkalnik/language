"""First-run setup for the packaged app: installs the tools and their helpers with uv.

The app ships the workspace sources and a `uv` binary. On first start (and after an update)
it copies the sources to the app's data folder and runs `uv tool install` for each tool into
that folder, so nothing touches the user's own Python. uv fetches Python itself. ffmpeg and
Deno (yt-dlp's JavaScript runtime for YouTube) come from PyPI wheels, so no other download
site is involved. Models are not fetched here: each tool downloads its model on first use.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from . import __version__

TOOLS = ["transcribe", "vocals", "accent", "translate"]
PYTHON = "3.12"
FFMPEG_WHEEL = "imageio-ffmpeg==0.8.1"
DENO = "deno==2.9.8"
EXE = ".exe" if sys.platform == "win32" else ""

# One line per paragraph or list item: the window wraps them and indents wrapped list items.
INTRO = """\
Welcome! Before the first use, Language installs its tools on this computer.

Now, once (about 2–3 GB):
• Python and PyTorch, which run the speech and music models
• the four tools: transcription, song vocals (separates the singing from the music), accent recognition and translation
• ffmpeg, which reads audio and video files, and Deno, which YouTube downloads need

Later, the first time you use a feature, its model is downloaded:
• transcription: 3.1 GB
• song vocals: 0.3 GB
• accent recognition: 0.1 GB
• offline translation: 2.5 GB (not needed if you use Ollama)

Everything then runs on this computer: nothing is uploaded, and no accounts or tokens are needed.

Setup needs an internet connection and takes 10–30 minutes. Everything goes into:
{folder}

Press “Set up” to start.
"""


def data_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "Language"


def bin_dir() -> Path:
    return data_dir() / "bin"


def frozen() -> bool:
    """Running as the packaged app (PyInstaller), not from source."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def tool_env(base: dict | None = None) -> dict:
    """Environment for every subprocess: our bin folder first, so tools find ffmpeg and deno.

    Apps started from the Finder get a minimal PATH, so this matters there even when the user
    has Homebrew tools.
    """
    env = dict(os.environ if base is None else base)
    env["PATH"] = os.pathsep.join([str(bin_dir()), env.get("PATH", "")])
    env["UV_TOOL_DIR"] = str(data_dir() / "tools")
    env["UV_TOOL_BIN_DIR"] = str(bin_dir())
    env["UV_PYTHON_PREFERENCE"] = "only-managed"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def marker() -> Path:
    return data_dir() / "installed.json"


def needed() -> bool:
    """Only the packaged app installs anything; from source the tools are on PATH or in the workspace."""
    if not frozen():
        return False
    try:
        return json.loads(marker().read_text()).get("version") != __version__
    except (OSError, ValueError):
        return True


def uv_binary() -> str:
    bundled = bundle_dir() / "uv" / f"uv{EXE}"
    if bundled.is_file():
        return str(bundled)
    found = shutil.which("uv")
    if not found:
        raise RuntimeError("uv was not found; install it from https://docs.astral.sh/uv/")
    return found


def commands(uv: str, src: Path) -> list[tuple[str, list[str]]]:
    """(what, command) for every install step, in order."""
    steps = [
        (f"Installing {tool}", [
            uv, "tool", "install", "--python", PYTHON, "--reinstall-package", tool,
            "--reinstall-package", "ingest", str(src / "tools" / tool)])
        for tool in TOOLS
    ]
    steps.append(("Installing Deno (for YouTube)", [uv, "tool", "install", "--python", PYTHON, DENO]))
    steps.append(("Fetching ffmpeg", [
        uv, "run", "--no-project", "--python", PYTHON, "--with", FFMPEG_WHEEL, "python", "-c",
        "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"]))
    return steps


def install_ffmpeg(path: str) -> Path:
    target = bin_dir() / f"ffmpeg{EXE}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


def run_setup(on_status: Callable[[str], None],
              on_step: Callable[[int, int, str], None] = lambda i, n, title: None) -> None:
    """Install everything. Safe to run again after a failure or an update."""
    uv = uv_binary()
    src = data_dir() / "src"
    on_status("Copying the tools...")
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(bundle_dir() / "src", src)
    env = tool_env()
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    steps = commands(uv, src)
    for index, (what, cmd) in enumerate(steps, 1):
        on_step(index, len(steps), what)
        on_status(f"{what}...")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=env, creationflags=flags)
        lines = []
        for line in proc.stdout:
            lines.append(line.strip())
            if line.strip():
                on_status(line.strip())
        if proc.wait() != 0:
            tail = next((ln for ln in reversed(lines) if ln), "no output")
            raise RuntimeError(f"{what} failed: {tail}")
        if what == "Fetching ffmpeg":
            install_ffmpeg([ln for ln in lines if ln][-1])
    marker().write_text(json.dumps({"version": __version__}))
    on_status("Setup finished.")
