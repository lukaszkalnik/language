"""Build the double-clickable app: Language.app in a .dmg (macOS) or Language.exe (Windows).

uv run --no-project --with "pyinstaller>=6.11,<7" --with pillow python tools/gui/packaging/build.py

Bundles the window (standard library only), the `uv` found on PATH, and the workspace sources
(tracked files), which the app installs on first start. Output goes to dist/.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GUI = ROOT / "tools" / "gui"
BUILD = ROOT / "build" / "gui"
DIST = ROOT / "dist"
SOURCES = ("pyproject.toml", "uv.lock", ".python-version", "packages/", "tools/")


def version() -> str:
    ns: dict = {}
    exec((GUI / "src" / "gui" / "__init__.py").read_text(), ns)
    return ns["__version__"]


def stage() -> Path:
    stage = BUILD / "stage"
    shutil.rmtree(stage, ignore_errors=True)
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for name in files.splitlines():
        if name.startswith(SOURCES) and not name.startswith("tools/gui/"):
            dest = stage / "src" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
    uv = shutil.which("uv")
    if not uv:
        sys.exit("error: uv must be on PATH to be bundled")
    (stage / "uv").mkdir(parents=True)
    shutil.copy2(uv, stage / "uv" / Path(uv).name)
    return stage


def main() -> int:
    ver = version()
    st = stage()
    icon = GUI / "src" / "gui" / "icon.png"
    windows = sys.platform == "win32"
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed",
        "--name", "Language", "--icon", str(icon),
        "--distpath", str(BUILD / "dist"), "--workpath", str(BUILD / "work"), "--specpath", str(BUILD),
        "--paths", str(GUI / "src"),
        "--add-data", f"{st / 'src'}{os.pathsep}src",
        "--add-data", f"{st / 'uv'}{os.pathsep}uv",
        "--add-data", f"{icon}{os.pathsep}.",
        "--onefile" if windows else "--onedir",
    ]
    if sys.platform == "darwin":
        cmd += ["--osx-bundle-identifier", "io.github.lukaszkalnik.language"]
    subprocess.run([*cmd, str(GUI / "packaging" / "launch.py")], check=True)

    DIST.mkdir(exist_ok=True)
    if windows:
        out = DIST / f"Language-{ver}-windows-x64.exe"
        shutil.copy2(BUILD / "dist" / "Language.exe", out)
    elif sys.platform == "darwin":
        app = BUILD / "dist" / "Language.app"
        plist = app / "Contents" / "Info.plist"
        subprocess.run(["plutil", "-replace", "CFBundleShortVersionString", "-string", ver, str(plist)], check=True)
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True)
        dmg_root = BUILD / "dmg"
        shutil.rmtree(dmg_root, ignore_errors=True)
        dmg_root.mkdir()
        subprocess.run(["ditto", str(app), str(dmg_root / "Language.app")], check=True)
        (dmg_root / "Applications").symlink_to("/Applications")
        out = DIST / f"Language-{ver}-macos-arm64.dmg"
        out.unlink(missing_ok=True)
        subprocess.run(["hdiutil", "create", "-volname", "Language", "-srcfolder", str(dmg_root),
                        "-format", "UDZO", "-ov", str(out)], check=True)
    else:
        sys.exit("error: builds are made on macOS and Windows")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
