"""Architecture rules (docs/architecture.md), checked by reading the sources.

- Dependency direction: every tool uses `ingest`, never another tool; `ingest` uses no tool;
  the window (`gui`) uses only the standard library and runs the tools as subprocesses.
- Heavy libraries are imported lazily: at module level a source or test file imports only the
  standard library, our own packages and the root environment's libraries (the `dev` group:
  numpy, scipy, ...), so every module loads in the root environment and the tools start fast.
"""

import ast
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).parent.parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
MEMBERS = {Path(m).name: ROOT / m for m in PROJECT["tool"]["uv"]["workspace"]["members"]}
STDLIB = set(sys.stdlib_module_names)

# What each member may import of ours, besides itself.
ALLOWED_FIRST_PARTY = {name: {"ingest"} for name in MEMBERS}
ALLOWED_FIRST_PARTY["ingest"] = set()
ALLOWED_FIRST_PARTY["gui"] = set()

# Deliberate exceptions, each with its reason: (file relative to ROOT, imported module).
EXCEPTIONS = {
    # Checks the window's language list against translate's; the window itself only runs translate.
    ("tools/gui/tests/test_pipeline.py", "translate.languages"),
}


def dev_group_modules() -> set[str]:
    """Top-level modules of the root environment's direct libraries (and pytest, packaging)."""
    names = {canonicalize_name(Requirement(r).name) for r in PROJECT["dependency-groups"]["dev"]}
    names |= {"pytest", "packaging"}
    return {
        module
        for module, dists in metadata.packages_distributions().items()
        if any(canonicalize_name(d) in names for d in dists)
    }


ROOT_LIBRARIES = dev_group_modules()


def python_files(members):
    for name, member in members.items():
        for path in sorted(member.rglob("*.py")):
            if ".venv" not in path.parts:
                yield name, path


def imports(tree: ast.Module):
    """(module, at module level) for every absolute import; TYPE_CHECKING blocks are skipped."""

    def visit(node, top_level):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and "TYPE_CHECKING" in ast.unparse(child.test):
                continue
            if isinstance(child, ast.Import):
                for alias in child.names:
                    yield alias.name, top_level
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                yield child.module, top_level
            else:
                inner = top_level and not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
                yield from visit(child, inner)

    yield from visit(tree, True)


def violations(members=MEMBERS, root=ROOT):
    found = []
    for name, path in python_files(members):
        rel = path.relative_to(root).as_posix()
        kind = path.relative_to(members[name]).parts[0]  # src, tests, or build scripts (packaging)
        local = {p.stem for p in path.parent.glob("*.py")} if kind == "tests" else set()
        for module, top_level in imports(ast.parse(path.read_text(encoding="utf-8"), rel)):
            top = module.partition(".")[0]
            if (rel, module) in EXCEPTIONS or top in STDLIB or top == name or top in local:
                continue
            if top in MEMBERS:
                if top not in ALLOWED_FIRST_PARTY[name]:
                    found.append(f"{rel}: imports {module}, but {name} must not depend on {top}")
            elif name == "gui" and kind == "src":
                found.append(f"{rel}: imports {module}; the window uses the standard library only")
            elif kind in ("src", "tests") and top_level and top not in ROOT_LIBRARIES:
                found.append(f"{rel}: imports {module} at module level; import it inside the function that needs it")
    return found


def test_the_checks_see_the_sources():
    files = [path for _, path in python_files(MEMBERS)]
    assert len(files) > 50
    assert {"numpy", "scipy", "py3langid"} <= ROOT_LIBRARIES


def test_import_rules():
    found = violations()
    assert not found, "architecture rules broken (see docs/architecture.md):\n" + "\n".join(found)


@pytest.mark.parametrize(
    ("member", "source", "expected"),
    [
        ("vocals", "from transcribe import cli", "vocals must not depend on transcribe"),
        ("ingest", "import translate.core", "ingest must not depend on translate"),
        ("accent", "import torch", "at module level"),
        ("gui", "def f():\n    import numpy", "standard library only"),
    ],
)
def test_import_rules_catch_violations(member, source, expected, tmp_path):
    package = tmp_path / member / "src" / member
    package.mkdir(parents=True)
    (package / "bad.py").write_text(source)
    found = violations({member: tmp_path / member}, tmp_path)
    assert len(found) == 1 and expected in found[0]


def test_lazy_imports_are_allowed(tmp_path):
    package = tmp_path / "accent" / "src" / "accent"
    package.mkdir(parents=True)
    (package / "ok.py").write_text("import numpy\nimport ingest\n\ndef load():\n    import torch\n")
    assert not violations({"accent": tmp_path / "accent"}, tmp_path)
