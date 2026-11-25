"""Size and complexity limits (docs/architecture.md), checked by reading the sources.

Code written before the limits sits in code_size_baseline.json with the value it had then.
That list may only shrink: a listed function or class must not get worse, and once it meets
the limit its entry has to go (the test says which). Everything else must meet the limits.
"""

import ast
import json
import re
import subprocess
from pathlib import Path

from ruff.__main__ import find_ruff_bin

ROOT = Path(__file__).parent.parent
BASELINE = Path(__file__).with_name("code_size_baseline.json")
SOURCES = ("packages", "tools")

LIMITS = {
    "functions per module": 20,  # source modules only; test modules are exempt
    "parameters": 5,  # self/cls, *args and **kwargs not counted; keyword-only ones are
    "lines per function": 40,
    "nesting": 3,  # if/for/while/with/try/match blocks inside a function; elif is not deeper
    "complexity": 10,  # cyclomatic, as ruff measures it (C901)
    "attributes per class": 7,
    "fields per record": 12,  # dataclasses, NamedTuples and TypedDicts without other logic
}

FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
BLOCKS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.TryStar, ast.Match)


def nesting(node: ast.AST, depth: int = 0) -> int:
    deepest = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (*FUNCTIONS, ast.ClassDef, ast.Lambda)):
            continue
        inner = depth + 1 if isinstance(child, BLOCKS) else depth
        if isinstance(node, ast.If) and node.orelse == [child] and isinstance(child, ast.If):
            inner = depth  # elif
        deepest = max(deepest, nesting(child, inner))
    return deepest


def complexities(root: Path) -> dict[tuple[str, int], int]:
    """Cyclomatic complexity by (file, line of the def), from ruff's C901."""
    out = subprocess.run(
        [find_ruff_bin(), "check", "--isolated", "--select", "C901", "--config", "lint.mccabe.max-complexity=0",
         "--output-format", "json", "--exit-zero", *SOURCES],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    found = {}
    for item in json.loads(out):
        value = int(re.search(r"\((\d+) > 0\)", item["message"]).group(1))
        found[(Path(item["filename"]).relative_to(root.resolve()).as_posix(), item["location"]["row"])] = value
    return found


def parameters(func: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = func.args.posonlyargs + func.args.args + func.args.kwonlyargs
    return len([a for a in args if a.arg not in ("self", "cls")])


def is_record(cls: ast.ClassDef) -> bool:
    names = {ast.unparse(d).split("(")[0].rsplit(".", 1)[-1] for d in cls.decorator_list}
    bases = {ast.unparse(b).rsplit(".", 1)[-1] for b in cls.bases}
    return "dataclass" in names or bool(bases & {"NamedTuple", "TypedDict"})


def is_enum(cls: ast.ClassDef) -> bool:
    return any(ast.unparse(b).rsplit(".", 1)[-1].endswith(("Enum", "Flag")) for b in cls.bases)


def attributes(cls: ast.ClassDef) -> int:
    fields = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    stored = {
        n.attr
        for n in ast.walk(cls)
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
        and isinstance(n.value, ast.Name) and n.value.id == "self"
    }
    return len(fields | stored)


def measure(tree: ast.Module, is_test: bool, complexity=lambda line: 1):
    """(name, limit, value) for every function and class, nested ones included."""
    functions = [n for n in ast.walk(tree) if isinstance(n, FUNCTIONS)]
    if not is_test:
        yield "<module>", "functions per module", len(functions)

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, FUNCTIONS):
                name = prefix + child.name
                yield name, "parameters", parameters(child)
                yield name, "lines per function", child.end_lineno - child.lineno + 1
                yield name, "nesting", nesting(child)
                yield name, "complexity", complexity(child.lineno)
                yield from walk(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                name = prefix + child.name
                if not is_enum(child):
                    limit = "fields per record" if is_record(child) else "attributes per class"
                    yield name, limit, attributes(child)
                yield from walk(child, name + ".")
            else:
                yield from walk(child, prefix)

    yield from walk(tree, "")


def current(root: Path = ROOT) -> dict[str, int]:
    """Every value over its limit, as {"path::name: limit": value}."""
    over = {}
    measured = complexities(root)
    for top in SOURCES:
        for path in sorted((root / top).rglob("*.py")):
            if ".venv" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), rel)
            def complexity(line, rel=rel):
                return measured.get((rel, line), 1)

            for name, limit, value in measure(tree, "tests" in path.parts, complexity):
                if value > LIMITS[limit]:
                    over[f"{rel}::{name}: {limit}"] = value
    return over


def test_code_meets_the_limits():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    found = current()
    new = [f"{key} is {value} (limit {LIMITS[key.rsplit(': ', 1)[1]]})" for key, value in found.items()
           if key not in baseline]
    worse = [f"{key} grew from {baseline[key]} to {value}" for key, value in found.items()
             if key in baseline and value > baseline[key]]
    assert not new + worse, "limits broken (see docs/architecture.md):\n" + "\n".join(new + worse)


def test_the_baseline_only_shrinks():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    found = current()
    fixed = [key for key in baseline if key not in found]
    better = [f'"{key}": {found[key]}' for key in baseline if key in found and found[key] < baseline[key]]
    assert not fixed, "now within the limits; remove from tests/code_size_baseline.json:\n" + "\n".join(fixed)
    assert not better, "improved; lower in tests/code_size_baseline.json:\n" + "\n".join(better)


def test_measures():
    source = '''
def f(a, b, *args, c, d=1, **kw):
    if a:
        for x in b:
            if x and c or d:
                pass
    elif b:
        with open(a) as fh:
            while fh:
                try:
                    pass
                except OSError:
                    pass
    return [y for y in b if y]

class Box:
    def __init__(self):
        self.a = self.b = 1

@dataclass
class Point:
    x: int
    y: int
'''
    values = {(name, limit): value for name, limit, value in measure(ast.parse(source), False)}
    assert values[("f", "parameters")] == 4
    assert values[("f", "nesting")] == 4  # elif, with, while, try
    assert values[("Box", "attributes per class")] == 2
    assert values[("Point", "fields per record")] == 2
    assert values[("<module>", "functions per module")] == 2


def test_complexity_comes_from_ruff():
    measured = complexities(ROOT)
    tree = ast.parse((ROOT / "packages/ingest/src/ingest/timeparse.py").read_text(encoding="utf-8"))
    lines = [n.lineno for n in ast.walk(tree) if isinstance(n, FUNCTIONS)]
    assert all(("packages/ingest/src/ingest/timeparse.py", line) in measured for line in lines)
