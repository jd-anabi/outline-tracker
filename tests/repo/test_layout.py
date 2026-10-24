"""The table of paths (tests/support/layout_helpers.py) and the rules that keep the tests on it (docs/ROADMAP.md, W2).

A move of files changes the table and the import lines, and no test body. These tests guard that:

- the table is true: every path in it exists, and the dotted name it gives for a package file is the
  name under which Python finds that file;
- the repository is found by walking up to pyproject.toml, so the helper module can lie at any depth;
- no test takes a folder of the repository from its own file (`parents[1]` of its own path and its kin),
  because such a line breaks when the test moves;
- every relative link in a tracked .md file points to a file or folder that exists, so that a move of
  a document cannot leave a dead link;
- `provenance.PACKAGE_FOLDER` comes from the imported package, not from the file of core/provenance.py.

The scans are checked on small made-up files in `tmp_path`, so that a scan that never finds anything
would show up here. Paths in this file are repository paths; no units or coordinates.
"""

import ast
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from layout_helpers import PATHS, REPO, below, dead_links, module
from layout_helpers import path as table_path

import outline_tracker
from outline_tracker.core import provenance

TESTS = REPO / "tests"


# ---------------------------------------------------------------------------------------------
# The table


def test_every_entry_of_the_table_is_written_with_slashes_and_exists():
    assert PATHS  # the real table
    not_written_so = [role for role, text in PATHS.items()
                      if "\\" in text or text.startswith("/") or ".." in text.split("/")]
    assert not_written_so == []
    assert [role for role in PATHS if not table_path(role).exists()] == []


def test_the_dotted_name_of_a_package_path_is_the_name_python_finds_that_file_under():
    in_package = [role for role, text in PATHS.items() if text.startswith("outline_tracker/")]
    assert in_package  # the real table
    wrong = []
    for role in in_package:
        try:
            spec = importlib.util.find_spec(module(role))
        except ModuleNotFoundError:
            spec = None
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != table_path(role).resolve():
            wrong.append(f"{role}: {module(role)} is not found at {PATHS[role]}")
    assert wrong == []


def test_path_module_and_below_read_the_table(monkeypatch):
    # Made-up entries, so that the answers do not depend on where the real files lie.
    monkeypatch.setitem(PATHS, "made up module", "outline_tracker/part/sub/name.py")
    monkeypatch.setitem(PATHS, "made up package", "outline_tracker/part/__init__.py")
    monkeypatch.setitem(PATHS, "made up top", "outline_tracker/__init__.py")
    assert table_path("made up module") == REPO / "outline_tracker" / "part" / "sub" / "name.py"
    assert module("made up module") == "outline_tracker.part.sub.name"
    assert module("made up package") == "outline_tracker.part"  # an __init__.py names its package
    assert module("made up top") == "outline_tracker"
    package = REPO / "outline_tracker"
    assert below(package, "made up module", "made up package") == {"part/sub/name.py", "part/__init__.py"}
    assert below(package / "part", "made up module") == {"sub/name.py"}
    with pytest.raises(ValueError):
        below(REPO / "tests", "made up module")  # not below that folder
    with pytest.raises(KeyError):
        table_path("no such role")


@pytest.mark.parametrize("text", [
    "SPEC.md",
    "docs/outline_tracker/name.py",
    "tests/test_made_up.py",
    "outline_tracker/notes.txt",
    "outline_tracker",
], ids=["a document", "the package's name below another folder", "a test file", "a file that is no module",
        "the folder alone"])
def test_the_dotted_name_of_a_path_outside_the_package_is_refused(text, monkeypatch):
    monkeypatch.setitem(PATHS, "made up", text)
    with pytest.raises(ValueError, match="made up"):
        module("made up")


def test_the_real_table_gives_no_dotted_name_to_a_document():
    documents = [role for role, text in PATHS.items() if text.endswith(".md")]
    assert documents  # the developer page, the spec, ...
    for role in documents:
        with pytest.raises(ValueError):
            module(role)


# ---------------------------------------------------------------------------------------------
# The repository is found by walking up


def _load_copy_of_the_helpers(into: Path):
    """The text of tests/support/layout_helpers.py, copied to `into`, and loaded from there."""
    into.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(table_path("layout helpers"), into)
    spec = importlib.util.spec_from_file_location("layout_helpers_copy", into)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_the_repository_is_the_first_folder_above_the_helpers_that_holds_pyproject_toml(tmp_path):
    folders = list(table_path("layout helpers").resolve().parents)
    assert REPO in folders and (REPO / "pyproject.toml").is_file()
    assert [folder for folder in folders[:folders.index(REPO)] if (folder / "pyproject.toml").exists()] == []
    # In a made-up tree the file lies one, two and three folders below the repository: the answer is the
    # same each time, and an outer folder that also holds a pyproject.toml is not taken.
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "pyproject.toml").write_text("[project]\n")
    for depth, parts in enumerate([("tests",), ("tests", "support"), ("tests", "support", "deeper")], start=1):
        loaded = _load_copy_of_the_helpers(inner.joinpath(*parts, "layout_helpers.py"))
        assert loaded.REPO == inner.resolve(), depth


# ---------------------------------------------------------------------------------------------
# No test takes a folder of the repository from its own file


# The name that every Python file has for its own place, put together here: a search of tests/ for it
# then finds the table's line, the form that stays and the uses of a module's attribute, and no more.
# In the sources and the texts below, `<own>` stands for it.
OWN = "__fi" + "le__"


def _own_file_uses(folder: Path) -> list[str]:
    """`file:line` for every use of the bare name `<own>` in a .py file under `folder`, except the one form
    that stays: `Path(<own>).resolve().parent` or `Path(<own>).parent`, the folder of a file that lies beside
    the test. `parents[1]`, a second `.parent`, `os.path.dirname(<own>)` and every other use are found.
    So is a step up from a name that holds the form that stays: after `HERE = Path(<own>).parent`, the
    expressions `HERE.parent`, `HERE.parents[1]` and `HERE / ".."` are found, on the line of `HERE`.
    Not found: a step made by a call that takes the name (`os.path.join(HERE, "..")`, `Path(HERE, "..")`,
    `os.path.dirname(HERE)`), and a second name for the same folder (`ROOT = HERE`, then `ROOT.parent`).
    The attribute of an imported module (`json.<own>`) is not the bare name; a docstring, a string and a
    comment hold no name. The files are read as syntax trees."""
    found = []
    for path in sorted(folder.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        names = [node for node in ast.walk(tree) if isinstance(node, ast.Name)]
        forms = {node: _own_folder_form(node, parents) for node in names if node.id == OWN}
        held = _names_that_hold(forms.values(), parents)
        lines = sorted({node.lineno for node, form in forms.items() if form is None}
                       | {node.lineno for node in names
                          if node.id in held and isinstance(node.ctx, ast.Load) and _goes_up(node, parents)})
        found += [f"{path.relative_to(folder).as_posix()}:{line}" for line in lines]
    return found


def _own_folder_form(name: ast.Name, parents: dict) -> ast.expr | None:
    """The expression `Path(<own>).resolve().parent` or `Path(<own>).parent` that the name `<own>` is the
    start of, and nothing else: None if the name is in another form (a second `.parent`, or a `.parents`,
    is a folder higher up)."""
    call = parents.get(name)
    func = getattr(call, "func", None)
    named_path = (isinstance(func, ast.Name) and func.id == "Path"
                  or isinstance(func, ast.Attribute) and func.attr == "Path")
    if not (isinstance(call, ast.Call) and call.args == [name] and not call.keywords and named_path):
        return None
    steps, current = [], call
    while True:
        up = parents.get(current)
        if isinstance(up, ast.Attribute) and up.value is current:
            steps.append(up.attr)
            current = up
        elif isinstance(up, ast.Call) and up.func is current and not up.args and not up.keywords:
            current = up  # `.resolve()` has no arguments
        else:
            return current if steps in (["parent"], ["resolve", "parent"]) else None


def _names_that_hold(forms, parents: dict) -> set[str]:
    """The names that are assigned one of these expressions, as `HERE` in `HERE = Path(<own>).parent`."""
    held = set()
    for form in forms:
        assign = parents.get(form)
        if form is not None and isinstance(assign, (ast.Assign, ast.AnnAssign)) and assign.value is form:
            targets = assign.targets if isinstance(assign, ast.Assign) else [assign.target]
            held |= {target.id for target in targets if isinstance(target, ast.Name)}
    return held


def _goes_up(name: ast.Name, parents: dict) -> bool:
    """Whether the expression that starts at this use of a name that holds the own folder goes up from it:
    a `.parent`, a `.parents`, or a division by a path with a `..` step in it (`HERE / ".."`, also
    `(HERE / "data") / "../.."`). `.resolve()` and a division by a name do not."""
    current = name
    while True:
        up = parents.get(current)
        if isinstance(up, ast.Attribute) and up.value is current:
            if up.attr in ("parent", "parents"):
                return True
        elif isinstance(up, ast.BinOp) and isinstance(up.op, ast.Div) and up.left is current:
            step = up.right.value if isinstance(up.right, ast.Constant) and isinstance(up.right.value, str) else ""
            if ".." in re.split(r"[/\\]", step):
                return True
        elif not (isinstance(up, ast.Call) and up.func is current):
            return False
        current = up


def test_no_test_takes_a_folder_of_the_repository_from_its_own_file():
    # A line such as `REPO = Path(<own>).resolve().parents[1]` is true for one depth of one file. When the
    # file moves, the line is wrong and the test fails, or worse, passes on the wrong folder. The tests ask
    # tests/support/layout_helpers.py (`REPO`, `path`, `module`), which is the one file that may use its own place.
    assert table_path("test helpers") in set(TESTS.rglob("*.py"))  # the scan looks at the real folder
    helpers = table_path("layout helpers").relative_to(TESTS).as_posix()
    found = _own_file_uses(TESTS)
    assert [use for use in found if use.startswith(f"{helpers}:")] != []  # the scan sees the table's own line
    assert [use for use in found if not use.startswith(f"{helpers}:")] == []


def test_the_own_file_scan_finds_each_wrong_form_and_leaves_the_right_ones(tmp_path):
    imports = "from pathlib import Path\n\n"
    sources = {
        "parents.py": imports + "REPO = Path(<own>).resolve().parents[1]\n",
        "parent_twice.py": imports + "REPO = Path(<own>).parent.parent\n",
        "resolved_parent_twice.py": "import pathlib\n\nREPO = pathlib.Path(<own>).resolve().parent.parent\n",
        "dirname.py": "import os\n\nREPO = os.path.dirname(os.path.dirname(<own>))\n",
        "alone.py": "import os\nHERE = os.path.abspath(<own>)\nprint(open(<own>).read())\n",
        "in_folder/deeper/one_of_two.py": imports + "OK = Path(<own>).parent\nBAD = Path(<own>).parents[2]\n",
        # a name that holds the form that stays, climbed in a second step
        "held_parent.py": imports + "HERE = Path(<own>).parent\nREPO = HERE.parent\n",
        "held_parents.py": (imports + "HERE = Path(<own>).resolve().parent\nREPO = HERE.parents[1]\n"
                            "DATA = (HERE / 'data').parent\n"),
        "held_up_step.py": (imports + "HERE = Path(<own>).parent\nREPO = (HERE / '..').resolve()\n"
                            "TOP = HERE / 'data' / '../..'\n"),
        # the form that stays
        "own_folder.py": imports + "SLOW = Path(<own>).resolve().parent\n",
        "own_folder_plain.py": imports + "HERE = Path(<own>).parent\nDATA = HERE / 'data'\n",
        "held_not_climbed.py": (imports + "HERE = Path(<own>).resolve().parent\nNAME = HERE.name\n"
                                "SAME = HERE.resolve()\nFILES = HERE / 'a..b' / '..x'\n"),
        "other_name.py": imports + "ELSE = Path('data')\nUP = ELSE.parent / '..'\n",  # holds nothing of the kind
        # not the bare name
        "of_a_module.py": ("import json\nfrom pathlib import Path\n\n"
                           "lines = Path(json.<own>).read_text(encoding='utf-8').splitlines()\n"
                           "name = Path(getattr(json, '<own>')).name\n"),
        # no name at all
        "words.py": '"""Says <own> in a docstring."""\n\nNAME = "uses <own> in a string"  # and <own> in a comment\n',
    }
    for rel, text in sources.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.replace("<own>", OWN), encoding="utf-8")
    assert _own_file_uses(tmp_path) == [
        "alone.py:2",
        "alone.py:3",
        "dirname.py:3",
        "held_parent.py:4",
        "held_parents.py:4",
        "held_parents.py:5",
        "held_up_step.py:4",
        "held_up_step.py:5",
        "in_folder/deeper/one_of_two.py:4",
        "parent_twice.py:3",
        "parents.py:3",
        "resolved_parent_twice.py:3",
    ]


# ---------------------------------------------------------------------------------------------
# Every relative link in a tracked .md file points to something that exists


def _broken_links(root: Path, names: list[str]) -> list[str]:
    """`file:line: target does not exist` for every relative link `[text](target)` in the .md files `names`
    (paths below `root`) whose target is not a file or folder. A target is read from the folder of the file
    that holds the link. The scan is `dead_links` of tests/support/layout_helpers.py, which tests/repo/test_readme.py
    uses too."""
    return [f"{name}:{line}: {target} does not exist" for name in names
            for line, target in dead_links((root / name).read_text(encoding="utf-8"), (root / name).parent)]


def test_every_relative_link_in_a_tracked_markdown_file_points_to_something_that_exists():
    # Links are what a move of a document breaks without any test noticing: the page is still there, and
    # the path in the next page is not. This reads the files git tracks, so a note in a folder of the
    # working tree that git ignores is not asked.
    done = subprocess.run(["git", "ls-files", "-z", "--", "*.md"], cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=60)
    assert done.returncode == 0, done.stderr
    names = [name for name in done.stdout.split("\0") if name and (REPO / name).is_file()]
    assert {"README.md", "CLAUDE.md"} <= set(names)  # the real list
    assert _broken_links(REPO, names) == []


def test_the_link_scan_finds_a_dead_link_and_skips_the_rest(tmp_path):
    (tmp_path / "docs" / "sub").mkdir(parents=True)
    (tmp_path / "docs" / "other.md").write_text("# Other\n", encoding="utf-8")
    (tmp_path / "docs" / "guide.md").write_text("\n".join([
        "# Guide",
        "A [live page](other.md), a [page and its anchor](other.md#top), a [folder](sub) and the [root](../top.md).",
        "A [dead page](gone.md) and a [dead page with an anchor](also_gone.md#part).",
        "The [site](https://example.org/x), a [mail](mailto:someone@example.org) and a [jump](#guide).",
        "A link written from the root, not from this folder: [other](docs/other.md).",
        "An image ![shape](shape.png) and a [titled link](other.md \"the title\").",
        "In `code [a](code_gone.md)` and below, no link is read:",
        "```",
        "[b](block_gone.md)",
        "```",
        "~~~text",
        "[c](tilde_gone.md)",
        "~~~",
        "After the blocks: [d](after_gone.md).",
        "A link whose text wraps [onto the next",
        "line](wrapped_gone.md) is read, and so is a wrapped [live one",
        "too](other.md).",
        "A `code span that wraps",
        "[e](span_gone.md)` is code, not a link.",
        "```",
        "~~~",
        "[f](inside_gone.md)",
        "```",
        "After the block that holds a tilde line: [g](later_gone.md).",
        ""]), encoding="utf-8")
    (tmp_path / "top.md").write_text("A [guide](docs/guide.md) and [one that is not](docs/nothing.md).\n",
                                     encoding="utf-8")
    assert _broken_links(tmp_path, ["docs/guide.md", "top.md"]) == [
        "docs/guide.md:3: gone.md does not exist",
        "docs/guide.md:3: also_gone.md does not exist",
        "docs/guide.md:5: docs/other.md does not exist",
        "docs/guide.md:6: shape.png does not exist",
        "docs/guide.md:14: after_gone.md does not exist",
        "docs/guide.md:16: wrapped_gone.md does not exist",  # the line that holds the target
        "docs/guide.md:24: later_gone.md does not exist",
        "top.md:1: docs/nothing.md does not exist",
    ]


# ---------------------------------------------------------------------------------------------
# core/provenance.py


def test_provenance_names_the_folder_of_the_imported_package():
    # `PACKAGE_FOLDER` is where `git rev-parse` looks for the checkout. It is the folder of the imported
    # package, which stays the package's top folder wherever core/provenance.py lies in it. This holds before
    # the line that says so too: it is the guard for the move of the module (W2 step 2).
    assert provenance.PACKAGE_FOLDER == Path(outline_tracker.__file__).resolve().parent
