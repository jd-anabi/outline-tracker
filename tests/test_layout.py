"""The table of paths (tests/layout_helpers.py) and the rules that keep the tests on it (docs/ROADMAP.md, W2).

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
from layout_helpers import PATHS, REPO, below, module
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
    """The text of tests/layout_helpers.py, copied to `into`, and loaded from there."""
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
    The attribute of an imported module (`json.<own>`) is not the bare name; a docstring, a string and a
    comment hold no name. The files are read as syntax trees."""
    found = []
    for path in sorted(folder.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        lines = sorted(node.lineno for node in ast.walk(tree)
                       if isinstance(node, ast.Name) and node.id == OWN and not _is_own_folder(node, parents))
        found += [f"{path.relative_to(folder).as_posix()}:{line}" for line in lines]
    return found


def _is_own_folder(name: ast.Name, parents: dict) -> bool:
    """Whether the name `<own>` is the one in `Path(<own>).resolve().parent` or `Path(<own>).parent`, and
    that expression goes no further than `.parent` (a second `.parent`, or a `.parents`, is a folder higher
    up)."""
    call = parents.get(name)
    func = getattr(call, "func", None)
    named_path = (isinstance(func, ast.Name) and func.id == "Path"
                  or isinstance(func, ast.Attribute) and func.attr == "Path")
    if not (isinstance(call, ast.Call) and call.args == [name] and not call.keywords and named_path):
        return False
    steps, current = [], call
    while True:
        up = parents.get(current)
        if isinstance(up, ast.Attribute) and up.value is current:
            steps.append(up.attr)
            current = up
        elif isinstance(up, ast.Call) and up.func is current and not up.args and not up.keywords:
            current = up  # `.resolve()` has no arguments
        else:
            return steps in (["parent"], ["resolve", "parent"])


def test_no_test_takes_a_folder_of_the_repository_from_its_own_file():
    # A line such as `REPO = Path(<own>).resolve().parents[1]` is true for one depth of one file. When the
    # file moves, the line is wrong and the test fails, or worse, passes on the wrong folder. The tests ask
    # tests/layout_helpers.py (`REPO`, `path`, `module`), which is the one file that may use its own place.
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
        # the form that stays
        "own_folder.py": imports + "SLOW = Path(<own>).resolve().parent\n",
        "own_folder_plain.py": imports + "HERE = Path(<own>).parent\nDATA = HERE / 'data'\n",
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
        "in_folder/deeper/one_of_two.py:4",
        "parent_twice.py:3",
        "parents.py:3",
        "resolved_parent_twice.py:3",
    ]


# ---------------------------------------------------------------------------------------------
# Every relative link in a tracked .md file points to something that exists

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s[^)]*)?\)")  # `[text](target)`, also with a title after the target
SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")  # `https:`, `mailto:`


def _broken_links(root: Path, names: list[str]) -> list[str]:
    """`file:line: target does not exist` for every relative link `[text](target)` in the .md files `names`
    (paths below `root`) whose target is not a file or folder. A target is read from the folder of the file
    that holds the link; an anchor (`#...`) is cut off. A link with a scheme (`https:`, `mailto:`) and a
    link that is only an anchor are skipped. Code is not read: neither fenced blocks nor `inline code`."""
    found = []
    for name in names:
        folder, inside = (root / name).parent, False
        for number, line in enumerate((root / name).read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith(("```", "~~~")):
                inside = not inside
            elif not inside:
                for target in LINK.findall(re.sub(r"`[^`]*`", "", line)):
                    local = target.split("#")[0]
                    if local and not SCHEME.match(target) and not (folder / local).exists():
                        found.append(f"{name}:{number}: {local} does not exist")
    return found


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
        ""]), encoding="utf-8")
    (tmp_path / "top.md").write_text("A [guide](docs/guide.md) and [one that is not](docs/nothing.md).\n",
                                     encoding="utf-8")
    assert _broken_links(tmp_path, ["docs/guide.md", "top.md"]) == [
        "docs/guide.md:3: gone.md does not exist",
        "docs/guide.md:3: also_gone.md does not exist",
        "docs/guide.md:5: docs/other.md does not exist",
        "docs/guide.md:6: shape.png does not exist",
        "docs/guide.md:14: after_gone.md does not exist",
        "top.md:1: docs/nothing.md does not exist",
    ]


# ---------------------------------------------------------------------------------------------
# core/provenance.py


def test_provenance_names_the_folder_of_the_imported_package():
    # `PACKAGE_FOLDER` is where `git rev-parse` looks for the checkout. It is the folder of the imported
    # package, which stays the package's top folder wherever core/provenance.py lies in it. This holds before
    # the line that says so too: it is the guard for the move of the module (W2 step 2).
    assert provenance.PACKAGE_FOLDER == Path(outline_tracker.__file__).resolve().parent
