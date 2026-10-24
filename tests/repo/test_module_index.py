"""The module index, docs/developer/modules.md (docs/ROADMAP.md, W2 step 5).

The page is generated from the first line of the docstring of every module of `outline_tracker/`, by
tests/support/module_index.py, as docs/OUTPUTS.md is generated from the schema. These tests keep it so:

- the committed page is byte for byte what the generator writes, and when it is not, the message names the
  command that writes it (change a module's first line and this is the test that fails);
- every .py file of the package is one row of the page, in the section of its folder, and the sections are in
  the order of the layers;
- the generator is run on made-up packages in `tmp_path`: the text it writes is written out here by hand, and
  it raises, naming the module, for a module without a docstring and for one whose first line is empty;
- the first line of every module's docstring is a sentence of its own, at most 110 characters long;
- docs/developer/README.md links to the index, keeps no table of modules, and names only files that exist.

Paths are repository paths. Nothing here has units or a coordinate frame.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
from layout_helpers import PACKAGE_NAME, REPO
from layout_helpers import path as table_path

import module_index

PACKAGE = REPO / PACKAGE_NAME
PAGE = table_path("module index page")
GENERATOR = table_path("module index generator")
DEVELOPER = table_path("developer page")

# The sections of the page, in the order of the layers, as their headings are written (docs/ROADMAP.md, W2).
SECTIONS = ["outline_tracker/", "core/", "video/", "measure/", "segmenter/", "tracking/", "synthetic/",
            "export/", "workflows/", "gui/", "gui/view/", "gui/widgets/", "gui/panels/", "cli/"]
MAX_FIRST_LINE = 110  # characters


def package_files() -> list[str]:
    """Every .py file of the package as its path under `outline_tracker/`, with `/`, sorted."""
    return sorted(file.relative_to(PACKAGE).as_posix() for file in PACKAGE.rglob("*.py"))


def first_line(file: Path) -> str:
    """The first line of the module docstring of `file`, read here with `ast` and not with the generator."""
    docstring = ast.get_docstring(ast.parse(file.read_text(encoding="utf-8")), clean=False)
    assert docstring, f"{file.relative_to(REPO).as_posix()} has no docstring"
    return docstring.splitlines()[0]


def sections_of_the_page() -> list[tuple[str, str, list[str]]]:
    """`(heading, sentence, rows)` for each section of the committed page, in its order: the heading without its
    backticks, the line under it, and the module path in the first cell of each row of its table."""
    found: list[tuple[str, str, list[str]]] = []
    lines = PAGE.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines):
        if heading := re.fullmatch(r"## `([^`]+)`", line):
            found.append((heading.group(1), lines[number + 2], []))
        elif row := re.match(r"\| `([^`]+\.py)` \|", line):
            found[-1][2].append(row.group(1))
    return found


def folder_of(heading: str) -> str:
    """The folder a section is about, as the path under `outline_tracker/` that a row of it starts with."""
    return "." if heading == f"{PACKAGE_NAME}/" else heading.rstrip("/")


# ---------------------------------------------------------------------------------------------
# The committed page


def test_the_page_is_byte_for_byte_what_the_generator_writes_and_the_message_names_the_command():
    if not PAGE.is_file():
        pytest.fail(f"{PAGE.relative_to(REPO).as_posix()} is missing; run: {module_index.COMMAND}", pytrace=False)
    if PAGE.read_bytes() != module_index.module_index().encode("utf-8"):
        pytest.fail(f"{PAGE.relative_to(REPO).as_posix()} is out of date; run: {module_index.COMMAND}", pytrace=False)


def test_the_page_says_in_its_first_line_that_it_is_generated_and_by_which_command():
    first = PAGE.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("<!-- Generated from ") and first.endswith('. Do not edit by hand. -->')
    assert f'by "{module_index.COMMAND}"' in first


def test_every_module_is_one_row_of_the_page_in_the_section_of_its_folder_and_no_row_is_missing_a_file():
    sections, files = sections_of_the_page(), package_files()
    rows = [row for _, _, found in sections for row in found]
    assert len(files) > 80  # the real package
    assert [row for row in rows if rows.count(row) > 1] == []  # once
    assert [name for name in files if name not in rows] == []  # every module
    assert [row for row in rows if not (PACKAGE / row).is_file()] == []  # no row for a file that is not there
    for heading, _, found in sections:
        in_folder = [name for name in files if PurePosixPath(name).parent.as_posix() == folder_of(heading)]
        by_name = sorted(in_folder, key=lambda name: (PurePosixPath(name).name != "__init__.py", name))
        assert found == by_name, heading  # its own modules, sorted by path, __init__.py first


def test_the_sections_are_in_the_order_of_the_layers_and_each_opens_with_the_line_of_its_init():
    sections = sections_of_the_page()
    assert [heading for heading, _, _ in sections] == SECTIONS
    for heading, sentence, _ in sections:
        init = PACKAGE / folder_of(heading) / "__init__.py"
        assert sentence == first_line(init), heading


# ---------------------------------------------------------------------------------------------
# The generator, on made-up packages


def make_package(root: Path, files: dict[str, str]) -> Path:
    """The package `pkg` in `root` with these files (path under it -> text), and the folders they need."""
    package = root / "pkg"
    for name, text in files.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return package


# A small package that has every case of the text: the order of the layers is not the order of the alphabet
# (`cli` comes after `core`), `__init__.py` comes before `__main__.py` and `a.py`, only the first line of a
# docstring is used, a bar in a line is escaped in a table, and a letter outside ASCII is written as it is.
MADE_UP = {
    "__init__.py": '"""The top of pkg.\n\nA second paragraph."""\n',
    "launch.py": '"""Starts the tool, `launch.main`.\n\nThe second paragraph."""\n',
    "cli/__init__.py": '"""The command line."""\n',
    "cli/main.py": '"""Entry point: a | b."""\n',
    "core/b.py": '"""B holds the é of café."""\n',
    "core/a.py": '"""A holds a.\nIts second line is not used."""\n',
    "core/__main__.py": '"""Runs `core.a`."""\nimport os\n',
    "core/__init__.py": '"""The base."""\n',
    "gui/view/__init__.py": '"""What is drawn."""\n',
    "gui/view/draw.py": '"""Draws it."""\n',
    "gui/__init__.py": '"""The window."""\n',
    "gui/app.py": '"""Starts the window."""\n',
}

MADE_UP_SECTIONS = """\
## `pkg/`

The top of pkg.

| module | what it holds |
|---|---|
| `__init__.py` | The top of pkg. |
| `launch.py` | Starts the tool, `launch.main`. |

## `core/`

The base.

| module | what it holds |
|---|---|
| `core/__init__.py` | The base. |
| `core/__main__.py` | Runs `core.a`. |
| `core/a.py` | A holds a. |
| `core/b.py` | B holds the é of café. |

## `gui/`

The window.

| module | what it holds |
|---|---|
| `gui/__init__.py` | The window. |
| `gui/app.py` | Starts the window. |

## `gui/view/`

What is drawn.

| module | what it holds |
|---|---|
| `gui/view/__init__.py` | What is drawn. |
| `gui/view/draw.py` | Draws it. |

## `cli/`

The command line.

| module | what it holds |
|---|---|
| `cli/__init__.py` | The command line. |
| `cli/main.py` | Entry point: a \\| b. |
"""


def test_the_generator_writes_a_section_for_each_folder_in_the_order_of_the_layers(tmp_path):
    text = module_index.module_index(make_package(tmp_path, MADE_UP))
    assert text[text.index("\n## ") + 1:] == MADE_UP_SECTIONS  # everything from the first section on
    assert "\r" not in text and text.endswith("|\n") and not text.endswith("\n\n")  # LF, one line end at the end


def test_the_generator_writes_a_header_a_title_and_one_sentence_before_the_sections(tmp_path):
    lines = module_index.module_index(make_package(tmp_path, MADE_UP)).splitlines()
    assert lines[0].startswith("<!-- Generated from ") and lines[0].endswith("Do not edit by hand. -->")
    assert module_index.COMMAND in lines[0]
    assert lines[1] == "" and lines[2].startswith("# ") and lines[3] == ""
    assert lines[4] and lines[5] == "" and lines[6] == "## `pkg/`"  # one paragraph, then the first section


def test_the_generator_gives_the_same_text_every_time(tmp_path):
    package = make_package(tmp_path, MADE_UP)
    assert module_index.module_index(package) == module_index.module_index(package)


def test_a_folder_without_modules_is_not_a_section(tmp_path):
    package = make_package(tmp_path, {"__init__.py": '"""Top."""\n', "launch.py": '"""Starts."""\n'})
    assert [line for line in module_index.module_index(package).splitlines() if line.startswith("## ")] == ["## `pkg/`"]


@pytest.mark.parametrize("name, text, said", [
    ("core/x.py", "x = 1\n", "no docstring"),
    ("core/x.py", 'import os\n"""Not the first statement."""\n', "no docstring"),
    ("core/x.py", '"""\nThe text begins on the second line.\n"""\n', "first line"),
    ("core/x.py", '""""""\n', "first line"),
    ("core/x.py", '"""   \nOnly spaces on the first line.\n"""\n', "first line"),
    ("core/__init__.py", "x = 1\n", "no docstring"),
    ("launch.py", "", "no docstring"),
], ids=["no-docstring", "docstring-not-first", "starts-on-second-line", "empty-docstring", "spaces-only",
        "package-without-docstring", "empty-file"])
def test_the_generator_raises_with_the_path_of_a_module_without_a_first_line(tmp_path, name, text, said):
    package = make_package(tmp_path, {"__init__.py": '"""Top."""\n', "core/__init__.py": '"""The base."""\n'})
    (package / name).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"pkg/{name}")) as raised:
        module_index.module_index(package)
    assert said in str(raised.value)


def test_the_generator_raises_for_a_folder_that_is_not_in_the_order_of_the_layers(tmp_path):
    package = make_package(tmp_path, {"__init__.py": '"""Top."""\n', "extra/__init__.py": '"""New."""\n'})
    with pytest.raises(ValueError, match="extra"):
        module_index.module_index(package)


def test_the_generator_raises_for_a_folder_with_modules_and_no_init(tmp_path):
    package = make_package(tmp_path, {"__init__.py": '"""Top."""\n', "core/a.py": '"""A."""\n'})
    with pytest.raises(ValueError, match=re.escape("pkg/core/__init__.py")):
        module_index.module_index(package)


def test_the_command_writes_the_page_as_utf8_with_lf_line_ends(tmp_path):
    target = tmp_path / "modules.md"
    done = subprocess.run([sys.executable, str(GENERATOR), str(target)], capture_output=True, text=True,
                          cwd=REPO, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout == "" and done.stderr == ""
    assert target.read_bytes() == module_index.module_index().encode("utf-8")


def test_the_command_needs_one_path():
    done = subprocess.run([sys.executable, str(GENERATOR)], capture_output=True, text=True, cwd=REPO, timeout=60)
    assert done.returncode == 2
    assert "usage" in done.stderr.lower() and "OUTPUT.md" in done.stderr


# ---------------------------------------------------------------------------------------------
# The first lines


def test_the_first_line_of_every_module_docstring_is_at_most_110_characters():
    too_long = {name: len(first_line(PACKAGE / name)) for name in package_files()
                if len(first_line(PACKAGE / name)) > MAX_FIRST_LINE}
    assert too_long == {}


def test_the_first_line_of_every_module_docstring_is_a_sentence_that_ends_on_that_line():
    # A first line that goes on in the next line ends in a word or a comma; the index shows the first line alone.
    open_ended = {name: first_line(PACKAGE / name) for name in package_files()
                  if not first_line(PACKAGE / name).endswith(".")}
    assert open_ended == {}


# ---------------------------------------------------------------------------------------------
# docs/developer/README.md


def test_the_developers_page_links_to_the_index_and_gives_the_command_and_keeps_no_table_of_modules():
    text = DEVELOPER.read_text(encoding="utf-8")
    link = Path(os.path.relpath(PAGE, DEVELOPER.parent)).as_posix()
    assert f"]({link})" in text
    assert module_index.COMMAND in text
    assert re.findall(r"^\| `[\w/]+\.py` \|", text, flags=re.MULTILINE) == []  # no row of a table of modules
    assert "| module |" not in text


def test_every_py_file_the_developers_page_names_exists():
    text = DEVELOPER.read_text(encoding="utf-8")
    named = set(re.findall(r"`([\w/]+\.py)`", text))
    assert len(named) > 5  # the page names some
    # a file of the package, or of the repository
    assert [name for name in sorted(named) if not (PACKAGE / name).is_file() and not (REPO / name).is_file()] == []
