"""The size of a file: no .py file of `outline_tracker/` or `tests/` is over 400 lines (docs/ROADMAP.md, W2 step 6).

A file that grows past the limit is split (docs/developer/README.md says so); this test is what notices it.
The length is `len(text.splitlines())`, the number `wc -l` gives for a file that ends with a line end.

`OVER_400` names the files that were over the limit when W2 step 6 began, each with the task of that step that
splits it: 6a the package files, 6b the tests of core/ and segmenter/, 6c the tests of repo/, support/, tracking/,
synthetic/, measure/ and slow/, 6d the tests of gui/. The scan of the real repository must find exactly those:

- a file over 400 lines that is not in the list fails the test, so a new long file cannot slip in;
- a file in the list that is no longer over 400 lines fails it too, so the commit that splits a file takes its
  entry off, and the list ends empty.

When the list is empty it stays in the file, empty, as the place a future exception would have to be written
down, with the reason.

The scan is checked on small made-up trees in `tmp_path`, so that a scan that finds nothing would show up here.
Lengths are numbers of lines; paths are repository paths written with `/`.
"""

from pathlib import Path

import pytest
from layout_helpers import PACKAGE_NAME, REPO

LIMIT = 400  # lines

# Where the scan looks: the package and the tests, relative to the repository.
SCANNED = (PACKAGE_NAME, "tests")

# The files over LIMIT when W2 step 6 began, each with the task that splits it (6a, the package files, and 6b, the
# tests of core/ and segmenter/, are all split).
# An entry comes off in the commit that splits its file: the first test below fails for a long file without an
# entry, the second for an entry whose file is no longer long.
OVER_400: dict[str, str] = {
    # 6c: the tests of repo/, support/, tracking/, synthetic/, measure/ and slow/
    "tests/slow/test_regression_reference.py": "6c",
    # 6d: the tests of gui/
    "tests/gui/test_export_panel.py": "6d",
    "tests/gui/test_joins.py": "6d",
    "tests/gui/test_model_choice.py": "6d",
    "tests/gui/test_prompts.py": "6d",
    "tests/gui/test_review.py": "6d",
    "tests/gui/test_session_panels.py": "6d",
    "tests/gui/test_session_saving.py": "6d",
    "tests/gui/test_tools.py": "6d",
    "tests/gui/test_tracking_panel.py": "6d",
    "tests/gui/test_worker.py": "6d",
    "tests/gui/test_worker_jobs.py": "6d",
}

# The folders of tests/ that each task of step 6 splits; the package belongs to 6a.
TASK_OF_FOLDER = {
    "outline_tracker": "6a",
    "tests/core": "6b", "tests/segmenter": "6b",
    "tests/repo": "6c", "tests/support": "6c", "tests/tracking": "6c", "tests/synthetic": "6c",
    "tests/measure": "6c", "tests/slow": "6c",
    "tests/gui": "6d",
}


def length(file: Path) -> int:
    """The number of lines of `file`, counted as `wc -l` counts a file that ends with a line end."""
    return len(file.read_text(encoding="utf-8").splitlines())


def python_files(root: Path) -> dict[str, int]:
    """Every .py file under the scanned folders of `root` as `{path with "/" relative to root: lines}`."""
    return {file.relative_to(root).as_posix(): length(file)
            for folder in SCANNED if (root / folder).is_dir()
            for file in (root / folder).rglob("*.py")}


def too_long_and_not_listed(root: Path, listed: dict[str, str]) -> list[str]:
    """The files under `root` with more than LIMIT lines that `listed` does not name, with their lengths."""
    return [f"{path} has {lines} lines" for path, lines in sorted(python_files(root).items())
            if lines > LIMIT and path not in listed]


def listed_and_not_too_long(root: Path, listed: dict[str, str]) -> list[str]:
    """The entries of `listed` whose file is missing under `root` or has LIMIT lines or fewer."""
    found = python_files(root)
    return [f"{path} is listed but " + ("is not a .py file of the scanned folders" if path not in found
                                        else f"has {found[path]} lines")
            for path in sorted(listed) if path not in found or found[path] <= LIMIT]


# ---------------------------------------------------------------------------------------------
# The repository


def test_no_file_is_over_400_lines_unless_it_is_in_the_list():
    assert python_files(REPO)  # the real repository: the scan found files
    assert too_long_and_not_listed(REPO, OVER_400) == []


def test_every_file_in_the_list_is_still_over_400_lines():
    # Fails for the commit that splits a file and leaves its entry in the list.
    assert listed_and_not_too_long(REPO, OVER_400) == []


def test_each_entry_of_the_list_names_the_task_of_its_folder():
    wrong = []
    for path, task in OVER_400.items():
        folder = next((name for name in TASK_OF_FOLDER if path.startswith(name + "/")), None)
        if folder is None or TASK_OF_FOLDER[folder] != task:
            wrong.append(f"{path}: listed under {task}")
    assert wrong == []


# ---------------------------------------------------------------------------------------------
# The scan, on made-up trees


def make_tree(root: Path, lines_of: dict[str, int]) -> None:
    """Write `{path: n}` as files of n lines each under `root`."""
    for path, count in lines_of.items():
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("x = 1\n" * count, encoding="utf-8")


def test_a_file_of_exactly_400_lines_is_fine_and_401_is_too_long(tmp_path):
    make_tree(tmp_path, {"outline_tracker/a.py": 400, "outline_tracker/b.py": 401, "tests/c.py": 1})
    assert too_long_and_not_listed(tmp_path, {}) == ["outline_tracker/b.py has 401 lines"]


def test_the_scan_looks_in_subfolders_of_the_package_and_of_the_tests(tmp_path):
    make_tree(tmp_path, {"outline_tracker/gui/panels/deep.py": 450, "tests/gui/test_deep.py": 460})
    assert too_long_and_not_listed(tmp_path, {}) == ["outline_tracker/gui/panels/deep.py has 450 lines",
                                                     "tests/gui/test_deep.py has 460 lines"]


def test_a_long_file_in_the_list_is_not_reported_as_too_long(tmp_path):
    make_tree(tmp_path, {"outline_tracker/long.py": 450, "tests/other.py": 460})
    assert too_long_and_not_listed(tmp_path, {"outline_tracker/long.py": "6a"}) == ["tests/other.py has 460 lines"]


def test_an_entry_whose_file_is_short_missing_or_not_scanned_is_reported(tmp_path):
    make_tree(tmp_path, {"outline_tracker/split.py": 400, "outline_tracker/long.py": 401, "docs/elsewhere.py": 900})
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "notes.txt").write_text("x\n" * 900, encoding="utf-8")
    listed = {"outline_tracker/split.py": "6a", "outline_tracker/long.py": "6a", "outline_tracker/gone.py": "6a",
              "docs/elsewhere.py": "6a", "tests/notes.txt": "6b"}
    assert listed_and_not_too_long(tmp_path, listed) == [
        "docs/elsewhere.py is listed but is not a .py file of the scanned folders",
        "outline_tracker/gone.py is listed but is not a .py file of the scanned folders",
        "outline_tracker/split.py is listed but has 400 lines",
        "tests/notes.txt is listed but is not a .py file of the scanned folders",
    ]


def test_only_py_files_of_the_package_and_the_tests_count(tmp_path):
    make_tree(tmp_path, {"outline_tracker/data.txt": 900, "docs/script.py": 900, "tests/data/table.csv": 900,
                         "other/module.py": 900})
    assert python_files(tmp_path) == {}
    assert too_long_and_not_listed(tmp_path, {}) == []


def test_the_length_does_not_depend_on_the_kind_of_line_end(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "windows.py").write_bytes(b"x = 1\r\n" * 401)
    (tmp_path / "tests" / "no_end.py").write_bytes(b"x = 1\n" * 399 + b"x = 1")  # the last line has no line end
    assert python_files(tmp_path) == {"tests/windows.py": 401, "tests/no_end.py": 400}


def test_an_empty_list_reports_every_long_file(tmp_path):
    make_tree(tmp_path, {"outline_tracker/a.py": 401, "tests/b.py": 402})
    assert len(too_long_and_not_listed(tmp_path, {})) == 2
    assert listed_and_not_too_long(tmp_path, {}) == []


@pytest.mark.parametrize("count", [1, 399, 400])
def test_a_listed_file_that_is_not_over_the_limit_is_reported(tmp_path, count):
    make_tree(tmp_path, {"tests/short.py": count})
    assert listed_and_not_too_long(tmp_path, {"tests/short.py": "6b"}) == [
        f"tests/short.py is listed but has {count} lines"]
