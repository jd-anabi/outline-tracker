"""The page docs/design/decisions.md holds the decisions X1 to X22 of the first build as docs/PLAN.md had them.

The page is a copy of section 4 of the plan (docs/ROADMAP.md, W2 step 3b). These tests hold it to the original:

- the ids X1 to X22 are in the table once each, in order;
- the 22 rows of the table are the plan's rows, byte for byte;
- the page ends with the whole of section 4 of the plan (the sentence under the heading, the table,
  the paragraph of the test conditions and its table), unchanged.

The original is `docs/PLAN.md` for as long as it is in the tree. When it has left (W2 step 3c), it
is the same file at the commit that the page names in its first paragraph, read with
`git show COMMIT:docs/PLAN.md`, and the comparison is made all the same. It is skipped only when that file
cannot be read: git is not installed, or the commit is not in the clone (a checkout without `.git`, or
without its history, as CI's is), as tests/core/test_provenance.py does for a checkout without `.git`. The
ids are checked in every case. `read_plan` is checked on a made-up repository, so that the step after 3c is
not tried for the first time when the file has already gone.

No quantities here: no units and no coordinates.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from layout_helpers import REPO

PAGE = REPO / "docs" / "design" / "decisions.md"
PLAN = "docs/PLAN.md"
IDS = [f"X{number}" for number in range(1, 23)]
HEADING = re.compile(rb"^## 4\. ", re.MULTILINE)
ROW = re.compile(rb"^\| (X[0-9]+) \|")
NAMED_COMMIT = re.compile(r"copied unchanged from section 4 of `docs/PLAN\.md` at commit `([0-9a-f]{7})`")
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def section_four(plan: bytes) -> bytes:
    """Section 4 of the plan as the page holds it: the lines below its heading `## 4. ...` and above the
    next line `---`, without the blank lines at the start and the end, and with one line end at the end."""
    found = HEADING.search(plan)
    assert found, "the plan has no heading of section 4"
    lines = plan[found.end():].split(b"\n")[1:]  # the rest of the heading's own line is not part of it
    end = lines.index(b"---")
    return b"\n".join(lines[:end]).strip(b"\n") + b"\n"


def rows(text: bytes) -> list[bytes]:
    """The lines of the table of decisions: the ones that start with `| X` and a number."""
    return [line for line in text.split(b"\n") if ROW.match(line)]


def ids(text: bytes) -> list[str]:
    """The ids (`X1`, ...) of the rows of the table, in the order they stand."""
    return [ROW.match(line).group(1).decode() for line in rows(text)]


def named_commit(page: bytes) -> str:
    """The seven digits of the commit that the page's first paragraph names as the one it was copied at."""
    found = NAMED_COMMIT.search(page.decode("utf-8"))
    assert found, "the page does not say 'copied unchanged from section 4 of `docs/PLAN.md` at commit `1234567`'"
    return found.group(1)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run git for `repo` whatever the user's own git settings are, and give back what it did."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}  # as inside a hook
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com",
                           "-c", "commit.gpgsign=false", *args], capture_output=True, timeout=60, env=env)


def read_plan(repo: Path, commit: str) -> tuple[bytes, str]:
    """The bytes of docs/PLAN.md and a few words for where they came from: the file in the tree of `repo`
    while there is one, else the file at `commit` (`git show`). Skips the test when git is not installed
    or `commit` is not in the clone. Fails when the commit has no such file: the page names a wrong one."""
    plan = repo / PLAN
    if plan.is_file():
        return plan.read_bytes(), PLAN
    if shutil.which("git") is None:
        pytest.skip(f"{PLAN} is not in the tree and git is not installed, so the file at commit {commit} cannot be read")
    if _git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        pytest.skip(f"{PLAN} is not in the tree and commit {commit} is not in this clone (no .git, or a shallow one)")
    shown = _git(repo, "show", f"{commit}:{PLAN}")
    assert shown.returncode == 0, f"commit {commit} has no {PLAN}: {shown.stderr.decode('utf-8', 'replace')}"
    return shown.stdout, f"{PLAN} at commit {commit}"


def _page() -> bytes:
    assert PAGE.is_file(), "docs/design/decisions.md does not exist"
    return PAGE.read_bytes()


# ---------------------------------------------------------------------------------------------
# The page


def test_the_table_of_the_page_has_x1_to_x22_once_each_in_order():
    assert ids(_page()) == IDS


def test_the_page_names_the_commit_of_the_copy_and_links_the_roadmap():
    page = _page()
    assert named_commit(page)  # seven digits
    assert b"](../ROADMAP.md" in page  # a relative link from docs/design/ to docs/ROADMAP.md (the link scan of
    # tests/repo/test_layout.py asks that the target exists)


def test_the_rows_of_the_page_are_the_rows_of_the_plan_byte_for_byte():
    page = _page()
    plan, source = read_plan(REPO, named_commit(page))
    assert ids(section_four(plan)) == IDS  # the scan sees the real table
    assert rows(page) == rows(section_four(plan)), f"the rows of the page differ from section 4 of {source}"


def test_the_page_ends_with_section_four_of_the_plan_unchanged():
    page = _page()
    plan, source = read_plan(REPO, named_commit(page))
    assert page.endswith(section_four(plan)), f"the page does not end with section 4 of {source}, unchanged"


# ---------------------------------------------------------------------------------------------
# The checks themselves, on made-up text and a made-up repository


def test_section_four_is_cut_between_its_heading_and_the_rule_that_ends_it():
    plan = (b"# Plan\n\n---\n\n## 3. Before\n\nnot this\n| X9 | not this row |\n\n---\n\n"
            b"## 4. Decisions where the spec leaves a choice\n\nIntro.\n\n| # | decision |\n|---|---|\n"
            b"| X1 | one |\n| X2 | two |\n\nA paragraph.\n\n---\n\n## 5. After\n\nnor this\n| X3 | nor this row |\n")
    assert section_four(plan) == (b"Intro.\n\n| # | decision |\n|---|---|\n| X1 | one |\n| X2 | two |\n\n"
                                  b"A paragraph.\n")
    assert rows(section_four(plan)) == [b"| X1 | one |", b"| X2 | two |"]
    assert ids(section_four(plan)) == ["X1", "X2"]


def test_a_row_starts_with_a_pipe_and_an_id_and_the_named_commit_has_seven_digits():
    assert rows(b"| # | decision |\n| X10 | ten |\nX11 not a row\n| Y1 | no |\n  | X12 | indented |\n") == [b"| X10 | ten |"]
    sentence = "It is copied unchanged from section 4 of `docs/PLAN.md` at commit `67908d8`."
    assert named_commit(sentence.encode()) == "67908d8"
    for wrong in ("at commit `67908d`", "at commit `67908d8e`", "at commit `HEAD`"):
        with pytest.raises(AssertionError, match="does not say"):
            named_commit(sentence.replace("at commit `67908d8`", wrong).encode())


@needs_git
def test_the_plan_is_read_from_the_tree_while_it_is_there_and_at_the_named_commit_after(tmp_path):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    committed = b"## 4. Decisions\n\nThe committed text.\n\n---\n"
    (repo / PLAN).write_bytes(committed)
    assert _git(repo, "init", "-q").returncode == 0
    assert _git(repo, "add", PLAN).returncode == 0
    assert _git(repo, "commit", "-q", "-m", "one").returncode == 0
    commit = _git(repo, "rev-parse", "--short=7", "HEAD").stdout.decode().strip()
    assert len(commit) == 7

    (repo / PLAN).write_bytes(b"## 4. Decisions\n\nChanged in the tree.\n")
    assert read_plan(repo, commit) == (b"## 4. Decisions\n\nChanged in the tree.\n", PLAN)  # the tree first

    (repo / PLAN).unlink()
    assert read_plan(repo, commit) == (committed, f"{PLAN} at commit {commit}")  # then the commit

    with pytest.raises(pytest.skip.Exception, match="not in this clone"):
        read_plan(repo, "0000000")  # a commit that is not here: skipped, with the reason
    assert _git(repo, "rm", "-q", "--cached", PLAN).returncode == 0
    assert _git(repo, "commit", "-q", "-m", "two").returncode == 0
    later = _git(repo, "rev-parse", "--short=7", "HEAD").stdout.decode().strip()
    with pytest.raises(AssertionError, match=f"commit {later} has no {re.escape(PLAN)}"):
        read_plan(repo, later)  # a commit without the file: the page names a wrong one, which is no skip


def test_without_git_and_without_the_file_the_plan_cannot_be_read_and_the_test_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(pytest.skip.Exception, match="git is not installed"):
        read_plan(tmp_path, "67908d8")
