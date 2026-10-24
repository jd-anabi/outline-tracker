"""The checks that read README.md, run on small made-up texts (task B6b).

The tests of tests/repo/test_readme.py hold the real README against the real tool with the checks of
tests/support/readme_helpers.py. A check that never fails would pass there for any README, so each test here gives a
check a text that is wrong in one way and expects the finding, and a text that is right and expects none: the
command check, the comment check, the link check, the install check, the table check, the output check (also with
the backslashes of Windows paths), the run folder check, the spelling check and the code block that is never closed.
Nothing here reads README.md. The tests of the Quickstart's reader (the step reader, the check for bold buttons) are
in tests/repo/test_readme.py, with the Quickstart.

Paths in this file are repository paths; no units or coordinates.
"""

import pytest
from readme_helpers import (BRITISH, TITLES, code_blocks, command_problems, comment_problems, install_lines,
                            install_problems, link_problems, output_problems, run_folder_names, table_problems,
                            tag_version)


# ---------------------------------------------------------------------------------------------
# The checks themselves: each finds what it is for


def fenced(language: str, *lines: str) -> str:
    return "\n".join([f"```{language}", *lines, "```"])


@pytest.mark.parametrize("bad", [
    fenced("zsh", "outline-tracker from-tracker v.mp4 e.csv --nope"),
    fenced("zsh", "outline-tracker frobnicate"),
    fenced("powershell", "outline-tracker export"),
    fenced("zsh", 'outline-tracker check "a video.mp4'),
    fenced("zsh", "outline-tracker from-tracker v.mp4 e.csv --fps fast"),
    "Run `outline-tracker check` now.",
    fenced("text", "outline-tracker selftest"),
    fenced("", "outline-tracker selftest"),
    fenced("shell", "outline-tracker --versions"),
], ids=["unknown option", "unknown command", "missing argument", "unbalanced quote", "wrong kind of value",
        "inline command without its video", "command in an output block", "untagged block", "unknown flag"])
def test_the_command_check_finds_what_the_parser_rejects(bad):
    assert command_problems(bad) != []


@pytest.mark.parametrize("good", [
    fenced("zsh", "outline-tracker selftest --device cpu"),
    fenced("powershell", 'outline-tracker from-tracker "path\\to\\v.mp4" "e.csv" --fine A,C --seconds 2'),
    fenced("shell", "outline-tracker --version"),
    fenced("text", "outline-tracker 0.1.0 (commit abc1234)"),
    "Run `outline-tracker export SESSION.json --overlay` now.",
])
def test_the_command_check_passes_real_commands(good):
    assert command_problems(good) == []


@pytest.mark.parametrize("bad", [
    fenced("zsh", "# check it", "outline-tracker selftest"),
    fenced("powershell", "outline-tracker selftest  # one minute"),
])
def test_the_comment_check_finds_comments_in_shell_blocks_only(bad):
    assert len(comment_problems(bad)) == 1
    assert comment_problems(fenced("python", "x = 1  # a comment") + "\n" + fenced("text", "# a heading")) == []


def test_the_link_check_finds_a_file_that_is_not_there(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "A.md").write_text("x")
    text = "[a](docs/A.md#top) [b](docs/B.md) [c](https://example.org/x) [d](#up)\n" + fenced("text", "[e](nope.md)")
    assert link_problems(text, tmp_path) == ["docs/B.md does not exist"]
    # the stricter reading of the one scan that tests/repo/test_layout.py shares: a titled link is a link, and
    # `inline code` and a block fenced with ~~~ are not
    more = '[f](docs/F.md "a title") and `[g](nope.md)`\n~~~text\n[h](nope.md)\n~~~\n'
    assert link_problems(more, tmp_path) == ["docs/F.md does not exist"]


TAGGED = "uv tool install git+https://github.com/jd-anabi/outline-tracker@refs/tags/v0.12.3 --python 3.12"


@pytest.mark.parametrize("second, problem", [
    (TAGGED.replace("@refs/tags/v0.12.3", ""), "no `@refs/tags/vX.Y.Z` after the address"),
    (TAGGED.replace("@refs/tags/v0.12.3", "@main"), "no `@refs/tags/vX.Y.Z` after the address"),
    (TAGGED.replace("v0.12.3", "v0.12.3rc1"), "no `@refs/tags/vX.Y.Z` after the address"),
    (TAGGED.replace("v0.12.3", "v0.12.4"), "another tag than v0.12.3"),
], ids=["no tag", "a branch", "a tag that is not a release", "another tag than the first"])
def test_the_install_check_finds_a_line_without_a_tag_and_a_line_with_another_tag(second, problem):
    bad = fenced("shell", TAGGED) + "\n" + fenced("shell", second)
    assert install_problems(bad) == [f"{problem}: {second}"]


def test_the_install_check_reads_the_lines_of_shell_blocks_that_install_from_the_repository():
    again = TAGGED.replace("install", "install --force")
    good = fenced("shell", TAGGED, "uv tool uninstall outline-tracker", "uv tool install --python 3.12 .", again)
    good += "\n" + fenced("text", TAGGED.replace("@refs/tags/v0.12.3", ""))  # printed text, not a command to paste
    assert install_lines(good) == [(TAGGED, "0.12.3"), (again, "0.12.3")]
    assert install_problems(good) == [] and tag_version(good) == "0.12.3"
    with pytest.raises(ValueError, match="no install line names a tag"):
        tag_version(fenced("shell", TAGGED.replace("@refs/tags/v0.12.3", ""), "uv tool uninstall outline-tracker"))


def test_the_table_check_finds_what_is_not_true():
    last, this = {"--seconds", "--no-video"}, {"--seconds", "--fine", "--no-overlay"}
    row = "| what | `--seconds S` | `--seconds S` |"
    assert table_problems(row, last, this) == []
    assert table_problems("| what | `--secondz S` | `--seconds S` |", last, this) != []
    assert table_problems("| what | `--seconds S` | `--fines S` |", last, this) != []
    assert table_problems("| what | not there | `--seconds S` |", last, this) != []  # not new: it was there


def test_the_output_check_finds_a_line_that_no_command_prints():
    real = ["OK: edgetam followed the test shrimp within 0.4 pixels (should be under 3).", "run folder: /tmp/a/b"]
    assert output_problems(fenced("text", "OK: edgetam followed the test shrimp within 12.5 pixels (should be under 3)."),
                           real) == []
    assert output_problems(fenced("text", "run folder: path/to/b"), real) == []
    assert output_problems(fenced("text", "run folder: ..."), real) == []
    assert output_problems(fenced("text", "OK: edgetam found the test shrimp"), real) != []
    assert output_problems(fenced("text", "OK: edgetam followed the test shrimp within 1 pixels (should be over 3)."),
                           real) != []


def test_the_output_check_reads_paths_the_same_with_slashes_and_with_backslashes():
    """`from-tracker` prints `str(Path)`: backslashes on Windows. The README writes them with `/`, once, for
    both systems; the Windows CI job must accept what Windows prints, and still refuse a wrong name."""
    folder = "D:\\recordings\\video_tracker_outline_sam"  # a drive letter and backslashes; not a home folder
    windows = [f"  run folder: {folder}",
               f"  saved {folder}\\edgetam\\A.csv, {folder}\\edgetam\\B.csv and {folder}\\overlay.mp4"]
    readme = ["  run folder: .../video_tracker_outline_sam",
              "  saved .../edgetam/A.csv, .../edgetam/B.csv and .../overlay.mp4"]
    assert output_problems(fenced("text", *readme), windows) == []
    assert output_problems(fenced("text", *readme), [line.replace("\\", "/") for line in windows]) == []
    assert output_problems(fenced("text", "  run folder: .../video_tracker_outline_ana"), windows) != []
    assert output_problems(fenced("text", "  saved .../edgetam/C.csv, .../edgetam/B.csv and .../overlay.mp4"),
                           windows) != []
    assert output_problems(fenced("text", "  saved .../edgetam/A.csv, .../edgetam/B.csv and .../overlay.mp4"),
                           [f"  saved {folder}\\edgetam\\A.csv, {folder}\\edgetam\\B.csv and {folder}\\overlay.avi"]) != []


def test_the_run_folder_check_reads_the_bullets_of_the_fallback_section():
    text = f"## {TITLES[0]}\n- `a.csv`: x\n- `edgetam/`: y\n- `--fine`: z\n`video_tracker.mp4` is no bullet\n## Next\n- `b.csv`"
    assert run_folder_names(text) == {"a.csv", "edgetam/"}


def test_the_spelling_check_finds_british_words_and_leaves_american_ones():
    assert len(BRITISH.findall("The centre of 3 millimetres, a Grey colour; we analysed it.")) == 5
    assert BRITISH.findall("The center of 3 millimeters, a gray color; two analyses; a meter; analyzed.") == []


def test_a_block_that_is_not_closed_is_an_error():
    with pytest.raises(ValueError, match="never closed"):
        code_blocks("```zsh\noutline-tracker selftest\n")
