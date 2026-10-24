"""README.md (task B6b): the fallback instructions that students follow when the app does not open.

The README is what 16 students paste commands from, so it must not promise anything the tool does not do.
These tests read it as data and hold it against the real tool:

- every `outline-tracker ...` line in a code block, and every inline `outline-tracker ...`, is parsed by the
  real parser (`cli.build_parser()`); an option or command that does not exist fails the test;
- every `--option` the README names exists in some command's help or is `uv`'s own; the side-by-side
  table says "same" only for options that both last week's script (its nine options are typed here,
  `LAST_WEEKS_OPTIONS`) and `from-tracker` have, and "new" only for options that last week's lacks;
- the example output lines of the README are shaped like what the real commands print, here run on small
  made-up clips with a stand-in model (`...` and `path/to/...` stand for any text, numbers for any number);
- every install line names a release tag, and the example version line is that release's: until the next
  release the README on `main` is the page of the tagged release (docs/ROADMAP.md, section 2, rule 1);
- the file names of the run folder are those of `outline_tracker/core/schema.py`; relative links resolve; no
  `#` comment in a shell block (zsh pastes `#` as a command); American spelling;
- the section "Quickstart" (task C8b) has ten numbered steps of one or two sentences each, in the order of
  the window's panels, with the advice the real model needs; the buttons it names are held against the
  window itself in tests/gui/test_finish.py.

The helpers that read the README and find the problems are in tests/support/readme_helpers.py. They are tested
on small bad texts, so that a check that never fails would show up: those tests are in
tests/repo/test_readme_checks.py (the tests of the Quickstart's reader are below, with the Quickstart). Paths in
this file are repository paths; no units or coordinates.
"""

import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from from_tracker_helpers import two_disks
from layout_helpers import REPO
from readme_helpers import (BRITISH, SHELLS, TITLES, code_blocks, command_problems, commands_in, comment_problems,
                            headings, help_options, install_lines, install_problems, link_problems, numbered_steps,
                            option_rows, options_in, output_problems, parse_command, printed, prose,
                            run_folder_names, section, sentences, table_problems, tag_version)

from outline_tracker import __version__
from outline_tracker.cli import main as cli
from outline_tracker.core import fileio, schema
from outline_tracker.workflows.from_tracker import from_tracker
from outline_tracker.core.provenance import tool_version
from outline_tracker.segmenter.fake import ThresholdFake
from outline_tracker.workflows.selftest import selftest

README = REPO / "README.md"

COMMANDS = ["selftest", "from-tracker", "export", "check", "convert"]  # the README must show each of them
# The options of last week's script (`python -m shrimp.segment`), typed from its nine `add_argument` lines
LAST_WEEKS_OPTIONS = {"--selftest", "--model", "--seconds", "--step", "--fps", "--out", "--device", "--no-video",
                      "--edgetam-checkpoint"}
UV_OPTIONS = {"--force", "--python", "--with"}  # of `uv`, not of this tool: `uv tool install --force --python 3.12`
# (SPEC 15: the install line names the Python version), and last week's `uv run --with`


# ---------------------------------------------------------------------------------------------
# The README as it is


@pytest.fixture(scope="module")
def text() -> str:
    return README.read_text(encoding="utf-8")


def test_the_three_sections_are_there_with_their_exact_titles_in_this_order(text):
    titles = headings(text)
    assert all(title in titles for title in TITLES), titles
    assert [titles.index(title) for title in TITLES] == sorted(titles.index(title) for title in TITLES)


def test_it_comes_in_the_order_install_selftest_fallback(text):
    install = text.index("uv tool install git+https://github.com/jd-anabi/outline-tracker")
    version = text.index("outline-tracker --version")
    check = text.index("outline-tracker selftest")
    fallback = text.index(f"## {TITLES[0]}")
    assert install < version < check < fallback


def test_it_names_what_the_tool_needs_and_where_it_runs(text):
    needs = prose(text)
    for word in ("uv", "git", "Windows 11", "Apple", "Intel"):
        assert word in needs, word
    assert "about 56 MB" in section(text, "Check the installation")


def test_every_command_in_it_is_one_the_real_parser_accepts(text):
    assert command_problems(text) == []


def test_each_command_of_the_fallback_is_shown(text):
    shown = {result.command for _, result in commands_in(text) if result is not None and not isinstance(result, str)}
    assert set(COMMANDS) <= shown, sorted(set(COMMANDS) - shown)
    assert any(line == "outline-tracker --version" for line, _ in commands_in(text))


def test_every_option_it_names_exists(text):
    # the README names no option but the tool's own and those of `uv`
    assert options_in(text) - help_options() - UV_OPTIONS == set()


def test_the_table_of_options_is_true_and_has_the_options_of_the_brief(text):
    last_week = LAST_WEEKS_OPTIONS
    this_week = options_in(printed(cli.build_parser().parse_args, ["from-tracker", "--help"]))
    assert table_problems(text, last_week, this_week) == []
    same = {opt for _, before, now in option_rows(text) for opt in before & now}
    new = {opt for _, _, now in option_rows(text) for opt in now} - last_week  # in from-tracker, not last week
    assert same >= {"--seconds", "--step", "--fps", "--device", "--out"}
    assert new >= {"--fine", "--student", "--no-overlay"}


def test_no_shell_block_holds_a_comment(text):
    assert comment_problems(text) == []


def test_every_relative_link_points_at_a_file_and_the_outputs_page_is_linked(text):
    assert link_problems(text, REPO) == []
    assert "(docs/OUTPUTS.md)" in text


def test_the_run_folder_lists_the_files_of_schema_py(text):
    # a from-tracker run writes every file of the schema but probes.csv; the Tracker files are in <model>/
    expected = {spec.name for spec in schema.FILES} - {schema.PROBES_CSV, schema.TRACKER_FILE} | {"edgetam/"}
    assert run_folder_names(text) == expected


def test_the_name_of_a_file_that_was_locked_is_the_one_the_tool_uses(text):
    locked = fileio.new_name(Path(schema.POSITIONS_CSV)).name
    assert f"`{locked}`" in section(text, "Troubleshooting")


def test_every_install_line_names_a_tag(text):
    # docs/ROADMAP.md, section 2, rule 1: no install line on `main` is without a tag until the next release
    assert install_problems(text) == []
    assert len(install_lines(text)) == 2  # the first install, and the install again with `--force`


def test_the_version_line_is_the_one_of_the_tag_that_the_install_line_names(text):
    # until the next release this page is the page of the tagged release (rule 1): its example line is that
    # release's line, not the line of the build under test
    version = tag_version(text)
    assert f"outline-tracker {version} (commit " in text
    assert f"installs version {version}" in text


def test_it_holds_no_email_address_and_only_american_spelling(text):
    assert re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text) == []
    assert BRITISH.findall(text) == []


# ---------------------------------------------------------------------------------------------
# The Quickstart (task C8b)


QUICKSTART = "Quickstart"
# What each of the ten steps is about, in the order of the window's panels: a word every step must hold.
STEP_WORDS = ["outline-tracker", "name", "fps_true", "Stick", "Circle", "Add", "Track", "flag", "Export all",
              "run folder"]


def test_the_quickstart_is_a_section_between_the_check_and_the_fallback(text):
    titles = headings(text)
    assert titles.count(QUICKSTART) == 1
    assert titles.index("Check the installation") < titles.index(QUICKSTART) < titles.index(TITLES[0])
    assert "is coming" not in text  # the window is there: the paragraph that announced it is gone


def test_the_quickstart_has_ten_numbered_steps_of_one_or_two_sentences(text):
    steps = numbered_steps(text)
    assert [number for number, _ in steps] == list(range(1, 11))
    assert [number for number, step in steps if not 1 <= len(sentences(step)) <= 2] == []


def test_the_steps_follow_the_panels(text):
    steps = dict(numbered_steps(text))
    missing = [(number, word) for number, word in enumerate(STEP_WORDS, 1) if word not in steps.get(number, "")]
    assert missing == []
    for number, panel in ((2, 1), (3, 2), (4, 3), (5, 4), (6, 6), (7, 7), (8, 8), (9, 9)):
        assert f"panel {panel}" in steps[number], (number, panel)


def test_the_quickstart_starts_the_app_with_a_command_that_parses(text):
    lines = text.splitlines()
    start, fallback = lines.index(f"## {QUICKSTART}") + 1, lines.index(f"## {TITLES[0]}") + 1
    blocks = [block for block in code_blocks(text) if start < block.number < fallback]
    assert [(block.language in SHELLS, block.lines) for block in blocks] == [(True, ["outline-tracker"])]
    parsed = parse_command("outline-tracker")  # no command: the window
    assert parsed is not None and not isinstance(parsed, str)


def test_the_quickstart_carries_what_the_real_model_needs(text):
    steps, whole = dict(numbered_steps(text)), section(text, QUICKSTART)
    assert "antenna" in steps[6] and "body" in steps[6]  # docs/VALIDATION.md 4.2: one click leaves the antennae out
    assert "oom in" in steps[4]  # "Zoom in" for the stick
    assert "egative" in whole and "empty" in whole  # docs/VALIDATION.md 2.1: a negative point near a positive one


# The buttons of panels 8 and 9 as SPEC 10.1 writes them (task C9). Where the Quickstart names one, it is in bold like
# the buttons of the other panels: what is in bold is held against the window itself in tests/gui/.
LATE_BUTTONS = ["Next flag", "Previous flag", "Re-track from here", "End track here", "Continue as new track",
                "Export all", "Open folder"]


def test_the_quickstart_names_the_buttons_of_panels_8_and_9_in_bold(text):
    steps = dict(numbered_steps(text))
    assert "**Re-track from here**" in steps[8] and "**Export all**" in steps[9]
    unmarked = re.sub(r"\*\*[^*]+\*\*", "", section(text, QUICKSTART))  # the section without what is in bold
    assert [name for name in LATE_BUTTONS if name in unmarked] == []


def test_the_check_for_bold_finds_a_button_that_is_named_plainly():
    page = "## Quickstart\n8. Click **Next flag**, then use Re-track from here.\n9. Click Export all.\n"
    unmarked = re.sub(r"\*\*[^*]+\*\*", "", section(page, QUICKSTART))
    assert [name for name in LATE_BUTTONS if name in unmarked] == ["Re-track from here", "Export all"]


@pytest.mark.parametrize("bad, found", [
    ("## Quickstart\n1. One.\n2. Two.\n4. Four.\n", [1, 2, 4]),
    ("## Quickstart\n1. One.\n## Other\n2. Two.\n", [1]),
    ("## Quickstart\n```text\n1. in a block\n```\n1. One.\n   Goes on.\n", [1]),
], ids=["a number is skipped", "a step in another section", "a number in a code block"])
def test_the_step_reader_reads_the_numbers_as_they_are_written(bad, found):
    assert [number for number, _ in numbered_steps(bad)] == found


def test_the_step_reader_joins_a_step_that_goes_on_and_counts_its_sentences():
    (number, step), = numbered_steps("## Quickstart\n1. Click **Add**. Then click.\n   It goes on. And on.\n")
    assert (number, step) == (1, "Click **Add**. Then click. It goes on. And on.") and len(sentences(step)) == 4
    assert len(sentences("Type fps_true (239.6 fps). **Stopwatch…** measures it.")) == 2
    assert len(sentences("It writes README.txt to the run folder.")) == 1


# ---------------------------------------------------------------------------------------------
# What the commands print, on small made-up clips


def write_clip(path: Path, n_frames: int = 48) -> Path:
    """A 1280 x 720 px clip at 240 frames per s of a dark disk on a light background: nothing for `check` to
    warn about."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 240, (1280, 720))
    for i in range(n_frames):
        frame = np.full((720, 1280, 3), 200, np.uint8)
        cv2.circle(frame, (20 + i, 360), 8, (40, 40, 40), -1)
        out.write(frame)
    out.release()
    return path


@pytest.fixture(scope="module")
def real_lines(tmp_path_factory, text) -> list[str]:
    """Every line that the commands print on made-up clips: `--version`, `check`, `convert`, `from-tracker`
    (two disks, a stand-in model, overlay on), `export` of that run, and `selftest` with the stand-in. The
    stand-in is called edgetam, as the README's example is.

    The version line is the tool's own (`tool_version()` gives its shape and the commit), but with the version
    of the tag that the README's install line names in place of this build's version. Until the next release
    the README on `main` is the page of the tagged release (docs/ROADMAP.md, section 2, rule 1), so its example
    is the line that release prints. A build between two releases has a development version, which is written
    with one more part (`X.Y.Z.devN`) and would fit no example line of a release."""
    folder = tmp_path_factory.mktemp("readme")
    lines = [tool_version().replace(__version__, tag_version(text))]
    clip = write_clip(folder / "video.mp4")
    for argv in (["check", str(clip)], ["convert", str(clip)]):
        lines += printed(cli.main, argv).splitlines()
    video, export = two_disks(folder, student="sam")
    video = Path(shutil.move(video, folder / "video_tracker.mp4"))
    from_tracker(video, export, model="edgetam", fps=240.0, segmenter=ThresholdFake(), overlay=True,
                 log=lines.append)
    lines += printed(cli.main, ["export", str(folder / "video_tracker_outline_sam" / "session.json")]).splitlines()
    selftest(model="edgetam", segmenter=ThresholdFake(), folder=folder / "selftest", log=lines.append)
    return [piece for line in lines for piece in line.splitlines()]


def test_the_example_output_is_what_the_commands_print(text, real_lines):
    assert output_problems(text, real_lines) == []


def test_the_readme_has_example_output_for_each_command(text):
    shown = "\n".join(line for block in code_blocks(text) if block.language == "text" for line in block.lines)
    for start in ("outline-tracker ", "OK: edgetam followed", "Estimate for 10 s", "OK: no problems found",
                  "  wrote ", "edgetam: ", "  saved ", "Exporting the run folder"):
        assert start in shown or start.strip() in shown, start
