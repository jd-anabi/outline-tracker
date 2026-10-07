"""README.md (task B6b): the fallback instructions that students follow when the app does not open.

The README is what 16 students paste commands from, so it must not promise anything the tool does not do.
These tests read it as data and hold it against the real tool:

- every `outline-tracker ...` line in a code block, and every inline `outline-tracker ...`, is parsed by the
  real parser (`cli.build_parser()`); an option or command that does not exist fails the test;
- every `--option` the README names exists in some command's help, or is last week's (read from the
  unmodified reference script) or `uv`'s own; the side-by-side table says "same" only for options that
  both last week's script and `from-tracker` have, and "new" only for options that last week's lacks;
- the example output lines of the README are shaped like what the real commands print, here run on small
  made-up clips with a stand-in model (`...` and `path/to/...` stand for any text, numbers for any number);
- the file names of the run folder are those of `outline_tracker/schema.py`; relative links resolve; no
  `#` comment in a shell block (zsh pastes `#` as a command); American spelling.

The helpers that find the problems are tested on small bad texts, so that a check that never fails would
show up here. Paths in this file are repository paths; no units or coordinates.
"""

import contextlib
import io
import re
import shlex
import shutil
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pytest
from from_tracker_helpers import two_disks

from outline_tracker import __version__, cli, fileio, schema
from outline_tracker.from_tracker import from_tracker
from outline_tracker.provenance import tool_version
from outline_tracker.segmenter.fake import ThresholdFake
from outline_tracker.selftest import selftest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
REFERENCE_SEGMENT = REPO / "tests" / "reference" / "shrimp" / "segment.py"

TITLES = ["If the app does not open", "Troubleshooting", "Getting the original video off your phone"]
SHELLS = {"zsh", "bash", "sh", "powershell", "pwsh", "shell"}  # `shell`: the same on macOS and on Windows
LANGUAGES = SHELLS | {"text", "python"}  # `text` is what a command prints
COMMANDS = ["selftest", "from-tracker", "export", "check", "convert"]  # the README must show each of them
UV_OPTIONS = {"--force", "--with"}  # of `uv`, not of this tool: `uv tool install --force`, last week's `uv run --with`
COMMIT = re.compile(r"commit (?:[0-9a-f]{7}|unknown)")
BRITISH = re.compile(r"\b(?:colours?|centres?|centimetres?|millimetres?|metres?|analys(?:ed|ing)|behaviours?|greys?|"
                     r"licences?|organis\w*|programmes?|recognis\w*|normalis\w*|labelled|travelled)\b", re.IGNORECASE)


class Block(NamedTuple):
    """A fenced code block: its language tag, its lines, and the number of its opening line."""

    language: str
    lines: list[str]
    number: int


def code_blocks(text: str) -> list[Block]:
    """The fenced code blocks of a Markdown text, in order. A block that is never closed raises."""
    blocks, current = [], None
    for number, line in enumerate(text.splitlines(), 1):
        if current is None:
            if line.startswith("```"):
                current = Block(line[3:].strip(), [], number)
        elif line.startswith("```"):
            blocks.append(current)
            current = None
        else:
            current.lines.append(line)
    if current is not None:
        raise ValueError(f"the code block that opens on line {current.number} is never closed")
    return blocks


def prose(text: str) -> str:
    """The text without its fenced code blocks (what is left is read as sentences)."""
    kept, inside = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            inside = not inside
        elif not inside:
            kept.append(line)
    return "\n".join(kept)


def headings(text: str) -> list[str]:
    """The titles of the `## ` headings, in order, outside code blocks."""
    return re.findall(r"^## (.+?)\s*$", prose(text), re.MULTILINE)


def section(text: str, title: str) -> str:
    """The sentences and bullets of the `## title` section, up to the next `## ` heading (no code blocks)."""
    parts = [part.partition("\n") for part in re.split(r"^## ", prose(text), flags=re.MULTILINE)]
    return next((body for head, _, body in parts if head.strip() == title), "")


def printed(call, *args) -> str:
    """What `call(*args)` writes to standard output and standard error; an exit with code 0 is not an error."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            call(*args)
        except SystemExit as stop:
            if stop.code not in (0, None):
                out.write(f"exit {stop.code}\n")
    return out.getvalue()


def parse_command(line: str):
    """The parsed arguments of one `outline-tracker ...` line by the real parser, or a text saying why not.
    The line is split as zsh splits it; PowerShell lines hold the same words in double quotes."""
    try:
        words = shlex.split(line)
    except ValueError as err:
        return f"cannot be split into words ({err})"
    if words[0] != "outline-tracker":
        return "does not start with the command outline-tracker"
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            return cli.build_parser().parse_args(words[1:])
        except SystemExit as stop:  # `--version` and `--help` end the parse with code 0: they are real
            if stop.code in (0, None):
                return None
    return out.getvalue().strip().splitlines()[-1]


def commands_in(text: str) -> list[tuple[str, object]]:
    """Every `outline-tracker ...` line of the shell blocks, and every inline `outline-tracker ...`, as
    (line, what `parse_command` says)."""
    lines = [line.strip() for block in code_blocks(text) if block.language in SHELLS for line in block.lines]
    lines += re.findall(r"`(outline-tracker [^`]*)`", prose(text))
    return [(line, parse_command(line)) for line in lines if line.startswith("outline-tracker")]


def command_problems(text: str) -> list[str]:
    """What is wrong with the commands of a README text: a line the real parser rejects, an untagged or
    unknown code block, a command in a block that is not a shell block (it would not be checked)."""
    problems = []
    for block in code_blocks(text):
        if block.language not in LANGUAGES:
            problems.append(f"line {block.number}: a code block must be tagged {sorted(LANGUAGES)}, not "
                            f"'{block.language}'")
        if block.language not in SHELLS:
            problems += [f"line {block.number}: a command in a '{block.language}' block: {line}" for line in block.lines
                         if re.match(r"outline-tracker [a-z-]", line.strip())]
    problems += [f"{line}: {result}" for line, result in commands_in(text) if isinstance(result, str)]
    return problems


def comment_problems(text: str) -> list[str]:
    """Lines of shell blocks that hold a `#` comment, which zsh pastes as a command or an argument."""
    return [f"line {block.number}: {line}" for block in code_blocks(text) if block.language in SHELLS
            for line in block.lines if line.lstrip().startswith("#") or re.search(r"\s#", line)]


def link_problems(text: str, folder: Path) -> list[str]:
    """Relative links `[text](target)` whose file does not exist in `folder`."""
    targets = re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", prose(text))
    local = [target.split("#")[0] for target in targets if not re.match(r"(?:https?:|mailto:|#)", target)]
    return [f"{target} does not exist" for target in local if not (folder / target).exists()]


def options_in(text: str) -> set[str]:
    """Every `--option` that a text names."""
    return set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", text))


def help_options() -> set[str]:
    """Every option of `outline-tracker` and of the commands listed in its help, from the real parser's help."""
    names = [[], *([name] for name in ["convert", "check", "export", "probe", "from-tracker", "selftest"])]
    return options_in("\n".join(printed(cli.build_parser().parse_args, [*name, "--help"]) for name in names))


def option_rows(text: str) -> list[tuple[str, set[str], set[str]]]:
    """The rows of the table of last week's and this week's command: (what you want, the options in the
    last-week cell, the options in the this-week cell). Rows have three cells."""
    rows = []
    for line in prose(text).splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and len(cells) == 3:
            rows.append((cells[0], options_in(cells[1]), options_in(cells[2])))
    return rows


def table_problems(text: str, last_week: set[str], this_week: set[str]) -> list[str]:
    """Rows of the side-by-side table that claim more than is true: an option in the last-week cell that
    last week's script lacks, one in the this-week cell that `from-tracker` lacks, and an option that is in
    the this-week cell only but that last week's script already had (it is not new)."""
    problems = []
    for what, before, now in option_rows(text):
        problems += [f"{what}: {opt} is not an option of last week's script" for opt in before - last_week]
        problems += [f"{what}: {opt} is not an option of from-tracker" for opt in now - this_week]
        problems += [f"{what}: {opt} is shown as new, but last week's script had it" for opt in (now - before) & last_week
                     if not before]
    return problems


def line_pattern(line: str) -> str:
    """A regular expression for an example output line: `...` and `path/to/...` stand for any text, a
    number for any number, a commit for any commit."""
    line = COMMIT.sub("commit X", line.strip())
    out, last = [], 0
    for found in re.finditer(r"\.\.\.|path/to/[\w./-]+|\d+(?:\.\d+)?", line):
        out.append(re.escape(line[last:found.start()]))
        out.append(r"\d+(?:\.\d+)?" if found.group()[0].isdigit() else ".+?")
        last = found.end()
    out.append(re.escape(line[last:]))
    return "".join(out)


def output_problems(text: str, real_lines: list[str]) -> list[str]:
    """Lines of the README's `text` blocks that no line printed by the real commands fits."""
    real = [COMMIT.sub("commit X", line.strip()) for line in real_lines]
    return [f"line {block.number}: {line.strip()}" for block in code_blocks(text) if block.language == "text"
            for line in block.lines if line.strip() and not any(re.fullmatch(line_pattern(line), r) for r in real)]


def run_folder_names(text: str) -> set[str]:
    """The files and folders that the README lists (as a bullet that starts with a name in backticks) in its
    section "If the app does not open"."""
    names = re.findall(r"^- `([^`]+)`", section(text, TITLES[0]), re.MULTILINE)
    return {name for name in names if re.fullmatch(r"[\w.-]+\.(?:csv|json|npz|mp4|log|txt)|[\w.-]+/", name)}


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
    last_week = options_in(REFERENCE_SEGMENT.read_text(encoding="utf-8"))
    unknown = options_in(text) - help_options() - last_week - UV_OPTIONS
    assert unknown == set()


def test_the_table_of_options_is_true_and_has_the_options_of_the_brief(text):
    last_week = set(re.findall(r'add_argument\("(--[a-z-]+)"', REFERENCE_SEGMENT.read_text(encoding="utf-8")))
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


def test_the_version_line_is_the_one_of_this_version(text):
    assert f"outline-tracker {__version__} (commit " in text


def test_it_holds_no_email_address_and_only_american_spelling(text):
    assert re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text) == []
    assert BRITISH.findall(text) == []


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
def real_lines(tmp_path_factory) -> list[str]:
    """Every line that the commands print on made-up clips: `--version`, `check`, `convert`, `from-tracker`
    (two disks, a stand-in model, overlay on), `export` of that run, and `selftest` with the stand-in. The
    stand-in is called edgetam, as the README's example is."""
    folder = tmp_path_factory.mktemp("readme")
    lines = [tool_version()]
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


def test_the_run_folder_check_reads_the_bullets_of_the_fallback_section():
    text = f"## {TITLES[0]}\n- `a.csv`: x\n- `edgetam/`: y\n- `--fine`: z\n`video_tracker.mp4` is no bullet\n## Next\n- `b.csv`"
    assert run_folder_names(text) == {"a.csv", "edgetam/"}


def test_the_spelling_check_finds_british_words_and_leaves_american_ones():
    assert len(BRITISH.findall("The centre of 3 millimetres, a Grey colour; we analysed it.")) == 5
    assert BRITISH.findall("The center of 3 millimeters, a gray color; two analyses; a meter; analyzed.") == []


def test_a_block_that_is_not_closed_is_an_error():
    with pytest.raises(ValueError, match="never closed"):
        code_blocks("```zsh\noutline-tracker selftest\n")
