"""The reader of README.md for the tests: its code blocks, sections, steps, commands and install lines, and the
checks that find what is wrong in a README text.

The README is what 16 students paste commands from, so it must not promise anything the tool does not do.
Each `*_problems` function takes the text of a README, with what it is held against where it needs that, and
returns one line for each thing that is wrong; the same function runs on the real README and on small bad texts
that show it finds something. The commands are parsed by the real parser (`cli.build_parser()`), the options are
read from the real help, and the example output lines are compared with lines that the real commands printed
(`output_problems`).

tests/repo/test_readme.py holds the real README against the tool with these functions;
tests/repo/test_readme_checks.py runs the checks on made-up texts. Paths in this module are repository paths;
no units or coordinates.
"""

import contextlib
import io
import re
import shlex
from pathlib import Path
from typing import NamedTuple

from layout_helpers import dead_links

from outline_tracker.cli import main as cli

TITLES = ["If the app does not open", "Troubleshooting", "Getting the original video off your phone"]
SHELLS = {"zsh", "bash", "sh", "powershell", "pwsh", "shell"}  # `shell`: the same on macOS and on Windows
LANGUAGES = SHELLS | {"text", "python"}  # `text` is what a command prints
ADDRESS = "github.com/jd-anabi/outline-tracker"  # the repository, as an install line names it
TAG = re.compile(re.escape(ADDRESS) + r"@refs/tags/v(\d+\.\d+\.\d+)(?![\w.+-])")  # a release tag: `vX.Y.Z`
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


def numbered_steps(text: str, title: str = "Quickstart") -> list[tuple[int, str]]:
    """The numbered steps of the `## title` section, in order: (the number written, the step's text). A step
    is a line that starts with a number and a full stop; lines that follow it without one belong to it."""
    steps: list[tuple[int, str]] = []
    for line in section(text, title).splitlines():
        start = re.match(r"(\d+)\. +(.*)", line)
        if start:
            steps.append((int(start.group(1)), start.group(2).strip()))
        elif steps and line.startswith(" ") and line.strip():
            steps[-1] = (steps[-1][0], f"{steps[-1][1]} {line.strip()}")
    return steps


def sentences(step: str) -> list[str]:
    """The sentences of a step: it is cut after a full stop, a question mark or a colon-free end mark that a
    blank and a capital letter (or a bold word) follow. "fps_true (239.6)" and "README.txt to" stay whole."""
    return [part for part in re.split(r"(?<=[.?!])\s+(?=[A-Z*])", step.strip()) if part]


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
    """Relative links `[text](target)` whose file does not exist in `folder`. The scan is `dead_links` of
    tests/support/layout_helpers.py, which tests/repo/test_layout.py runs over every tracked document."""
    return [f"{target} does not exist" for _, target in dead_links(text, folder)]


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
    """Lines of the README's `text` blocks that no line printed by the real commands fits. The README writes
    every path with `/`; Windows prints `\\`, so the real lines are read with `/` (nothing else is relaxed)."""
    real = [COMMIT.sub("commit X", line.strip().replace("\\", "/")) for line in real_lines]
    return [f"line {block.number}: {line.strip()}" for block in code_blocks(text) if block.language == "text"
            for line in block.lines if line.strip() and not any(re.fullmatch(line_pattern(line), r) for r in real)]


def run_folder_names(text: str) -> set[str]:
    """The files and folders that the README lists (as a bullet that starts with a name in backticks) in its
    section "If the app does not open"."""
    names = re.findall(r"^- `([^`]+)`", section(text, TITLES[0]), re.MULTILINE)
    return {name for name in names if re.fullmatch(r"[\w.-]+\.(?:csv|json|npz|mp4|log|txt)|[\w.-]+/", name)}


def install_lines(text: str) -> list[tuple[str, str | None]]:
    """The lines of the shell blocks that install the tool from its repository (they hold `uv tool install` and
    the repository's address), in order, each with the version of the tag it names: "0.1.0" for
    `@refs/tags/v0.1.0` right after the address, None for a line that names no such tag."""
    lines = [line.strip() for block in code_blocks(text) if block.language in SHELLS for line in block.lines
             if "uv tool install" in line and ADDRESS in line]
    return [(line, found.group(1) if found else None) for line, found in zip(lines, map(TAG.search, lines))]


def tag_version(text: str) -> str:
    """The version of the tag that the first install line with a tag names. A text without one raises."""
    versions = [version for _, version in install_lines(text) if version is not None]
    if not versions:
        raise ValueError("no install line names a tag")
    return versions[0]


def install_problems(text: str) -> list[str]:
    """Install lines that name no tag, and install lines that name another tag than the first one."""
    lines = install_lines(text)
    first = next((version for _, version in lines if version is not None), None)
    return ([f"no `@refs/tags/vX.Y.Z` after the address: {line}" for line, version in lines if version is None]
            + [f"another tag than v{first}: {line}" for line, version in lines if version not in (None, first)])
