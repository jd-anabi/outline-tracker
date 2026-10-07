"""The `gui` command (SPEC 10, 11): open the window, with a video or a session if one is named.

`cli.py` adds the command with `add_parser`; `outline-tracker` with no arguments runs it too
(outline_tracker/launch.py). Only argparse and the standard library are imported here. The window's
code (outline_tracker/gui/app.py, which loads torch and then Qt) is imported when the command runs,
and only after the path was checked: a file that is not there is one `ERROR: ...` line on stderr
and exit code 1, with nothing heavy loaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXAMPLE = """\
Usage:
    outline-tracker
    outline-tracker gui
    outline-tracker gui "path/to/your video_tracker.mp4"
    outline-tracker gui "path/to/the run folder/session.json"

`outline-tracker` alone opens the same window. Starting takes a few seconds: the libraries of the
model are loaded before the window shows.
"""


def add_parser(commands) -> argparse.ArgumentParser:
    """Add `gui` to the subcommands of `outline-tracker` (the object that
    `ArgumentParser.add_subparsers` returns) and return its parser. No quantities, so no units."""
    parser = commands.add_parser(
        "gui",
        help="open the window, with a video or a session if you name one",
        description="Open the window, with a video or a saved session if you name one.",
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", metavar="VIDEO|SESSION.json",
                        help="a video to track (a _tracker.mp4 file), or the session.json of a run folder")
    parser.set_defaults(run=run)
    return parser


def run(args: argparse.Namespace) -> int:
    """`gui [VIDEO | SESSION.json]`: open the window and return its exit code when it is closed.

    Returns 1 with one `ERROR: ...` line on stderr, and opens nothing, when the named file does not
    exist. No quantities here, so no units or frame.
    """
    arguments = []
    if args.path is not None:
        if not Path(args.path).is_file():
            print(f"ERROR: No such file: {args.path}", file=sys.stderr)
            return 1
        arguments.append(args.path)
    from outline_tracker.gui import app  # imported here: its `main` loads torch, then Qt

    return app.main(arguments)
