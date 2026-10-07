"""What the installed command `outline-tracker` starts (SPEC 11).

With no arguments at all it opens the window: it runs the `gui` command. With any argument it is
the command line of outline_tracker/cli.py, unchanged. Nothing heavy is imported here: torch and
Qt are loaded only when the window really opens (outline_tracker/gui/app.py).
"""

from __future__ import annotations

import sys

from outline_tracker import cli


def main(argv: list[str] | None = None) -> int:
    """Run `outline-tracker` and return the process exit code (0 = success).

    `argv` is the argument list without the program name; None means `sys.argv[1:]`. An empty list
    runs the `gui` command (the window, with no video); anything else is handed to `cli.main`.
    No quantities here, so no units or frame.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)
    return cli.main(arguments or ["gui"])


if __name__ == "__main__":
    raise SystemExit(main())
