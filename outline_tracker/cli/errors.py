"""The one `ERROR: ...` line that the commands print on stderr when they stop (SPEC 11)."""

from __future__ import annotations

import sys


def report_error(error: Exception) -> None:
    """Print one error as a single `ERROR: ...` line on stderr (no traceback)."""
    print(f"ERROR: {error}", file=sys.stderr)
