"""The `ERROR: ...` line that convert, check, probe, synth and compare-tracks print on stderr when they stop."""

from __future__ import annotations

import sys


def report_error(error: Exception) -> None:
    """Print one error as a single `ERROR: ...` line on stderr (no traceback)."""
    print(f"ERROR: {error}", file=sys.stderr)
