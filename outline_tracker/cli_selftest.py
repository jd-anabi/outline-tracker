"""The `selftest` command (SPEC 11, 13.4): its two options, and `selftest.selftest` behind them.

`cli.py` adds the command with `add_parser`. Nothing heavy is imported here: numpy, OpenCV, pandas
and the tracking code are loaded when the command runs, torch when it loads the model.

What it prints is what `from_tracker` prints for the made-up clip, then last week's two lines: the
verdict (`OK` or `PROBLEM`, the largest error in px in Tracker's image coordinates, and the s per
tracked frame on this computer) and the estimate in minutes for 10 s of video.
"""

from __future__ import annotations

import argparse
import sys

EXAMPLE = """\
Usage (the first run downloads the model; after that it takes about a minute):
    outline-tracker selftest
    outline-tracker selftest --device cpu

It makes a short video of one dark, shrimp-sized ellipse in a temporary folder, tracks it for 20
frames and compares the positions with where the ellipse was drawn. The last two lines are the
result: "OK: edgetam followed the test shrimp within ... pixels (should be under 3)", with the
seconds per frame on this computer, and an estimate of how long 10 s of your video will take.
Exit code 0 means OK; 1 means PROBLEM or an error.
"""


def add_parser(commands) -> argparse.ArgumentParser:
    """Add `selftest` to the subcommands of `outline-tracker` (the object that
    `ArgumentParser.add_subparsers` returns) and return its parser. No quantities, so no units."""
    parser = commands.add_parser(
        "selftest",
        help="check the installation and time the model on a made-up clip",
        description=(
            "Check the installation and time the model on this computer: track a made-up video of one "
            "shrimp-sized ellipse and compare the result with where the ellipse was drawn."
        ),
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="edgetam", choices=["edgetam", "sam2"],
                        help="the model (default: edgetam)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="cpu, mps (Apple GPU) or cuda (NVIDIA GPU); default: the fastest one that works")
    parser.set_defaults(run=run)
    return parser


def run(args: argparse.Namespace) -> int:
    """`selftest [--model M] [--device D]`: run `selftest.selftest` and print what it does.

    Returns 0 when the verdict is OK: the test shrimp was found on every frame, less than 3 px (in
    Tracker's image coordinates) from where it was drawn. Returns 1 for PROBLEM, and 1 with one
    `ERROR: ...` line on stderr when there is no verdict: the model could not be loaded, tracking
    failed, or the run was stopped with Ctrl+C.
    """
    import cv2  # imported here so that the other commands start without OpenCV

    from outline_tracker.selftest import selftest

    try:
        report = selftest(args.model, args.device)
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"ERROR: {' '.join(str(error).split())}", file=sys.stderr)  # one line, whatever the reason holds
        return 1
    return 0 if report["ok"] else 1
