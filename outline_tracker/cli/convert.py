"""The `convert` command (SPEC 11): make a copy of a phone video that Tracker can open.

`cli/main.py` adds the command with `add_parser`. Only argparse and the standard library are imported here:
OpenCV and the video modules are loaded when the command runs, so that the other commands start without them.
"""

from __future__ import annotations

import argparse

from outline_tracker.cli.errors import report_error

EXAMPLE = """\
Usage (from the repository folder):
    outline-tracker convert "data/raw/groupB_2026-09-29_1325_main.MOV"

It writes data/raw/groupB_2026-09-29_1325_main_tracker.mp4 next to the original and checks
that the copy has the same number of frames and frame rate.
"""


def add_parser(commands) -> argparse.ArgumentParser:
    """Add `convert` to the subcommands of `outline-tracker` (the object that
    `ArgumentParser.add_subparsers` returns) and return its parser. No quantities, so no units."""
    convert = commands.add_parser(
        "convert",
        help="make a copy of a phone video that Tracker can open",
        description=(
            "Make a copy of a phone video that Tracker can open. Tracker cannot read HEVC (H.265), "
            "the format iPhones use for 240 fps slow motion. This re-encodes the ORIGINAL video "
            "to H.264 with every frame kept, in order, with nothing added or dropped and the "
            "phone's rotation applied."
        ),
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    convert.add_argument("videos", nargs="+", metavar="VIDEO", help="the original video file(s)")
    convert.set_defaults(run=run)
    return convert


def run(args: argparse.Namespace) -> int:
    """`convert VIDEO...`: write `<stem>_tracker.mp4` (H.264, every frame kept) next to each video.

    Frame sizes are in pixels, frame rates in frames per second of file time. Goes on with the
    next video after an error; returns 1 if any video failed, else 0.
    """
    import cv2  # imported here so that the other commands start without OpenCV

    from outline_tracker.video import convert, reader as video

    status = 0
    for path in args.videos:
        print(f"Converting {path} (a minute of 240 fps video takes a few minutes) ...")
        try:
            out = convert.convert_for_tracker(path)
            info = video.probe(out)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            report_error(error)
            status = 1
            continue
        print(f"  wrote {out}: {info.n_frames} frames, {info.width} x {info.height} px, "
              f"{info.fps_container:.2f} fps in the file")
        print("  Open this copy in Tracker. Real time still comes from fps_true (your stopwatch clip).")
    return status
