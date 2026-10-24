"""The `check` command (SPEC 11): check a video before you analyze it, with `--seek` also its frame jumps.

`cli/main.py` adds the command with `add_parser`. Only argparse and the standard library are imported here:
OpenCV and the video modules are loaded when the command runs, so that the other commands start without them.
`--seek` is a hidden option (decision X19): it also checks that jumping to a frame gives the frame the tracker
would read.
"""

from __future__ import annotations

import argparse

from outline_tracker.cli.errors import report_error

EXAMPLE = """\
Usage (from the repository folder):
    outline-tracker check "path/to/your video.MOV"

Prints what the file says about itself and any warnings. Exit code 1 means "do not use".
"""


def add_parser(commands) -> argparse.ArgumentParser:
    """Add `check` to the subcommands of `outline-tracker` (the object that
    `ArgumentParser.add_subparsers` returns) and return its parser. No quantities, so no units."""
    check = commands.add_parser(
        "check",
        help="check a video before you analyze it",
        description="Check a video before you analyze it.",
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check.add_argument("videos", nargs="+", metavar="VIDEO", help="the video file(s) to check")
    check.add_argument("--seek", action="store_true", help=argparse.SUPPRESS)  # hidden (decision X19)
    check.set_defaults(run=run)
    return check


def run(args: argparse.Namespace) -> int:
    """`check VIDEO... [--seek]`: print what each file says about itself, and any warnings.

    Frame sizes are in pixels, frame rates in frames per second of file time, durations in seconds
    of file time (not real time). Goes on with the next video after an error; returns 1 if any
    video has a warning or failed, else 0.

    With `--seek`, 20 random frames of each video are read by jumping to them (`FrameSource`) and
    compared with the same frames of the sequential decode that tracking uses. Two more lines are
    printed, the second one last: the number of gaps in the file's timestamps, and
    `seek: N of 20 frames exact`. Fewer than all exact also returns 1.
    """
    import cv2  # imported here so that the other commands start without OpenCV

    from outline_tracker.video.frame_source import check_seek
    from outline_tracker.video.reader import check_video

    status = 0
    for path in args.videos:
        try:
            check = check_video(path)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            report_error(error)
            status = 1
            continue
        info = check.info
        print(f"\n{info.path}")
        print(f"  frames delivered as {info.width} x {info.height} px, codec {info.codec}")
        print(f"  the file says: {info.fps_container:.2f} fps, {info.n_frames} frames, "
              f"{info.duration_s:.2f} s of file time")
        for note in check.notes:
            print(f"  note: {note}")
        for warning in check.warnings:
            print(f"  WARNING: {warning}")
        if check.ok:
            print("  OK: no problems found. Now measure fps_true from your stopwatch clip.")
        else:
            status = 1
        if args.seek:
            try:
                seek = check_seek(path)
            except (OSError, RuntimeError, ValueError, cv2.error) as error:
                report_error(error)
                status = 1
                continue
            if seek.gaps is None:
                print("  timestamps: no usable table; jumps decode from the start of the file (slower)")
            else:
                print(f"  timestamps: {seek.gaps} gap{'' if seek.gaps == 1 else 's'} in {seek.n_frames} frames")
            print(f"  seek: {seek.exact} of {seek.tested} frames exact")
            if seek.exact < seek.tested:
                status = 1
    return status
