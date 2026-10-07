"""The `from-tracker` command (SPEC 11, decision X19): its options, and `from_tracker` behind them.

`cli.py` adds the command with `add_parser`. Nothing heavy is imported here: OpenCV, pandas and the
tracking code are loaded when the command runs, torch when it loads the model.

Units: `--seconds` is s of real time, `--fps` is fps_true in frames per second, `--step` counts
video frames. Coordinates come from the Tracker export (px with pixel centers at +0.5, mm in
Tracker's axes with y up).
"""

from __future__ import annotations

import argparse
import sys

EXAMPLE = """\
Usage (the first run downloads the model, about 56 MB for edgetam):
    outline-tracker from-tracker "path/to/your video_tracker.mp4" "data/tracks/your video/ana/extra/start.csv"
    outline-tracker from-tracker VIDEO EXPORT --seconds 2 --fps 239.6 --fine A,C

VIDEO is the ..._tracker.mp4 copy you opened in Tracker, so that frame numbers and pixels match.
EXPORT is a file exported from Tracker (File > Export > Data) with the columns frame, x, y, pixelx,
pixely: one track, or one point mass per object, each marked once on the SAME frame and exported
together. The files go to a new folder next to the video, <video name>_outline_<your name>/:
positions.csv, shapes.csv, the Tracker-format files in edgetam/ (one per object, as last week),
overlay.mp4, run.log and the rest. Ctrl+C stops the run and saves what was tracked so far.
"""


def _names(text: str) -> list[str]:
    """`--fine A, mass B` as a list of names: split at the commas, spaces around a name dropped."""
    return [name.strip() for name in text.split(",") if name.strip()]


def add_parser(commands) -> argparse.ArgumentParser:
    """Add `from-tracker` to the subcommands of `outline-tracker` (the object that
    `ArgumentParser.add_subparsers` returns) and return its parser. `--seconds` is in s, `--fps` in
    frames per second, `--step` in video frames."""
    parser = commands.add_parser(
        "from-tracker",
        help="track the objects marked in a Tracker export (last week's way to start, this week's files)",
        description=(
            "Track the objects marked in a file exported from Tracker, with EdgeTAM or SAM 2. The calibration "
            "(pixels to mm) and the start positions come from the export, as last week. Writes positions, "
            "outline shapes, Tracker-format files and an overlay video into one run folder."
        ),
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", metavar="VIDEO", help="the ..._tracker.mp4 video you opened in Tracker")
    parser.add_argument("export", metavar="EXPORT",
                        help="file exported from Tracker with the columns frame, x, y, pixelx, pixely")
    parser.add_argument("--fine", type=_names, default=[], metavar="IDS",
                        help="names of the objects to track in fine mode (a close-up window that follows the "
                             "object), as they are in the export, separated by commas: --fine A,C")
    parser.add_argument("--step", type=int, metavar="K",
                        help="use every K-th frame (default: the Tracker track's step, or 2)")
    parser.add_argument("--seconds", type=float, metavar="S",
                        help="how long to track, in s (default: the length of the Tracker track, or 10 s)")
    parser.add_argument("--fps", type=float, metavar="F",
                        help="fps_true, the real frame rate in frames per second (default: from data/manifest.csv, "
                             "else from the t column of the export)")
    parser.add_argument("--out", metavar="DIR",
                        help="the run folder (default: <video folder>/<video name>_outline_<your name>/)")
    parser.add_argument("--student", metavar="NAME",
                        help="your name, for the run folder (default: the name of the folder the export is in)")
    parser.add_argument("--model", default="edgetam", choices=["edgetam", "sam2"],
                        help="the model (default: edgetam); the Tracker-format files go into a folder of this name")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="cpu, mps (Apple GPU) or cuda (NVIDIA GPU); default: the fastest one that works")
    parser.add_argument("--no-overlay", "--no-video", dest="no_overlay", action="store_true",
                        help="do not write overlay.mp4 (--no-video is last week's spelling)")
    parser.set_defaults(run=run)
    return parser


def options(args: argparse.Namespace) -> dict:
    """The keyword arguments of `from_tracker.from_tracker` for parsed arguments: `seconds` in s,
    `fps` in frames per second, `step` in video frames, `fine_ids` a list of names."""
    return dict(fine_ids=args.fine, step=args.step, seconds=args.seconds, fps=args.fps, out=args.out,
                student=args.student, model=args.model, device=args.device, overlay=not args.no_overlay)


def run(args: argparse.Namespace) -> int:
    """`from-tracker VIDEO EXPORT [options]`: run `from_tracker` and print what it does.

    Returns 0 when the files were written (also after Ctrl+C, with the frames tracked so far), or 1
    with one `ERROR: ...` line on stderr when something is missing or wrong. Positions in the files
    are px in image coordinates and mm in Tracker's axes, times are s (see `from_tracker`).
    """
    import cv2  # imported here so that the other commands start without OpenCV

    from outline_tracker.from_tracker import from_tracker

    try:
        from_tracker(args.video, args.export, **options(args))
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0
