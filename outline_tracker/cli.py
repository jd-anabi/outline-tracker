"""Command line entry point: `outline-tracker` and its subcommands (SPEC 11).

Nothing heavy is imported here: torch, transformers and Qt are loaded only by the commands that
need them, so `--version`, `export` and `probe` start at once. The commands that read or write
videos (`convert`, `check`, `synth`) import OpenCV only when they run.

`synth` is a hidden command (not listed in `--help`): it writes one of the synthetic test clips of
outline_tracker/synthetic.py, for checking an installation by hand. `check VIDEO --seek` is a
hidden option: it also checks that jumping to a frame gives the frame the tracker would read.

Every command returns the process exit code: 0 = success, 1 = an error or "do not use this file",
2 = a mistake in the command line itself (argparse). A problem the user can fix is one
`ERROR: ...` line on stderr, never a traceback.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from outline_tracker import __version__

CONVERT_EXAMPLE = """\
Usage (from the repository folder):
    outline-tracker convert "data/raw/groupB_2026-09-29_1325_main.MOV"

It writes data/raw/groupB_2026-09-29_1325_main_tracker.mp4 next to the original and checks
that the copy has the same number of frames and frame rate.
"""

CHECK_EXAMPLE = """\
Usage (from the repository folder):
    outline-tracker check "path/to/your video.MOV"

Prints what the file says about itself and any warnings. Exit code 1 means "do not use".
"""


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of `outline-tracker` (no quantities here, so no units or frame)."""
    parser = argparse.ArgumentParser(
        prog="outline-tracker",
        description="Track objects in videos with EdgeTAM; export positions and outline shapes.",
    )
    parser.add_argument("--version", action="version", version=f"outline-tracker {__version__}")
    commands = parser.add_subparsers(title="commands", dest="command", metavar="COMMAND")

    convert = commands.add_parser(
        "convert",
        help="make a copy of a phone video that Tracker can open",
        description=(
            "Make a copy of a phone video that Tracker can open. Tracker cannot read HEVC (H.265), "
            "the format iPhones use for 240 fps slow motion. This re-encodes the ORIGINAL video "
            "to H.264 with every frame kept, in order, with nothing added or dropped and the "
            "phone's rotation applied."
        ),
        epilog=CONVERT_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    convert.add_argument("videos", nargs="+", metavar="VIDEO", help="the original video file(s)")
    convert.set_defaults(run=run_convert)

    check = commands.add_parser(
        "check",
        help="check a video before you analyze it",
        description="Check a video before you analyze it.",
        epilog=CHECK_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check.add_argument("videos", nargs="+", metavar="VIDEO", help="the video file(s) to check")
    check.add_argument("--seek", action="store_true", help=argparse.SUPPRESS)  # hidden (decision X19)
    check.set_defaults(run=run_check)

    # Hidden: a subcommand added without `help=` is not listed in `outline-tracker --help`.
    synth = commands.add_parser(
        "synth",
        description="Write a synthetic test clip (1920 x 1080 px, 240 fps, H.264) with known contents.",
    )
    synth.add_argument("scene", choices=["dish", "closeup"],
                       help="dish: small shrimp in a dish, a contact and an LED (32.4 um/px); "
                            "closeup: one shrimp beating its antennae at 9 Hz (10 um/px)")
    synth.add_argument("out", metavar="OUT.mp4", help="the clip to write; its folder is created if needed")
    synth.add_argument("--seconds", type=float, default=2.0, metavar="S",
                       help="length of the clip in s (default: 2)")
    synth.set_defaults(run=run_synth)
    return parser


def _report_error(error: Exception) -> None:
    """Print one error as a single `ERROR: ...` line on stderr (no traceback)."""
    print(f"ERROR: {error}", file=sys.stderr)


def run_convert(args: argparse.Namespace) -> int:
    """`convert VIDEO...`: write `<stem>_tracker.mp4` (H.264, every frame kept) next to each video.

    Frame sizes are in pixels, frame rates in frames per second of file time. Goes on with the
    next video after an error; returns 1 if any video failed, else 0.
    """
    import cv2  # imported here so that the other commands start without OpenCV

    from outline_tracker import convert, video

    status = 0
    for path in args.videos:
        print(f"Converting {path} (a minute of 240 fps video takes a few minutes) ...")
        try:
            out = convert.convert_for_tracker(path)
            info = video.probe(out)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            _report_error(error)
            status = 1
            continue
        print(f"  wrote {out}: {info.n_frames} frames, {info.width} x {info.height} px, "
              f"{info.fps_container:.2f} fps in the file")
        print("  Open this copy in Tracker. Real time still comes from fps_true (your stopwatch clip).")
    return status


def run_check(args: argparse.Namespace) -> int:
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

    from outline_tracker.frame_source import check_seek
    from outline_tracker.video import check_video

    status = 0
    for path in args.videos:
        try:
            check = check_video(path)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            _report_error(error)
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
                _report_error(error)
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


def run_synth(args: argparse.Namespace) -> int:
    """`synth {dish,closeup} OUT.mp4 [--seconds S]`: write a synthetic clip and say what it shows.

    The clip is 1920 x 1080 px at 240 fps (also its fps_true) and S seconds long. The printed
    positions are in px in Tracker's convention (SPEC 3.1), frame numbers count from 0. Returns 1
    with an `ERROR:` line, and writes nothing, if S is not a positive number.
    """
    if not (math.isfinite(args.seconds) and args.seconds > 0):
        _report_error(ValueError(f"--seconds must be a positive number of seconds, not {args.seconds:g}."))
        return 1
    from outline_tracker import synthetic  # imported here: it loads OpenCV, pandas and scikit-image

    make = {"dish": synthetic.dish_scene, "closeup": synthetic.closeup_scene}[args.scene]
    scene = make(n_frames=max(int(round(args.seconds * synthetic.FPS)), 1))
    out = Path(args.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        synthetic.render(scene, out)
    except (OSError, RuntimeError, ValueError) as error:
        _report_error(error)
        return 1
    print(f"wrote {out}: {scene.n_frames} frames, {scene.size[0]} x {scene.size[1]} px, {scene.fps:g} fps")
    names = ", ".join(obj.track_id for obj in scene.objects)
    print(f"  scale {scene.mm_per_px:g} mm/px; fps_true {scene.fps:g}; objects {names}")
    if scene.dish is not None:
        print("  dish wall: center ({:.1f}, {:.1f}) px, radius {:.1f} px".format(*scene.dish))
    if scene.led is not None:
        print("  LED box u0,v0,u1,v1 = {},{},{},{} px switches on at frame {}".format(
            *scene.led.box_px, scene.led.onset_frame))
    if scene.contact is not None:
        print("  {} and {} pass {:g} px apart at frame {}".format(
            *scene.contact.track_ids, scene.contact.gap_px, scene.contact.frame))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return the process exit code (0 = success).

    `argv` is the argument list without the program name; None means `sys.argv[1:]`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
