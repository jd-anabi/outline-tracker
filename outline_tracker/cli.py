"""Command line entry point: `outline-tracker` and its subcommands (SPEC 11).

Nothing heavy is imported here: torch, transformers and Qt are loaded only by the commands that
need them, so `--version`, `export` and `probe` start at once. The commands that read or write
videos (`convert`, `check`, `probe`, `synth`) import OpenCV only when they run. `probe` itself is in
outline_tracker/cli_probe.py; its parser is here with the others.

`synth` is a hidden command (not listed in `--help`): it writes one of the synthetic test clips of
outline_tracker/synthetic.py, for checking an installation by hand. `compare-tracks` is the other
hidden command (decision X10): it compares the Tracker-format track files of two folders, in px.
`check VIDEO --seek` is a hidden option: it also checks that jumping to a frame gives the frame the
tracker would read.

Every command returns the process exit code: 0 = success, 1 = an error or "do not use this file",
2 = a mistake in the command line itself (argparse). A problem the user can fix is one
`ERROR: ...` line on stderr, never a traceback.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from outline_tracker import cli_probe

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


class _Version(argparse.Action):
    """`--version`: print `outline-tracker 0.1.0 (commit abc1234)` and exit. The line is made only
    when it is asked for (`provenance.tool_version` may ask git), not whenever a command starts."""

    def __call__(self, parser, namespace, values, option_string=None):
        from outline_tracker.provenance import tool_version

        print(tool_version())
        parser.exit()


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of `outline-tracker` (no quantities here, so no units or frame)."""
    parser = argparse.ArgumentParser(
        prog="outline-tracker",
        description="Track objects in videos with EdgeTAM; export positions and outline shapes.",
    )
    parser.add_argument("--version", action=_Version, nargs=0, help="show the tool's version and commit and exit")
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

    probe = commands.add_parser(
        "probe",
        help="measure the brightness inside rectangles (an LED) on every frame",
        description=(
            "Measure the mean red, green, blue and gray inside named rectangles on every frame of a video, "
            "for example to time a stimulus LED (Tracker's RGB Region). No model is loaded. Writes probes.csv."
        ),
        usage=cli_probe.USAGE,
        epilog=cli_probe.EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    probe.add_argument("source", metavar="VIDEO|SESSION.json",
                       help="the video to measure, or the session.json of a run folder")
    probe.add_argument("--rect", action="append", type=cli_probe.parse_rect, metavar="NAME:u0,v0,u1,v1",
                       help="a rectangle: a name and two opposite corners in image px; repeat it for more")
    probe.add_argument("--start", type=int, metavar="F", help="first frame to measure (default: 0)")
    probe.add_argument("--end", type=int, metavar="F",
                       help="last frame to measure, included (default: the last frame of the video)")
    probe.add_argument("--fps", type=float, metavar="F",
                       help="fps_true, the real frame rate in frames per second (default: from data/manifest.csv; "
                            "the frame rate written in the file is never used)")
    probe.add_argument("--student", metavar="NAME",
                       help="your name: probes.csv goes to <video folder>/<video stem>_outline_NAME/")
    probe.add_argument("--out", metavar="DIR", help="the folder for probes.csv, instead of the one --student gives")
    probe.set_defaults(run=run_probe)

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

    compare = commands.add_parser(  # hidden, like synth
        "compare-tracks",
        description=(
            "Compare the Tracker-format track files (<id>.csv) of two folders: for each pair of tracks, the "
            "difference in px on the frames both have. Exit code 1 unless every new track was compared "
            "and is within the limit."
        ),
    )
    compare.add_argument("new_dir", metavar="NEW_DIR", help="the folder with the new track files")
    compare.add_argument("old_dir", metavar="OLD_DIR", help="the folder with the track files to compare them with")
    compare.add_argument("--limit", type=float, default=2.0, metavar="PX",
                         help="the largest RMS difference of a pair that passes, in px (default: 2)")
    compare.add_argument("--by-position", action="store_true",
                         help="pair each new track with the old track nearest to it on the first frame the two "
                              "share (at most 15 px away), instead of the one with the same file name")
    compare.set_defaults(run=run_compare_tracks)
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


def run_probe(args: argparse.Namespace) -> int:
    """`probe VIDEO --rect NAME:u0,v0,u1,v1 ...` or `probe SESSION.json`: write probes.csv, no model.

    Rectangles are two opposite corners in image px (u to the right, v down, pixel centers at +0.5);
    `--start` and `--end` are video frame numbers counted from 0, both included; `--fps` is fps_true in
    frames per second. The work is `cli_probe.run`. Returns 0, or 1 with an `ERROR:` line when
    something is missing or wrong; nothing is written then.
    """
    import cv2  # imported here so that the other commands start without OpenCV

    try:
        return cli_probe.run(args)
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        _report_error(error)
        return 1


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


def run_compare_tracks(args: argparse.Namespace) -> int:
    """`compare-tracks NEW_DIR OLD_DIR [--limit PX] [--by-position]`: how far apart are two sets of tracks?

    Both folders hold Tracker-format files, one `<id>.csv` per track. Prints one line per new track:
    the old track it was paired with, the number of frames both have, the first and the last of them,
    the RMS and the largest difference, and the frame of the largest difference. Differences are
    distances in px in Tracker's image coordinates (SPEC 3.1); frames are the video's frame numbers;
    `--limit` is in px. The last line is `OK: worst RMS <x> px (limit <L> px)` when at least one pair
    was compared, every new track has a partner and common frames with it, and no RMS is above the
    limit; otherwise it starts with `CHECK:` and names what is wrong. A `note:` line before it names
    the old tracks that no new track was paired with; they do not change the verdict. Returns 0 for
    `OK`, else 1 (also after an `ERROR:` line for a folder or a file that cannot be read). The output
    is plain ASCII.
    """
    from outline_tracker import tracker_io  # imported here: it loads pandas

    try:
        table = tracker_io.compare_tracks(args.new_dir, args.old_dir, by_position=args.by_position)
    except (OSError, ValueError) as error:
        _report_error(error)
        return 1
    if args.by_position:
        no_partner = f"no old track within {tracker_io.PAIR_MAX_PX:g} px of it on a frame both have"
    else:
        no_partner = "OLD_DIR has no track of this name"
    rows = table.astype(object).where(table.notna(), None).to_dict("records")  # None where nothing is given
    lines, ok = _comparison_lines(rows, args.limit, no_partner)
    print("\n".join(lines))
    return 0 if ok else 1


def _ascii(name: object) -> str:
    """A track name for the console: characters outside ASCII are written as backslash escapes."""
    return str(name).encode("ascii", "backslashreplace").decode("ascii")


def _frames_text(first: int | None, last: int | None) -> str:
    """A track's frame range in words (`none` for a track with no frame)."""
    return "none" if first is None else f"{first} to {last}"


def _comparison_lines(rows: list[dict], limit_px: float, no_partner: str) -> tuple[list[str], bool]:
    """The lines `compare-tracks` prints, and whether the verdict is OK.

    `rows` are the rows of `tracker_io.compare_tracks` with None where a value is missing: frames are
    video frame numbers, `rms_px` and `max_px` are px in image coordinates, as is `limit_px`.
    `no_partner` says why a new track can be without a partner (it depends on how tracks are paired).
    """
    pairs = [row for row in rows if row["track"] is not None]
    names = [(_ascii(row["track"]), "-" if row["old_track"] is None else _ascii(row["old_track"])) for row in pairs]
    width_new = max([len("track"), *(len(new) for new, _ in names)])
    width_old = max([len("old_track"), *(len(old) for _, old in names)])

    def start(new: str, old: str, common: object) -> str:
        return f"{new:<{width_new}}  {old:<{width_old}}{common:>8}"

    lines = [start("track", "old_track", "common")
             + f"{'first':>7}{'last':>7}{'rms_px':>9}{'max_px':>9}{'at_frame':>10}"]
    problems = []
    compared = [row for row in pairs if row["n_common"] > 0]
    for row, (new, old) in zip(pairs, names):
        if row["old_track"] is None:
            lines.append(start(new, old, 0) + f"  no partner: {no_partner}")
            problems.append(f"{new}: no partner")
        elif row["n_common"] == 0:
            lines.append(start(new, old, 0) + "  no common frames: "
                         f"new {_frames_text(row['new_first'], row['new_last'])}, "
                         f"old {_frames_text(row['old_first'], row['old_last'])}")
            problems.append(f"{new}: no common frames")
        else:
            lines.append(start(new, old, row["n_common"]) + f"{row['first']:>7}{row['last']:>7}"
                         f"{row['rms_px']:>9.3f}{row['max_px']:>9.3f}{row['at_frame']:>10}")
            if not row["rms_px"] <= limit_px:  # written this way so that a NaN on either side does not pass
                problems.append(f"{new}: RMS {row['rms_px']:.3f} px is above the limit {limit_px:g} px (largest "
                                f"difference {row['max_px']:.3f} px at frame {row['at_frame']})")
    left_over = [_ascii(row["old_track"]) for row in rows if row["track"] is None]
    if left_over:
        lines.append("note: old tracks paired with no new track: " + ", ".join(left_over))
    if not compared:
        problems.append("nothing was compared" + ("" if pairs else ": NEW_DIR holds no track file (<id>.csv)"))
    if problems:
        lines.append("CHECK: " + "; ".join(problems))
        return lines, False
    worst = max(row["rms_px"] for row in compared)
    lines.append(f"OK: worst RMS {worst:.3f} px (limit {limit_px:g} px)")
    return lines, True


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
