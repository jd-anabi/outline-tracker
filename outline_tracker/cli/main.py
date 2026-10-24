"""Command line entry point: `outline-tracker` and its subcommands (SPEC 11).

Nothing heavy is imported here: torch, transformers and Qt are loaded only by the commands that
need them, so `--version`, `export` and `probe` start at once. The commands that read or write
videos (`convert`, `check`, `probe`, `synth`) import OpenCV only when they run. `probe` itself is in
outline_tracker/cli/probe.py; its parser is here with the others. Every other command is a module of
outline_tracker/cli/ with its parser: `convert` in convert.py, `check` in check.py, `export` in export.py,
`from-tracker` in from_tracker.py, `selftest` in selftest.py, `gui` (the window) in gui.py, and the two
hidden ones, `synth` in synth.py and `compare-tracks` in compare_tracks.py. This file adds them in the order
`--help` lists them, and holds what they share: `--version` and the way `main` starts. The one `ERROR: ...`
line is in errors.py.

`synth` is a hidden command (not listed in `--help`): it writes one of the synthetic test clips of
outline_tracker/synthetic/clips.py, for checking an installation by hand. `compare-tracks` is the other
hidden command (decision X10): it compares the Tracker-format track files of two folders, in px.
`check VIDEO --seek` is a hidden option: it also checks that jumping to a frame gives the frame the
tracker would read.

Every command returns the process exit code: 0 = success, 1 = an error or "do not use this file",
2 = a mistake in the command line itself (argparse). A problem the user can fix is one
`ERROR: ...` line on stderr, never a traceback.
"""

from __future__ import annotations

import argparse
import sys

from outline_tracker.cli import (
    check as cli_check, compare_tracks as cli_compare_tracks, convert as cli_convert, export as cli_export,
    from_tracker as cli_from_tracker, gui as cli_gui, probe as cli_probe, selftest as cli_selftest,
    synth as cli_synth,
)
from outline_tracker.cli.errors import report_error


class _Version(argparse.Action):
    """`--version`: print `outline-tracker 0.1.0 (commit abc1234)` and exit. The line is made only
    when it is asked for (`provenance.tool_version` may ask git), not whenever a command starts."""

    def __call__(self, parser, namespace, values, option_string=None):
        from outline_tracker.core.provenance import tool_version

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
    cli_gui.add_parser(commands)

    cli_convert.add_parser(commands)
    cli_check.add_parser(commands)
    cli_export.add_parser(commands)

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

    cli_from_tracker.add_parser(commands)
    cli_selftest.add_parser(commands)

    # Hidden: a subcommand added without `help=` is not listed in `outline-tracker --help`.
    cli_synth.add_parser(commands)
    cli_compare_tracks.add_parser(commands)
    return parser


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
        report_error(error)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return the process exit code (0 = success).

    `argv` is the argument list without the program name; None means `sys.argv[1:]`.
    """
    for stream in (sys.stdout, sys.stderr):  # a redirected Windows console cannot show every character: "?" then
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
