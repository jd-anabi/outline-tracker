"""The `export` command (SPEC 8, 11): write every output file of a run folder again, from session.json and
results.npz, with `export.export_all`.

`outline-tracker export SESSION.json [--overlay]` is for what changes after the tracking: a corrected stick
length, a new fps_true, a head click, a new origin of the axes. Nothing is tracked, no model is loaded and no
video is read, except for `--overlay`, which makes overlay.mp4 again by decoding the video. The run folder is
the folder that holds the session file; the outputs go there. The file must be named session.json, because
that is the name `export_all` reads (a session that was saved while session.json was locked is
session.new.json: the command names it, but does not read it).

What it prints, in this order: the run folder; one line from `export_all` for each file it removes (an older
`.new` neighbor, a track file that has no track any more); one line `wrote NAME: SIZE` per file written
(names relative to the run folder, sizes in plain units: `123 B`, `12.3 kB`, `4.5 MB`, with 1 kB = 1000 bytes);
then `NOTE:` lines (a partial session, a `session.new.json` beside the session, an overlay that was not
made) and `WARNING:` lines (the export report's: a locked file written as `<name>.new<ext>`, an overlay
skipped because its video was not found, a track left out). Notes and warnings come last, where they are
read. They do not change the exit code: 0 means the export ran to its end, even if a file went to a `.new`
name or the overlay was skipped. Anything that stops the export is one `ERROR: ...` line on stderr and exit
code 1, with nothing written (but for a file that stays locked, which can come after some were written).

The files hold what SPEC 8 says: mm in the user's axes with y up, s, rad, and px in Tracker's image
coordinates (pixel centers at +0.5); frames are the video's frame numbers. The text printed here is ASCII
apart from file names. No Qt, no torch; numpy, OpenCV and the rest are imported only when the command runs
(`cli.py` imports this module for every command).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

EXAMPLE = """\
Usage:
    outline-tracker export "path/to/the run folder/session.json"
    outline-tracker export "path/to/the run folder/session.json" --overlay

The run folder is the folder that holds session.json and results.npz. Every output file is made again from
them (positions.csv, shapes.csv, radial.csv, outlines.npz, the Tracker-format files, README.txt) and a
block is added to run.log. Use it after you correct the calibration, fps_true or a head click: nothing is
tracked again. The overlay video is made again only with --overlay, because that decodes the whole video.
"""

WARNING_PREFIX = "WARNING: "  # how `export_all` words a warning for its log; the report holds the same ones
PARTIAL = "This session is partial: its run did not finish, so the files hold only the frames tracked so far."
NEWER_SESSION = ("{new} lies next to {old}: a save of the session could not replace {old} while another program "
                 "held it open, so {new} holds the newer settings. This export used {old}. To use the newer "
                 "settings, close the other program, rename {new} to {old} and export again.")
NOT_REGENERATED = "{overlay} was not regenerated and is as it was. Add --overlay to make it again from the video."
NOT_MADE = "{overlay} was not made: add --overlay to make it from the video."


def add_parser(commands) -> None:
    """Add the `export` subcommand to the subparsers `commands` of the program's parser, with its handler
    `run`. Nothing is read or imported but argparse."""
    export = commands.add_parser(
        "export",
        help="write every output file again from the saved session and results (no tracking)",
        description=(
            "Write every output file of a run folder again from its session.json and results.npz:\n"
            "positions.csv, the Tracker-format files, shapes.csv, radial.csv, outlines.npz and README.txt,\n"
            "and a block in run.log. For example after you corrected the calibration stick, fps_true or a\n"
            "head click. Nothing is tracked and no model is loaded."
        ),
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    export.add_argument("session", metavar="SESSION.json",
                        help="the session.json of the run folder (the folder that holds it and results.npz)")
    export.add_argument("--overlay", action="store_true",
                        help="also make overlay.mp4 again, by decoding the video (it takes a while for a long "
                             "clip); without this option an overlay that is there is left as it is")
    export.set_defaults(run=run)


def size_text(n_bytes: int) -> str:
    """A file size in plain units: `999 B`, then kB, MB and GB with one decimal (`12.3 kB`), where
    1 kB = 1000 bytes. n_bytes: a number of bytes, 0 or more. A size that rounds up to 1000.0 of a
    unit is written in the next one (999,999 bytes: `1.0 MB`)."""
    if n_bytes < 1000:
        return f"{n_bytes} B"
    units = ("kB", "MB", "GB")
    value, step = n_bytes / 1000.0, 0
    while round(value, 1) >= 1000.0 and step < len(units) - 1:
        value, step = value / 1000.0, step + 1
    return f"{value:.1f} {units[step]}"


def run(args: argparse.Namespace) -> int:
    """Run `export` for parsed arguments: write every output file of the session's run folder, say what
    was written, and return the exit code.

    `args` has `session` (the path of a session.json) and `overlay` (also make overlay.mp4). Returns 0
    when the export ran to its end, also when the report holds warnings (printed after the list of files:
    a file written as `.new`, an overlay skipped for want of its video, a track left out). Returns 1 with
    one `ERROR: ...` line on stderr when something stops the export: no such file, a folder or a file with
    another name than session.json, a session of another schema version, no results.npz, a radial step
    other than 5 degrees, two track ids that give one file name, a file that stays locked. Files are
    written only after the checks pass; the error of a locked file can come after some were written. The
    files hold mm in the user's axes (y up), s, rad and px in Tracker's image coordinates, as SPEC 8 says.
    """
    import cv2  # imported here, as the other commands do, so that the command line starts without it

    try:
        return _export(Path(args.session), bool(args.overlay))
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


def _export(session_file: Path, overlay: bool) -> int:
    """`run` without the error handling: raises what `export_all` and `Session.load` raise."""
    from outline_tracker import export, fileio, schema
    from outline_tracker.session import Session

    _check_session_file(session_file)
    run_folder = Path(os.path.abspath(session_file.parent))
    session = Session.load(session_file)  # another schema version stops here, before anything is written
    notes = [PARTIAL] if not session.complete else []
    newer = fileio.new_name(session_file)
    if newer.is_file():
        notes.append(NEWER_SESSION.format(new=newer.name, old=session_file.name))
    had_overlay = (run_folder / schema.OVERLAY_MP4).is_file()

    print(f"Exporting the run folder {run_folder}")
    if overlay:
        print(f"{schema.OVERLAY_MP4} is made from the video: a long clip takes a few minutes.")
    report = export.export_all(run_folder, overlay=overlay, log=_show)
    for path in report.files:
        print(f"wrote {path.relative_to(run_folder).as_posix()}: {size_text(path.stat().st_size)}")
    if not overlay:
        notes.append((NOT_REGENERATED if had_overlay else NOT_MADE).format(overlay=schema.OVERLAY_MP4))
    for note in notes:
        print(f"NOTE: {note}")
    for warning in report.warnings:
        print(f"{WARNING_PREFIX}{warning}")
    return 0


def _show(line: str) -> None:
    """What `export_all` says while it works, printed as it comes, but for its warnings: the report holds
    the same ones, and they are printed after the list of files."""
    if not line.startswith(WARNING_PREFIX):
        print(line)


def _check_session_file(path: Path) -> None:
    """Raise FileNotFoundError or ValueError, with a message for the user, unless `path` is a file named
    session.json: the name `export_all` reads in the run folder, so another name would export another
    file than the one given."""
    from outline_tracker import schema

    if path.is_dir():
        raise ValueError(f"{path} is a folder, not a session file: give the {schema.SESSION_JSON} inside it.")
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    if path.name.lower() != schema.SESSION_JSON:
        raise ValueError(f"{path.name} is not named {schema.SESSION_JSON}: export works on the {schema.SESSION_JSON} "
                         f"of a run folder, next to its {schema.RESULTS_NPZ}. To export from this file, put it in "
                         f"such a folder under that name.")
