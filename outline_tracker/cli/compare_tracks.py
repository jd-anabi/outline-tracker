"""The hidden `compare-tracks` command (decision X10): compare the track files of two folders, in px.

`cli/main.py` adds the command with `add_parser`. It is not listed in `--help`. It compares the Tracker-format
track files (`<id>.csv`) of two folders: for each pair of tracks, the difference in px. Only argparse and the
standard library are imported here: pandas is loaded when the command runs.
"""

from __future__ import annotations

import argparse

from outline_tracker.cli.errors import report_error


def add_parser(commands) -> argparse.ArgumentParser:
    """Add `compare-tracks` to the subcommands of `outline-tracker` (the object that
    `ArgumentParser.add_subparsers` returns) and return its parser. No quantities, so no units."""
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
    compare.set_defaults(run=run)
    return compare


def run(args: argparse.Namespace) -> int:
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
    from outline_tracker.core import tracker_io  # imported here: it loads pandas

    try:
        table = tracker_io.compare_tracks(args.new_dir, args.old_dir, by_position=args.by_position)
    except (OSError, ValueError) as error:
        report_error(error)
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
