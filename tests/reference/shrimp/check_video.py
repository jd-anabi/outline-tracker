"""Check a video before you analyze it.

Usage (from the repository folder):
    uv run python -m shrimp.check_video "path/to/your video.MOV"

Prints what the file says about itself and any warnings. Exit code 1 means "do not use".
"""

from __future__ import annotations

import sys

from shrimp.video import check_video


def main(argv: list[str] | None = None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print(__doc__)
        return 2
    status = 0
    for path in paths:
        check = check_video(path)
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
    return status


if __name__ == "__main__":
    raise SystemExit(main())
