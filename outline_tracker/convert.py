"""Make a copy of a phone video that Tracker can open. Ported from the course's shrimp.convert.

Tracker's video engine cannot read HEVC (H.265), the format iPhones use for 240 fps slow motion.
This re-encodes the ORIGINAL video to H.264 with every frame kept, in order, with nothing added
or dropped and the phone's rotation applied. The slow motion is not "baked in": frame n of the
copy is frame n of the original.

The command is `outline-tracker convert VIDEO...` (cli.py); the functions here are last week's,
moved over unchanged, and last week's tests of them are tests/test_convert.py. Frame counts
are frames, frame rates are frames per second of file time.

Usage (from the repository folder):
    uv run outline-tracker convert "data/raw/groupB_2026-09-29_1325_main.MOV"

It writes data/raw/groupB_2026-09-29_1325_main_tracker.mp4 next to the original and checks
that the copy has the same number of frames and frame rate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from outline_tracker import video


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def convert_for_tracker(src, dst=None, crf: int = 16) -> Path:
    """Re-encode `src` to H.264 (8-bit, every frame kept) and return the path of the copy.

    dst defaults to <src stem>_tracker.mp4 next to src. Raises RuntimeError if the copy does
    not have the same number of frames and frame rate as the original.
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"No such file: {src}")
    dst = Path(dst) if dst is not None else src.with_name(src.stem + "_tracker.mp4")
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-g", "24", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-an",
           "-movflags", "+faststart", str(dst)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg could not convert {src}:\n{done.stderr.strip()}")
    a, b = video.probe(src), video.probe(dst)
    if abs(a.n_frames - b.n_frames) > 1 or abs(a.fps_container - b.fps_container) > 0.01 * a.fps_container:
        raise RuntimeError(f"The copy differs from the original: {a.n_frames} frames at {a.fps_container:.2f} fps "
                           f"became {b.n_frames} frames at {b.fps_container:.2f} fps.")
    return dst
