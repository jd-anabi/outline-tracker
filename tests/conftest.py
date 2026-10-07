"""Shared test helpers (provided by the course; you do not need to change this file)."""

import numpy as np
import pandas as pd
import pytest


def straight_track(track_id="A", n_frames=240, fps=240.0, vx_mm_s=3.0, vy_mm_s=4.0, x0=1.0, y0=-2.0, start_frame=0):
    """A shrimp swimming in a straight line at constant velocity: speed |(vx, vy)|."""
    frame = np.arange(start_frame, start_frame + n_frames)
    t = frame / fps
    return pd.DataFrame({"track_id": track_id, "frame": frame, "t_s": t,
                         "x_mm": x0 + vx_mm_s * (t - t[0]), "y_mm": y0 + vy_mm_s * (t - t[0])})


def java_sci(v):
    """A number the way Tracker writes it with Number Format "Full Precision" (Java's 0.000000E0)."""
    if v == 0:
        return "0.000000E0"
    mantissa, exponent = f"{v:.6E}".split("E")
    return f"{mantissa}E{int(exponent)}"


def tracker_text(track, fps_tracker=240.0, name="mass A", sep=",", clip_start_frame=None):
    """The text Tracker writes for one point mass (File > Export > Data, columns frame, x, y).

    Tracker's t is 0 at the clip's start frame: t = (frame - clip_start_frame) / fps_tracker,
    where fps_tracker is the frame rate in Tracker's Clip Settings.
    """
    start = int(track["frame"].iloc[0]) if clip_start_frame is None else clip_start_frame
    lines = [sep + name + sep * 3, sep.join(["t", "frame", "x", "y"]) + sep]
    for f, x, y in zip(track["frame"], track["x_mm"], track["y_mm"]):
        cells = [(f - start) / fps_tracker, f, x, y]
        lines.append(sep.join(java_sci(float(c)) for c in cells) + sep)
    return "\n".join(lines) + "\n"


@pytest.fixture
def tracker_file(tmp_path):
    """Return a function that writes text to a file in a temporary folder and returns its path."""

    def write(name, text, folder="tracks"):
        d = tmp_path / folder
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.write_text(text)
        return p

    return write
