"""The ported selftest with the real EdgeTAM (SPEC 13.4): `ok`, and a largest error under 3 px, on
`cpu` and once on the Apple GPU. Every test here is slow: it loads torch and the model.

`selftest` is called as the command calls it, so it loads the model itself (`from_tracker`'s loader)
and tracks last week's clip through `from_tracker`: one dark ellipse on a 1080p frame whose center is
at (700.5 + 0.6 f, 500.5 + 0.2 f) px in frame f, on the 20 frames 0, 2, ..., 38. The error is also
worked out here, from the run's Tracker-format file and those true centers.

Lines printed with the prefix `VALIDATION` are the numbers of docs/VALIDATION.md, section 5; show
them with `uv run pytest -m slow tests/slow/test_selftest_real.py -q -rP`. The s per frame are an upper
bound when anything else runs on the computer.

Coordinates: px in Tracker's convention (pixel centers at +0.5, SPEC 3.1); frames are video frame
numbers.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from outline_tracker.selftest import selftest

pytestmark = [pytest.mark.slow, pytest.mark.weights]

RUN_FOLDER = "selftest_tracker_outline_selftest"  # SPEC 8.1: <video stem>_outline_<student>, next to the clip


def _selftest_on(device: str, folder) -> None:
    """Run the selftest with the real EdgeTAM on `device`, in `folder`; assert the criterion of SPEC 13.4
    and print the numbers. Errors are px in Tracker's image coordinates, times are s per tracked frame."""
    lines = []
    report = selftest(model="edgetam", device=device, folder=folder, log=lines.append)

    run = folder / RUN_FOLDER
    got = pd.read_csv(run / "edgetam" / "selftest.csv", skiprows=1)
    assert got["frame"].tolist() == list(range(0, 40, 2))
    error = np.hypot(got["pixelx"] - (700.5 + 0.6 * got["frame"]), got["pixely"] - (500.5 + 0.2 * got["frame"]))
    devices = [one["device"] for one in json.loads((run / "session.json").read_text(encoding="utf-8"))["runs"]]
    print(f"VALIDATION selftest, {device}: max error {report['max_error_px']:.3f} px (from the file: "
          f"{error.max():.3f} px, mean {error.mean():.3f} px); {report['seconds_per_frame']:.2f} s per frame; "
          f"estimate one shrimp {report['minutes_one']:.1f} min, 10 shrimp {report['minutes_ten']:.1f} min; "
          f"device at the end: {', '.join(devices)}")
    print("\n".join(lines[-2:]))

    assert f"  running on: {device}" in lines  # the model was loaded here, on the device asked for
    assert devices == [device]  # and the run ended on it: no fall back to the processor
    assert np.isfinite(error).all() and error.max() < 3.0
    assert report["ok"] is True and report["max_error_px"] < 3.0
    assert report["max_error_px"] == pytest.approx(error.max(), abs=1e-6)  # the report is this file's error
    assert report["seconds_per_frame"] > 0
    assert re.fullmatch(r"\nOK: edgetam followed the test shrimp within \d\.\d pixels \(should be under 3\)\. "
                        r"\d+\.\d\d s per frame here\.", lines[-2]), lines[-2]
    assert lines[-1].startswith("Estimate for 10 s at step 2 (1,200 frames): one shrimp about ")


def test_selftest_with_the_real_model_on_cpu(tmp_path):
    _selftest_on("cpu", tmp_path)


def test_selftest_with_the_real_model_on_mps(tmp_path):
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("this computer has no Apple GPU (mps)")
    _selftest_on("mps", tmp_path)
