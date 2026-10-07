"""selftest: check the installation and time the model on this computer (SPEC 11, 13.4).

Last week's `shrimp.segment.selftest`, ported. What is last week's and stays: the made-up clip
(`synthetic.selftest_clip`: one shrimp-sized dark ellipse on a 1080p frame, 40 frames, tracked on
every 2nd), the limit of 3 px, the estimate for 1,200 frames with `EXTRA_PER_SHRIMP`, and the two
last lines. What changed is how the clip is tracked: through `from_tracker.from_tracker`, as a
student's own run is (coarse, no overlay, fps_true typed as 240 frames per second, so that no
manifest is looked for). The run folder is SPEC 8.1's default next to the clip,
`<folder>/selftest_tracker_outline_selftest/`, and the positions that are judged are read from its
Tracker-format file, `<model>/selftest.csv`. tests/test_port_fidelity.py lists every piece of the
function's text that differs from last week's.

Units and coordinates (SPEC 3): positions and errors are px in Tracker's image coordinates (pixel
centers at +0.5); times are s per tracked frame and estimated minutes. No Qt; torch is loaded only
by `from_tracker`, when it loads the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from outline_tracker.from_tracker import from_tracker
from outline_tracker.synthetic import selftest_clip

EXTRA_PER_SHRIMP = {"edgetam": 0.5, "sam2": 0.6, "sam2-small": 0.6}  # extra time per extra shrimp (measured)


def selftest(model="edgetam", device="auto", segmenter=None, folder=None, log=print) -> dict:
    """Check the installation and time the model on THIS computer: a made-up 1080p video of one
    shrimp-sized dark ellipse swimming 5 mm/s, tracked for 20 frames at step 2.

    Returns `ok` (found on every frame, and never 3 px or more from the true center), `max_error_px`
    (the largest distance from the true center, px in Tracker's image coordinates), `seconds_per_frame`
    (s per tracked frame) and `minutes_one`, `minutes_ten` (the estimated minutes for 1,200 frames of
    one shrimp, and of ten together). `folder` takes the clip and the run folder (default: a new
    temporary folder, which is kept); the other arguments are `from_tracker`'s. Raises RuntimeError
    when the run ended before its last frame (Ctrl+C), and whatever `from_tracker` raises."""
    import tempfile

    folder = Path(folder or tempfile.mkdtemp(prefix="shrimp-selftest-"))
    clip = selftest_clip(folder)
    video, export, tpx, tpy = clip["video"], clip["export"], clip["pixelx"], clip["pixely"]
    res = from_tracker(video, export, model=model, fps=240.0, student="selftest", device=device,
                       segmenter=segmenter, overlay=False, log=log)
    got = pd.read_csv(res.files[0], skiprows=1) if res.files else pd.DataFrame()
    if len(got) != len(tpx):  # stopped with Ctrl+C: `from_tracker` keeps the frames tracked so far
        raise RuntimeError(f"The test run ended after {len(got)} of {len(tpx)} frames, so there is no verdict. "
                           "Run the selftest again and let it finish.")
    err = np.hypot(got["pixelx"] - tpx, got["pixely"] - tpy)
    spf = res.seconds_per_frame
    extra = EXTRA_PER_SHRIMP.get(model, 0.6)
    one, ten = spf * 1200 / 60, spf * (1 + 9 * extra) * 1200 / 60
    ok = bool(np.isfinite(err).all() and np.nanmax(err) < 3.0)
    log(f"\n{'OK' if ok else 'PROBLEM'}: {model} followed the test shrimp within {np.nanmax(err):.1f} pixels "
        f"(should be under 3). {spf:.2f} s per frame here.")
    log(f"Estimate for 10 s at step 2 (1,200 frames): one shrimp about {one:.0f} min; "
        f"10 shrimp together about {ten:.0f} min.")
    return {"ok": ok, "max_error_px": float(np.nanmax(err)), "seconds_per_frame": spf, "minutes_one": one,
            "minutes_ten": ten}
