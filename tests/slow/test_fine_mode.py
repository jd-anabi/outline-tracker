"""Fine mode with the real model on the synthetic close-up shrimp (SPEC 13.4, 6.3, 7.7, 7.8).

The clip is `synthetic.closeup_scene()` as it stands: 1920 x 1080 px, 480 frames at 240 frames per
second (2 s), 0.010 mm per px. Its object A is a shrimp, a 47 x 20 px body with two antennae 30 px
long and 3 px wide that beat at 9 Hz. A is tracked in fine mode with one positive click on the
center of its body, on every frame (step 1), through the whole pipeline: `tracking.run_job`
chooses the window from the preview mask (SPEC 6.3), `export.export_all` writes shapes.csv. Once
on `cpu` and once on the Apple GPU (`mps`, skipped where there is none); each device is one run,
shared by its three tests (about 3.5 min on cpu and 1.5 min on mps; `-k cpu` or `-k mps` runs one).

What SPEC 13.4 asks of shapes.csv, one test each:
- `shape_ok` is 1;
- the spectrum of `solidity`, mean removed, peaks within 0.5 Hz of 9 Hz;
- the RMS difference between `solidity` and the true solidity is under 0.02.

The true solidity is the `solidity` column of the clip's ground-truth table (SPEC 13.2): the area
of the shape's analytic outline over the area of that outline's convex hull, the definition of
SPEC 7.7, at each frame's time. It is not measured on pixels. For the record, and not asserted,
the run also prints the RMS difference from the solidity of the true pixel mask as scikit-image
defines it (`regionprops`: pixels of the mask over pixels of its convex hull image).

The second and third test fail with the real model and are marked xfail (strict), with what was
measured: from one click on the body, EdgeTAM outlines the body without the antennae. Nothing was
tuned; docs/VALIDATION.md, section 4.2, has the numbers.

Lines printed with the prefix `VALIDATION` are the numbers of docs/VALIDATION.md; show them with
`uv run pytest -m slow tests/slow/test_fine_mode.py -q -rP`.

Coordinates: px in Tracker's convention (pixel centers at +0.5, SPEC 3.1), in the full frame;
frames are video frame numbers; t_s = frame / 240. Every test here is slow: it needs torch.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from pipeline_helpers import calibrated_session, expect_minutes, track_and_export
from tracking_helpers import track

from outline_tracker import synthetic

pytestmark = pytest.mark.slow

BEAT_HZ = 9.0  # the antennae's beat (SPEC 13.4)
FPS = 240.0    # frames per second of the clip, and its fps_true
N_FRAMES = 480  # 2 s, step 1

BODY_ONLY = (
    "Measured on 2026-10-07 with EdgeTAM, the same on cpu and on mps: from one positive click on the body the model "
    "outlines the body without the antennae, a convex shape, on all 480 frames (mask area 740 to 926 px, median "
    "814; the body alone is 738 px, the true mask with antennae 863 to 888 px). So the measured solidity is 0.990 "
    "to 0.999 while the true one swings between 0.507 and 0.711: the spectrum peaks at 0.56 Hz, not at 9 Hz, and "
    "the RMS difference is 0.4293 (limit 0.02). shape_ok = 1 on every frame, no frame lost. Nothing was tuned. For "
    "J (docs/VALIDATION.md 4.2): the clicks decide what the model takes as the object; with two more positive "
    "clicks, one on each antenna, the same run gave a peak at 9.00 Hz and an RMS difference of 0.0526, of which "
    "0.052 is a constant offset."
)


@pytest.fixture(scope="module")
def closeup(tmp_path_factory):
    """`closeup_scene()` written as H.264: the GroundTruth with its `path`."""
    scene = synthetic.closeup_scene()
    assert scene.size == (1920, 1080) and scene.fps == FPS and scene.n_frames == N_FRAMES
    assert scene.objects[0].track_id == "A" and scene.objects[0].shape.beat_hz == BEAT_HZ
    return synthetic.render(scene, tmp_path_factory.mktemp("closeup") / "closeup_tracker.mp4")


def _peak_hz(signal, fps: float = FPS, pad: int = 16) -> float:
    """The frequency, in Hz, at which the spectrum of `signal` minus its mean is largest. `signal`
    has one value per frame at `fps` frames per second; it is zero-padded to `pad` times its
    length, so the frequencies tried are fps / (pad x length) apart (0.03 Hz for 2 s)."""
    values = np.asarray(signal, float)
    values = values - values.mean()
    n = pad * len(values)
    return float(np.fft.rfftfreq(n, 1.0 / fps)[np.argmax(np.abs(np.fft.rfft(values, n)))])


def _pixel_mask_solidity(clip, track_id: str, frame: int) -> float:
    """Solidity of the true pixel mask of one frame as scikit-image defines it: the mask's pixels
    over the pixels of its convex hull image. No unit."""
    from skimage.measure import regionprops

    mask = clip.mask(track_id, frame)
    rows, cols = np.nonzero(mask)
    part = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return float(regionprops(part.astype(np.uint8))[0].solidity)


@pytest.fixture(scope="module", params=["cpu", "mps"])
def fine_run(request, closeup, tmp_path_factory):
    """Object A of the close-up clip tracked in fine mode on one device and exported, once per
    device. Returns shapes (the rows of A in shapes.csv), measured and true (solidity per frame,
    no unit), lost (frames without a mask) and peak_hz, rms (of measured against true)."""
    import torch

    from outline_tracker.segmenter import hf

    device = request.param
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("this computer has no Apple GPU (mps)")
    expect_minutes()
    frames = list(range(N_FRAMES))
    run_folder = tmp_path_factory.mktemp(f"fine_{device}") / "run"
    session = calibrated_session(closeup, run_folder, [track(closeup, "A", mode="fine")], step=1)
    done = track_and_export(closeup, session, run_folder, hf.HFSegmenter("edgetam", device))
    assert done.status == "complete", "\n".join(done.log)

    shapes = done.shapes[done.shapes.track_id == "A"]
    positions = done.positions[done.positions.track_id == "A"]
    assert shapes.frame.tolist() == frames and set(positions["mode"]) == {"fine"}
    truth = closeup.table[closeup.table.track_id == "A"].set_index("frame").loc[frames]
    true, measured = truth.solidity.to_numpy(), shapes.solidity.to_numpy(float)
    assert abs(_peak_hz(true) - BEAT_HZ) <= 0.5  # the ground truth itself beats at 9 Hz

    found = np.isfinite(measured)
    lost = int((positions.visible == 0).sum())
    rms = float(np.sqrt(np.mean((measured[found] - true[found]) ** 2))) if found.any() else float("nan")
    peak = _peak_hz(measured) if found.all() else float("nan")
    pixel = np.array([_pixel_mask_solidity(closeup, "A", frame) for frame in frames])
    rms_pixel = float(np.sqrt(np.mean((measured[found] - pixel[found]) ** 2))) if found.any() else float("nan")
    flags = sorted({code for cell in shapes["flags"] for code in cell.split(";") if code})
    counts = ", ".join(f"{code} {int(shapes['flags'].str.contains(code).sum())}" for code in flags) or "none"
    (stored,) = [one for one in done.session["tracks"] if one["id"] == "A"]
    area_px = shapes.area_mm2[found] / closeup.scene.mm_per_px ** 2
    body = closeup.scene.objects[0].shape
    print(f"VALIDATION fine mode, close-up shrimp ({len(frames)} frames, step 1, one click on the body, asked for "
          f"{device}, finished on {done.session['runs'][-1]['device']}): window {stored['fine_window_px']} px; "
          f"solidity peak at {peak:.2f} Hz (true signal: {_peak_hz(true):.2f} Hz); RMS difference from the true "
          f"solidity {rms:.4f} (from the pixel mask's: {rms_pixel:.4f}); measured solidity {np.nanmin(measured):.3f} "
          f"to {np.nanmax(measured):.3f}, mean {np.nanmean(measured):.3f}; true {true.min():.3f} to {true.max():.3f}, "
          f"mean {true.mean():.3f}; mask area {area_px.min():.0f} to {area_px.max():.0f} px (the body alone: "
          f"{math.pi * body.a_px * body.b_px:.0f} px; the true mask: {truth.area_px.min()} to {truth.area_px.max()} "
          f"px); shape_ok = 1 on {int((shapes.shape_ok == 1).sum())} of {len(shapes)} frames; px_along_major "
          f"{shapes.px_along_major.min():.1f} to {shapes.px_along_major.max():.1f}, cells_along_major "
          f"{shapes.cells_along_major.min():.1f} to {shapes.cells_along_major.max():.1f}; lost frames {lost}; flags: "
          f"{counts}; {done.seconds_per_frame:.2f} s per frame")
    return SimpleNamespace(shapes=shapes, measured=measured, true=true, lost=lost, peak_hz=peak, rms=rms)


def test_shape_ok_is_1_on_every_frame(fine_run):
    assert fine_run.lost == 0 and np.isfinite(fine_run.measured).all()
    assert (fine_run.shapes.shape_ok == 1).all()
    assert not fine_run.shapes["flags"].str.contains("LOWRES").any()


@pytest.mark.xfail(strict=True, reason=BODY_ONLY)
def test_solidity_spectrum_peaks_within_half_a_hz_of_9_hz(fine_run):
    assert abs(fine_run.peak_hz - BEAT_HZ) <= 0.5, f"the solidity spectrum peaks at {fine_run.peak_hz:.2f} Hz"


@pytest.mark.xfail(strict=True, reason=BODY_ONLY)
def test_solidity_is_within_0_02_rms_of_the_true_solidity(fine_run):
    assert fine_run.rms < 0.02, f"RMS difference from the true solidity: {fine_run.rms:.4f}"
