"""Fine mode with the real model on the synthetic close-up shrimp (SPEC 13.4, 6.3, 7.7, 7.8).

The clip is `synthetic.closeup_scene()` as it stands: 1920 x 1080 px, 480 frames at 240 frames per
second (2 s), 0.010 mm per px. Its object A is a shrimp, a 47 x 20 px body with two antennae 30 px
long and 3 px wide that beat at 9 Hz. A is tracked in fine mode on every frame (step 1) from three
positive clicks on frame 0, on the center of its body and on the middle of each antenna, through
the whole pipeline: `tracking.run_job` chooses the window from the preview mask (SPEC 6.3),
`export.export_all` writes shapes.csv. Once on `cpu` and once on the Apple GPU (`mps`, skipped
where there is none); each device is one run, shared by its three tests (about 3.5 min on cpu and
1.5 min on mps; `-k cpu` or `-k mps` runs one).

The clicks follow decision 26 of docs/ROADMAP.md: a click on the body and on each thin part. The
fixture `clicks` works them out from the scene alone and checks them against it before the model
runs. That rule fixes where they are; they are never moved to make a test pass.

What is asked of shapes.csv, one test each:
- `shape_ok` is 1 (SPEC 13.4);
- the spectrum of `solidity`, mean removed, peaks within 0.5 Hz of 9 Hz (SPEC 13.4);
- `solidity` follows the true solidity: with d = measured - true on each frame, the RMS of
  d - mean(d) is under 0.02. mean(d) is the offset, and what is left is the variation. The limit
  is SPEC 13.4's; decision 26 applies it to the variation, not to d itself.

The true solidity is the `solidity` column of the clip's ground-truth table (SPEC 13.2): the area
of the shape's analytic outline over the area of that outline's convex hull, the definition of
SPEC 7.7, at each frame's time. It is not measured on pixels. For the record, and not asserted,
the run also prints the RMS of d itself, the offset, and the RMS difference from the solidity of
the true pixel mask as scikit-image defines it (`regionprops`: pixels of the mask over pixels of
its convex hull image).

Lines printed with the prefix `VALIDATION` are the numbers of docs/VALIDATION.md, section 4.2;
show them with `uv run pytest -m slow tests/slow/test_fine_mode.py -q -rP`.

Coordinates: px in Tracker's convention (pixel centers at +0.5, SPEC 3.1), in the full frame;
frames are video frame numbers; t_s = frame / 240. A body frame is the scene's
(outline_tracker/synthetic_shapes.py): its origin at the center of the object, xi toward the head,
eta 90 degrees counterclockwise from xi on screen, px. Every test here is slow: it needs torch.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from pipeline_helpers import calibrated_session, expect_minutes, track_and_export
from tracking_helpers import track_at

from outline_tracker import synthetic
from outline_tracker.synthetic_shapes import Ellipse

pytestmark = pytest.mark.slow

BEAT_HZ = 9.0  # the antennae's beat (SPEC 13.4)
FPS = 240.0    # frames per second of the clip, and its fps_true
N_FRAMES = 480  # 2 s, step 1
RMS_LIMIT = 0.02  # of SPEC 13.4, no unit; decision 26 applies it to the variation of the solidity

# What the scene gives for the three clicks on frame 0, (u, v) in px to 0.1 px: the center of A's
# body, then the middle of each antenna. They are also the clicks of the one trial with three
# clicks that docs/VALIDATION.md, section 4.2, records for version 0.1.0.
CLICKS_PX = ((1393.67, 465.92), (1375.3, 446.4), (1395.2, 439.2))


@pytest.fixture(scope="module")
def closeup(tmp_path_factory):
    """`closeup_scene()` written as H.264: the GroundTruth with its `path`."""
    scene = synthetic.closeup_scene()
    assert scene.size == (1920, 1080) and scene.fps == FPS and scene.n_frames == N_FRAMES
    assert scene.objects[0].track_id == "A" and scene.objects[0].shape.beat_hz == BEAT_HZ
    return synthetic.render(scene, tmp_path_factory.mktemp("closeup") / "closeup_tracker.mp4")


def _in_image(pose, xi: float, eta: float) -> tuple[float, float]:
    """Where the point (xi, eta) of an object's body frame, px, is in the image: (u, v) in px.
    `pose` = (u, v, heading) of the object in that frame: its center in px and its heading in rad,
    counterclockwise on screen. The reverse of what `synthetic._distance` does with a pixel center."""
    u, v, heading = pose
    cos, sin = math.cos(heading), math.sin(heading)
    return u + xi * cos - eta * sin, v - (xi * sin + eta * cos)


@pytest.fixture(scope="module")
def clicks(closeup):
    """The three positive clicks on frame 0, (u, v) in px: the center of A's body, then the middle
    of each antenna. Worked out from the scene's shape and pose and checked against the scene; no
    model runs here."""
    (animal,) = [obj for obj in closeup.scene.objects if obj.track_id == "A"]
    shape, pose, t_s = animal.shape, animal.path.pose(0), 0 / FPS  # the pose and the time (s) of frame 0
    # In the body frame an antenna starts at (attach_px, 0) and points at the angle +beta or -beta
    # from the head direction: its middle is half its length along it.
    beta, half = shape.beta_rad(t_s), shape.antenna_length_px / 2
    middles = [(shape.attach_px + half * math.cos(beta), side * half * math.sin(beta)) for side in (1.0, -1.0)]
    points = [_in_image(pose, xi, eta) for xi, eta in [(0.0, 0.0), *middles]]
    assert np.abs(np.subtract(points, CLICKS_PX)).max() <= 0.1

    body, on_shape = Ellipse(shape.a_px, shape.b_px), closeup.mask("A", 0)
    for xi, eta in middles:
        # on the antenna's middle line, half its width (1.5 px) inside the shape, and not on the body
        assert shape.distance(xi, eta, t_s) == pytest.approx(shape.antenna_width_px / 2)
        assert body.distance(xi, eta) < 0
    assert all(on_shape[int(v), int(u)] for u, v in points)  # each click falls in a pixel of the true mask
    return points


def _peak_hz(signal, fps: float = FPS, pad: int = 16) -> float:
    """The frequency, in Hz, at which the spectrum of `signal` minus its mean is largest. `signal`
    has one value per frame at `fps` frames per second; it is zero-padded to `pad` times its
    length, so the frequencies tried are fps / (pad x length) apart (0.03 Hz for 2 s)."""
    values = np.asarray(signal, float)
    values = values - values.mean()
    n = pad * len(values)
    return float(np.fft.rfftfreq(n, 1.0 / fps)[np.argmax(np.abs(np.fft.rfft(values, n)))])


def _rms(values) -> float:
    """The root of the mean square of `values`, in their unit; NaN if there is no value."""
    values = np.asarray(values, float)
    return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")


def _pixel_mask_solidity(clip, track_id: str, frame: int) -> float:
    """Solidity of the true pixel mask of one frame as scikit-image defines it: the mask's pixels
    over the pixels of its convex hull image. No unit."""
    from skimage.measure import regionprops

    mask = clip.mask(track_id, frame)
    rows, cols = np.nonzero(mask)
    part = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return float(regionprops(part.astype(np.uint8))[0].solidity)


@pytest.fixture(scope="module", params=["cpu", "mps"])
def fine_run(request, closeup, clicks, tmp_path_factory):
    """Object A of the close-up clip tracked in fine mode from the three clicks on one device and
    exported, once per device. Returns shapes (the rows of A in shapes.csv), measured and true
    (solidity per frame, no unit), lost (frames without a mask), peak_hz (of measured), and of
    d = measured - true: rms, offset (its mean) and variation_rms (the RMS of d - offset)."""
    import torch

    from outline_tracker.segmenter import hf

    device = request.param
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("this computer has no Apple GPU (mps)")
    expect_minutes()
    frames = list(range(N_FRAMES))
    run_folder = tmp_path_factory.mktemp(f"fine_{device}") / "run"
    session = calibrated_session(closeup, run_folder, [track_at("A", 0, clicks, [1, 1, 1], mode="fine")], step=1)
    done = track_and_export(closeup, session, run_folder, hf.HFSegmenter("edgetam", device))
    assert done.status == "complete", "\n".join(done.log)

    shapes = done.shapes[done.shapes.track_id == "A"]
    positions = done.positions[done.positions.track_id == "A"]
    assert shapes.frame.tolist() == frames and set(positions["mode"]) == {"fine"}
    truth = closeup.table[closeup.table.track_id == "A"].set_index("frame").loc[frames]
    true, measured = truth.solidity.to_numpy(), shapes.solidity.to_numpy(float)
    assert abs(_peak_hz(true) - BEAT_HZ) <= 0.5  # the ground truth itself beats at 9 Hz
    # A measured solidity that does not beat differs from the true one by a constant minus the true
    # one. Its variation RMS is then the true solidity's own RMS about its mean: about 0.0685 for
    # this clip, from the ground-truth table alone. That is over the limit, so the test of the
    # variation cannot pass on a flat signal.
    flat_rms = _rms(true - true.mean())
    assert flat_rms > RMS_LIMIT

    found = np.isfinite(measured)
    lost = int((positions.visible == 0).sum())
    difference = (measured - true)[found]
    offset = float(difference.mean()) if found.any() else float("nan")
    rms, variation_rms = _rms(difference), _rms(difference - offset)
    peak = _peak_hz(measured) if found.all() else float("nan")
    pixel = np.array([_pixel_mask_solidity(closeup, "A", frame) for frame in frames])
    rms_pixel = _rms((measured - pixel)[found])
    flags = sorted({code for cell in shapes["flags"] for code in cell.split(";") if code})
    counts = ", ".join(f"{code} {int(shapes['flags'].str.contains(code).sum())}" for code in flags) or "none"
    (stored,) = [one for one in done.session["tracks"] if one["id"] == "A"]
    area_px = shapes.area_mm2[found] / closeup.scene.mm_per_px ** 2
    body = closeup.scene.objects[0].shape
    places = ", ".join(f"({u:.1f}, {v:.1f})" for u, v in clicks)
    print(f"VALIDATION fine mode, close-up shrimp ({len(frames)} frames, step 1, three clicks on frame 0: the body "
          f"and each antenna, at {places} px, asked for {device}, finished on {done.session['runs'][-1]['device']}): "
          f"window {stored['fine_window_px']} px; solidity peak at {peak:.2f} Hz (true signal: {_peak_hz(true):.2f} "
          f"Hz); RMS difference from the true solidity {rms:.4f} (from the pixel mask's: {rms_pixel:.4f}), of that a "
          f"constant offset {offset:+.4f}; RMS once the offset is taken out {variation_rms:.4f} (limit {RMS_LIMIT}; "
          f"a signal that does not beat: {flat_rms:.4f}); measured solidity {np.nanmin(measured):.3f} to "
          f"{np.nanmax(measured):.3f}, mean {np.nanmean(measured):.3f}; true {true.min():.3f} to {true.max():.3f}, "
          f"mean {true.mean():.3f}; mask area {area_px.min():.0f} to {area_px.max():.0f} px, mean "
          f"{area_px.mean():.0f} (the body alone: {math.pi * body.a_px * body.b_px:.0f} px; the true mask: "
          f"{truth.area_px.min()} to {truth.area_px.max()} px); shape_ok = 1 on {int((shapes.shape_ok == 1).sum())} "
          f"of {len(shapes)} frames; px_along_major {shapes.px_along_major.min():.1f} to "
          f"{shapes.px_along_major.max():.1f}, cells_along_major {shapes.cells_along_major.min():.1f} to "
          f"{shapes.cells_along_major.max():.1f}; lost frames {lost}; flags: {counts}; "
          f"{done.seconds_per_frame:.2f} s per frame")
    return SimpleNamespace(shapes=shapes, measured=measured, true=true, lost=lost, peak_hz=peak, rms=rms,
                           offset=offset, variation_rms=variation_rms)


def test_shape_ok_is_1_on_every_frame(fine_run):
    assert fine_run.lost == 0 and np.isfinite(fine_run.measured).all()
    assert (fine_run.shapes.shape_ok == 1).all()
    assert not fine_run.shapes["flags"].str.contains("LOWRES").any()


def test_solidity_spectrum_peaks_within_half_a_hz_of_9_hz(fine_run):
    assert abs(fine_run.peak_hz - BEAT_HZ) <= 0.5, f"the solidity spectrum peaks at {fine_run.peak_hz:.2f} Hz"


def test_solidity_follows_the_true_solidity_within_0_02_rms_once_the_offset_is_taken_out(fine_run):
    # This test can fail: a solidity that does not beat has a variation RMS of about 0.0685 against
    # this truth, the true solidity's own RMS about its mean (worked out and asserted in `fine_run`).
    assert fine_run.lost == 0 and np.isfinite(fine_run.measured).all()
    assert fine_run.variation_rms < RMS_LIMIT, (
        f"RMS difference from the true solidity once the offset of {fine_run.offset:+.4f} is taken out: "
        f"{fine_run.variation_rms:.4f} (with the offset: {fine_run.rms:.4f})")
