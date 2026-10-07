"""Fine mode (SPEC 6.3): the window rule, the crop that follows one object, the fine runner on
synthetic clips, and the job around a fine run. What a fine run does at the frame's border, with a
window set by hand, with clicks outside its window, and when the model finds nothing is in
tests/test_tracking_fine_cases.py.

Expected values come from the rule W = clip(ceil(3 F), 96, 512) worked out by hand, from shapes of
known size (tests/analytic_shapes.py), from the scenes' stated paths, and from the ground truth of
outline_tracker/synthetic.py. `ExactFake` returns that ground truth in whatever view the runner
names, so a position that is off shows a wrong crop corner or a wrong shift back.

The close-up scene at 320 x 240 px (`closeup_clip`, 120 frames):
- A, the shrimp: its center circles (198.7, 119.8) at radius 48 px, 2.08 px per frame. Its largest
  diameter on frame 0 (antennae at 45 degrees) runs from the far side of the body's rear end to an
  antenna tip: 64.2 px, so 3 F = 192.6.
- B, the plain 47 x 20 px body: its center circles (48.4, 120.3) at radius 16.8 px, so with a
  window of 141 px its crop hangs over the frame's left border on every frame.

Coordinates: px in Tracker's convention (SPEC 3.1): u to the right, v downward, pixel (column c,
row r) has its center at (c + 0.5, r + 0.5). A box is (c0, r0, width, height) in whole px of the
full frame; a fine crop's corner may be negative. Frames are video frame numbers.
"""

import math

import analytic_shapes as shapes
import numpy as np
import pytest
from tracking_helpers import (
    FULL_SMALL,
    Recorder,
    Watched,
    assert_centered,
    center,
    make_session,
    run,
    table_truth,
    track,
    track_at,
)

from outline_tracker import tracking, video
from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.base import MaskResult
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.session import Circle, RunRecord, Session
from outline_tracker.tracking import SessionChanges
from outline_tracker.tracking_fine import crop_box, cut_crop, fine_window, fine_window_px

GRID = list(range(0, 120, 2))  # the clips have 120 frames; the sessions here take every 2nd

# A dish circle whose square holds both objects of the close-up scene: center (160, 120), R = 150,
# m = 4.5: [5.5, 314.5] x [-34.5, 274.5], so columns 5 to 314 and, clipped, rows 0 to 239.
BOTH, BOTH_BOX = Circle(center_px=[160.0, 120.0], radius_px=150.0), (5, 0, 310, 240)


# ---------------------------------------------------------------------------------------------
# The window rule: W = clip(ceil(3 F), 96, 512)


@pytest.mark.parametrize("feret_px, expected", [
    (47, 141), (47.2, 142),          # 141.0; 141.6 rounds up
    (14, 96), (32, 96), (32.1, 97),  # 42 and 96 are at most the smallest window; 96.3 rounds up to 97
    (300, 512), (170.6, 512), (170.7, 512), (1e12, 512),  # 900; 511.8 rounds up to 512; 512.1 is cut to 512
    (float("nan"), 96), (0.0, 96),   # no outline to measure: the smallest window
])
def test_fine_window_px(feret_px, expected):
    window = fine_window_px(feret_px)
    assert window == expected and type(window) is int


def test_fine_window_px_takes_the_factor_of_the_settings():
    assert fine_window_px(60, factor=2.0) == 120
    assert fine_window_px(47.2, factor=4.0) == 189  # 188.8
    assert fine_window_px(47, factor=2.0) == 96     # 94


@pytest.mark.parametrize("factor", [0, -3.0, float("nan"), float("inf"), "3", None, True])
def test_fine_window_px_refuses_a_factor_that_is_not_a_positive_number(factor):
    with pytest.raises(ValueError, match="fine_window_factor"):
        fine_window_px(47, factor=factor)


def test_the_names_of_the_brief_are_in_tracking():
    assert tracking.fine_window_px is fine_window_px and tracking.fine_window is fine_window


def _shape_result(distance):
    """The result a perfect model gives for a shape on a 300 x 300 px raster (signed distance, px)."""
    return shapes.to_result(distance, origin=(40, 25))


def test_fine_window_is_three_times_the_largest_diameter_of_the_outline():
    u, v = shapes.pixel_centers(300, 300, origin=(40, 25))
    # a disk of radius 20.1 px: F = 40.2, 3 F = 120.6
    assert fine_window(_shape_result(shapes.disk(u, v, (190.3, 170.6), 20.1))) == 121
    # the same disk with the factor 4: 160.8
    assert fine_window(_shape_result(shapes.disk(u, v, (190.3, 170.6), 20.1)), factor=4.0) == 161
    # the 47 x 20 px body, turned: F = 47, 3 F = 141 (the brief's range for it)
    assert 139 <= fine_window(_shape_result(shapes.ellipse(u, v, (190.3, 170.6), 23.5, 10.0, 0.3))) <= 144
    # a disk of radius 100 px: 3 F = 600 is cut to the largest window
    assert fine_window(_shape_result(shapes.disk(u, v, (190.3, 170.6), 100.0))) == 512
    # a disk of radius 12 px: 3 F = 72 is raised to the smallest window
    assert fine_window(_shape_result(shapes.disk(u, v, (190.3, 170.6), 12.0))) == 96


def test_fine_window_is_96_px_when_there_is_no_outline_to_measure():
    empty = MaskResult("A", (0, 0), np.zeros((0, 0), bool), np.zeros((0, 0), np.float32), None)
    assert fine_window(empty) == 96
    # a straight row of pixels without logits: its outline is a line, which has no largest diameter (NaN)
    assert fine_window(shapes.pixel_result(shapes.blocks((5, 300), (2, 3, 10, 290)))) == 96


# ---------------------------------------------------------------------------------------------
# The crop: a whole-pixel corner, and the same size wherever it lies


@pytest.mark.parametrize("center_px, window, expected", [
    ((100.2, 60.7), 96, (52, 13, 96, 96)),       # 100.2 - 48 = 52.2; 60.7 - 48 = 12.7
    ((48.4, 120.3), 141, (-22, 50, 141, 141)),   # 48.4 - 70.5 = -22.1; 120.3 - 70.5 = 49.8
    ((10.2, 5.1), 96, (-38, -43, 96, 96)),       # -37.8; -42.9
    ((310.0, 230.3), 30, (295, 215, 30, 30)),    # 295.0; 215.3
])
def test_crop_box_is_centered_and_has_a_whole_pixel_corner(center_px, window, expected):
    box = crop_box(center_px, window)
    assert box == expected
    assert all(type(value) is int for value in box)


def _numbered(rows, cols):
    """An RGB image [row, column, 3] in which every pixel has its own value: 10 row + column in
    the red channel, the row in green, the column in blue."""
    r, c = np.mgrid[0:rows, 0:cols]
    return np.stack([10 * r + c, r, c], axis=2).astype(np.uint8)


def _nearest(rgb, box):
    """What a crop shows by the rule of SPEC 6.3: every pixel of the box shows the frame's pixel
    nearest to it, so beyond the border the edge pixels repeat."""
    c0, r0, width, height = box
    rows = np.clip(np.arange(r0, r0 + height), 0, rgb.shape[0] - 1)
    cols = np.clip(np.arange(c0, c0 + width), 0, rgb.shape[1] - 1)
    return rgb[np.ix_(rows, cols)]


def test_cut_crop_inside_the_frame_is_that_part_of_the_frame():
    rgb = _numbered(6, 8)
    crop = cut_crop(rgb, (2, 1), 4)
    np.testing.assert_array_equal(crop, rgb[1:5, 2:6])
    assert crop.flags.c_contiguous and crop.dtype == np.uint8


def test_cut_crop_beyond_the_border_repeats_the_edge_pixels_and_keeps_its_size():
    rgb = _numbered(6, 8)
    crop = cut_crop(rgb, (-2, -1), 4)  # two columns left of the frame, one row above it
    assert crop.shape == (4, 4, 3)
    # written out: red = 10 row + column of the frame pixel that each crop pixel shows
    assert crop[:, :, 0].tolist() == [[0, 0, 0, 1], [0, 0, 0, 1], [10, 10, 10, 11], [20, 20, 20, 21]]
    np.testing.assert_array_equal(crop, _nearest(rgb, (-2, -1, 4, 4)))
    assert crop.flags.c_contiguous

    # over the right and the bottom border
    np.testing.assert_array_equal(cut_crop(rgb, (6, 4), 4), _nearest(rgb, (6, 4, 4, 4)))
    assert cut_crop(rgb, (6, 4), 4)[:, :, 0].tolist() == [[46, 47, 47, 47], [56, 57, 57, 57], [56, 57, 57, 57],
                                                           [56, 57, 57, 57]]
    # a window larger than the frame hangs over on all four sides
    large = cut_crop(rgb, (-3, -4), 14)
    assert large.shape == (14, 14, 3)
    np.testing.assert_array_equal(large, _nearest(rgb, (-3, -4, 14, 14)))
    assert large[0, 0].tolist() == rgb[0, 0].tolist() and large[-1, -1].tolist() == rgb[-1, -1].tolist()


@pytest.mark.parametrize("corner", [(8, 1), (-4, 1), (2, 6), (2, -4)], ids=["right", "left", "below", "above"])
def test_cut_crop_refuses_a_window_that_lies_outside_the_frame(corner):
    with pytest.raises(ValueError, match="outside"):
        cut_crop(_numbered(6, 8), corner, 4)  # 8 columns, 6 rows: not one pixel of the window is in the frame


# ---------------------------------------------------------------------------------------------
# Positions against the ground truth (SPEC 13.2), with the dish crop on and off


@pytest.mark.parametrize("circle, box", [(BOTH, BOTH_BOX), (None, FULL_SMALL)], ids=["crop on", "crop off"])
def test_fine_positions_equal_the_true_centroids(closeup_clip, tmp_path, circle, box):
    tracks = [track(closeup_clip, name, mode="fine") for name in "AB"]
    session = make_session(closeup_clip, tmp_path, tracks, circle=circle)
    segmenter = Watched(ExactFake(closeup_clip))
    status, seen = run(closeup_clip, session, tmp_path, segmenter)
    assert status == "complete" and seen.finished == ["complete"]

    # The start-frame masks come from previews on the coarse view (the dish square, or the frame),
    # with the clicks shifted into it; then each object has a session of its own.
    assert segmenter.made == 1
    assert segmenter.calls == ["preview", "preview", *["start", *["step"] * 59, "close"] * 2]
    assert segmenter.views[:2] == [(0, box[:2], box[2:])] * 2
    for name, (shape, (prompt,)) in zip("AB", segmenter.previews, strict=True):
        assert shape == (box[3], box[2], 3)
        u, v = center(closeup_clip, name, 0)
        assert (prompt.obj_id, prompt.labels) == (name, [1])
        np.testing.assert_allclose(prompt.points_px, [(u - box[0], v - box[1])], rtol=0, atol=1e-9)
    assert [[prompt.obj_id for prompt in prompts] for _, prompts in segmenter.starts] == [["A"], ["B"]]

    store = ResultsStore.load(tmp_path / "results.npz")
    saved = Session.load(tmp_path / "session.json")
    assert store.track_ids == ["A", "B"]
    for k, name in enumerate("AB"):
        window = saved.tracks[k].fine_window_px
        arrays = store.arrays(name)
        true_u, true_v, true_area = table_truth(closeup_clip, name, GRID)
        assert arrays.frames.tolist() == GRID
        np.testing.assert_allclose(arrays.u, true_u, rtol=0, atol=0.01)
        np.testing.assert_allclose(arrays.v, true_v, rtol=0, atol=0.01)
        assert arrays.area_px.tolist() == true_area.tolist()
        # W = 3 F around the object: it reaches neither the window's border nor the frame's
        assert arrays.visible.all() and not arrays.edge.any()
        assert set(arrays.mode.tolist()) == {"fine"}
        np.testing.assert_allclose(arrays.cell_px, window / 256)

        # The crop: W x W on every frame; on the start frame around the centroid of the preview
        # mask, on each later frame around the centroid of the tracked frame before it.
        views = segmenter.views[2 + 60 * k:2 + 60 * (k + 1)]
        assert [frame for frame, _, _ in views] == GRID
        centers = [(true_u[0], true_v[0]), *zip(true_u[:-1], true_v[:-1])]
        for view, at in zip(views, centers, strict=True):
            assert_centered(view, window, at)
        assert set(segmenter.shapes[60 * k:60 * (k + 1)]) == {(window, window, 3)}
        # the click, shifted into the first crop
        (prompt,) = segmenter.starts[k][1]
        u, v = center(closeup_clip, name, 0)
        (c0, r0) = views[0][1]
        np.testing.assert_allclose(prompt.points_px, [(u - c0, v - r0)], rtol=0, atol=1e-9)

    assert [(r.tracks, r.start_frame, r.mode, r.frames_done) for r in saved.runs] == [
        (["A"], 0, "fine", 60), (["B"], 0, "fine", 60)]
    assert saved.complete is True
    assert [(name, frame) for name, frame, _ in seen.results] == [(name, frame) for name in "AB" for frame in GRID]
    assert [done for done, _, _, _ in seen.progress] == list(range(1, 121))
    assert {total for _, total, _, _ in seen.progress} == {120}


def test_the_window_chosen_for_the_47_by_20_px_body_lies_between_139_and_144(closeup_clip, tmp_path):
    session = make_session(closeup_clip, tmp_path, [track(closeup_clip, name, mode="fine") for name in "AB"])
    assert [a_track.fine_window_px for a_track in session.tracks] == [None, None]  # automatic
    segmenter = Watched(ExactFake(closeup_clip))
    status, seen = run(closeup_clip, session, tmp_path, segmenter)
    assert status == "complete"

    saved = Session.load(tmp_path / "session.json")
    window_a, window_b = (a_track.fine_window_px for a_track in saved.tracks)
    assert type(window_a) is int and type(window_b) is int
    assert 139 <= window_b <= 144  # F = 47 px, the body's length: 3 F = 141
    assert 190 <= window_a <= 195  # F = 64.2 px (see the module text): 3 F = 192.6
    # the caller's session holds them too, and they are what the model was shown, on every frame
    assert [a_track.fine_window_px for a_track in session.tracks] == [window_a, window_b]
    assert set(segmenter.shapes[:60]) == {(window_a, window_a, 3)}
    assert set(segmenter.shapes[60:]) == {(window_b, window_b, 3)}
    assert {size for _, _, size in segmenter.views[2:62]} == {(window_a, window_a)}
    assert any(f"{window_b} x {window_b}" in line for line in seen.log)  # the log says what the model saw


class Keeping(Watched):
    """Keeps a copy of every image it is given, and whether its memory was in one piece."""

    def __init__(self, inner):
        super().__init__(inner)
        self.images, self.contiguous = [], []

    def _see(self, image):
        self.images.append(image.copy())
        self.contiguous.append(bool(image.flags.c_contiguous))

    def start(self, image, prompts):
        self._see(image)
        return super().start(image, prompts)

    def step(self, image):
        self._see(image)
        return super().step(image)


def test_the_model_is_given_the_window_cut_from_the_frame_with_the_edge_pixels_repeated(closeup_clip, tmp_path):
    session = make_session(closeup_clip, tmp_path, [track(closeup_clip, "B", mode="fine")])
    segmenter = Keeping(ExactFake(closeup_clip))
    status, _ = run(closeup_clip, session, tmp_path, segmenter)
    assert status == "complete"

    views = segmenter.views[1:]
    window = session.tracks[0].fine_window_px
    # B's center is never right of u = 48.4 + 16.8 = 65.2, and half the window is at least 69.5 px
    assert len(views) == 60 and max(c0 for _, (c0, _), _ in views) < 0
    for (frame, rgb), (told, offset, size), image in zip(video.iter_rgb_frames(closeup_clip.path, GRID), views,
                                                         segmenter.images, strict=True):
        assert told == frame and size == (window, window)
        np.testing.assert_array_equal(image, _nearest(rgb, (*offset, *size)))
    assert all(segmenter.contiguous)


def test_a_fine_object_is_not_also_tracked_in_coarse_mode(closeup_clip, tmp_path):
    session = make_session(closeup_clip, tmp_path, [track(closeup_clip, "A", mode="fine"), track(closeup_clip, "B")])
    segmenter = Watched(ExactFake(closeup_clip))
    status, seen = run(closeup_clip, session, tmp_path, segmenter, record_session=True)
    assert status == "complete"

    # one coarse session with B alone on the whole frame, then A's own session on its window
    ((_, window),) = seen.changes[2].fine_windows
    assert 190 <= window <= 195
    assert [[prompt.obj_id for prompt in prompts] for _, prompts in segmenter.starts] == [["B"], ["A"]]
    assert [shape for shape, _ in segmenter.starts] == [(240, 320, 3), (window, window, 3)]
    assert segmenter.calls == ["preview", *["start", *["step"] * 59, "close"] * 2]
    assert [(name, frame) for name, frame, _ in seen.results] == [(name, frame) for name in "BA" for frame in GRID]

    store = ResultsStore.load(tmp_path / "results.npz")
    a, b = store.arrays("A"), store.arrays("B")
    assert a.frames.tolist() == b.frames.tolist() == GRID
    assert set(a.mode.tolist()) == {"fine"} and set(b.mode.tolist()) == {"coarse"}
    np.testing.assert_allclose(a.cell_px, window / 256)
    np.testing.assert_allclose(b.cell_px, 320 / 256)
    for name, arrays in (("A", a), ("B", b)):
        true_u, true_v, _ = table_truth(closeup_clip, name, GRID)
        np.testing.assert_allclose(arrays.u, true_u, rtol=0, atol=0.01)
        np.testing.assert_allclose(arrays.v, true_v, rtol=0, atol=0.01)

    # the session's writer is told: B's run, then A's; the window chosen for A goes with the start of its run
    assert [(change.complete, change.run_index, change.run.tracks, change.run.mode, change.run.frames_done,
             change.fine_windows) for change in seen.changes] == [
        (False, 0, ["B"], "coarse", 0, ()), (False, 0, ["B"], "coarse", 60, ()),
        (False, 1, ["A"], "fine", 0, (("A", window),)), (True, 1, ["A"], "fine", 60, ())]
    assert math.isclose(seen.progress[-1][3], 0.0) and seen.progress[-1][:2] == (120, 120)


# ---------------------------------------------------------------------------------------------
# The job around a fine run


def test_a_fine_run_that_starts_later_takes_its_window_and_center_from_that_frame(closeup_clip, tmp_path):
    tracks = [track(closeup_clip, "B"), track(closeup_clip, "A", frame=60, mode="fine")]
    session = make_session(closeup_clip, tmp_path, tracks)
    segmenter = Watched(ExactFake(closeup_clip))
    status, _ = run(closeup_clip, session, tmp_path, segmenter)
    assert status == "complete"

    later = list(range(60, 120, 2))
    # the preview of frame 60 comes before B's coarse run from frame 0: no run is under way then
    assert segmenter.calls == ["preview", "start", *["step"] * 59, "close", "start", *["step"] * 29, "close"]
    assert segmenter.views[0] == (60, (0, 0), (320, 240))
    # On frame 60 (t = 0.25 s) the antennae stand at 45 + 30 sin(2 pi 9 t) = 75 degrees from the head
    # direction: their ends are 2 x 30 sin(75 degrees) = 57.96 px apart, plus 1.5 px of thickness at each
    # end. F = 60.96 px from tip to tip, 3 F = 182.9. (On frame 0 it would be 192.6.)
    window = session.tracks[1].fine_window_px
    assert 180 <= window <= 185
    true_u, true_v, _ = table_truth(closeup_clip, "A", later)
    views = segmenter.views[61:]
    assert [frame for frame, _, _ in views] == later
    assert_centered(views[0], window, (true_u[0], true_v[0]))  # around A where it is on frame 60
    arrays = ResultsStore.load(tmp_path / "results.npz").arrays("A")
    assert arrays.frames.tolist() == later
    np.testing.assert_allclose(arrays.u, true_u, rtol=0, atol=0.01)
    np.testing.assert_allclose(arrays.v, true_v, rtol=0, atol=0.01)
    saved = Session.load(tmp_path / "session.json")
    assert [(r.tracks, r.start_frame, r.mode, r.frames_done) for r in saved.runs] == [
        (["B"], 0, "coarse", 60), (["A"], 60, "fine", 30)]
    assert saved.complete is True


def test_cancel_during_a_fine_run_keeps_its_frames_and_the_window(closeup_clip, tmp_path):
    session = make_session(closeup_clip, tmp_path, [track(closeup_clip, "B", mode="fine")])
    segmenter = Watched(ExactFake(closeup_clip))
    status, seen = run(closeup_clip, session, tmp_path, segmenter, Recorder(cancel_after_frame=10))
    assert status == "cancelled" and seen.finished == ["cancelled"]
    assert segmenter.calls == ["preview", "start", *["step"] * 5, "close"]

    kept = [0, 2, 4, 6, 8, 10]
    arrays = ResultsStore.load(tmp_path / "results.npz").arrays("B")
    assert arrays.frames.tolist() == kept
    true_u, true_v, _ = table_truth(closeup_clip, "B", kept)
    np.testing.assert_allclose(arrays.u, true_u, rtol=0, atol=0.01)
    np.testing.assert_allclose(arrays.v, true_v, rtol=0, atol=0.01)
    saved = Session.load(tmp_path / "session.json")
    assert saved.complete is False and saved.runs[0].frames_done == 6
    assert 139 <= saved.tracks[0].fine_window_px <= 144  # so that a later run of B sees it at the same scale


def test_a_chosen_window_is_stored_in_its_track_only():
    session = Session(tracks=[track_at("A", 0, [(5.0, 5.0)], [1], mode="fine"),
                              track_at("B", 0, [(9.0, 9.0)], [1], mode="fine")])
    record = RunRecord(tracks=["B"], start_frame=0, mode="fine")
    changes = SessionChanges(complete=False, run_index=0, run=record, fine_windows=(("B", 141), ("Z", 200)))
    changes.apply(session)  # Z: a track the session no longer has
    assert [(a_track.id, a_track.fine_window_px) for a_track in session.tracks] == [("A", None), ("B", 141)]
    assert session.runs == [record] and session.complete is False
