"""Fine mode (SPEC 6.3) where it gets hard: a mask that touches the window, a window set by hand,
the settings, frames on which the model finds nothing (review focus 4), and clicks outside the
window. The window rule and the positions are in tests/test_tracking_fine.py, the frame's border in
tests/test_tracking_fine_border.py.

Expected values come from the scenes' stated paths, worked out in the comments, and from the
ground truth of outline_tracker/synthetic.py: what a model can see of an object in a window is the
part of its true mask inside that window and inside the frame (`truth_in_box`).

Coordinates: px in Tracker's convention (SPEC 3.1): u to the right, v downward, pixel (column c,
row r) has its center at (c + 0.5, r + 0.5). A box is (c0, r0, width, height) in whole px of the
full frame; a fine crop's corner may be negative. Frames are video frame numbers.
"""

import numpy as np
import pytest
from tracking_helpers import (
    Watched,
    assert_centered,
    center,
    make_session,
    run,
    table_truth,
    track,
    track_at,
    truth_in_box,
)

from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.base import MaskResult
from outline_tracker.segmenter.fake import ExactFake, ThresholdFake
from outline_tracker.session import Session

GRID = list(range(0, 120, 2))  # the clips have 120 frames; the sessions here take every 2nd


# ---------------------------------------------------------------------------------------------
# The mask at the window's border; a window set by hand


def test_edge_when_the_mask_touches_the_crop_border_and_a_window_set_by_hand_is_used_as_it_is(closeup_clip,
                                                                                               tmp_path):
    # B is 47 px long and 20 px wide. Whichever way it is turned, it is at least 36 px across in u
    # or in v (at 45 degrees: 2 sqrt((23.5^2 + 10^2) / 2) = 36.1), so it never fits a 30 px window.
    b_track = track(closeup_clip, "B", mode="fine")
    b_track.fine_window_px = 30  # below the smallest window the rule would choose: the user's choice stands
    session = make_session(closeup_clip, tmp_path, [b_track])
    segmenter = Watched(ExactFake(closeup_clip))
    status, seen = run(closeup_clip, session, tmp_path, segmenter, record_session=True)
    assert status == "complete"
    assert all(change.fine_windows == () for change in seen.changes)  # nothing was chosen, nothing to store
    assert set(segmenter.shapes) == {(30, 30, 3)}

    arrays = ResultsStore.load(tmp_path / "results.npz").arrays("B")
    views = segmenter.views[1:]
    assert arrays.frames.tolist() == [frame for frame, _, _ in views] == GRID
    # what is measured on a frame is the part of the body inside that frame's window
    part_u, part_v, part_area, at_border = (np.array(column) for column in zip(*(
        truth_in_box(closeup_clip, "B", frame, (*offset, *size)) for frame, offset, size in views)))
    assert at_border.all() and arrays.edge.all() and arrays.visible.all()
    np.testing.assert_allclose(arrays.u, part_u, rtol=0, atol=0.01)
    np.testing.assert_allclose(arrays.v, part_v, rtol=0, atol=0.01)
    assert arrays.area_px.tolist() == part_area.tolist()
    np.testing.assert_allclose(arrays.cell_px, 30 / 256)
    # the first window lies around the whole body's centroid (the preview on the frame), each later
    # one around the centroid of the part seen on the frame before
    whole_u, whole_v, whole_area = table_truth(closeup_clip, "B", [0])
    assert part_area[0] < whole_area[0]
    centers = [(whole_u[0], whole_v[0]), *zip(part_u[:-1], part_v[:-1])]
    for view, at in zip(views, centers, strict=True):
        assert_centered(view, 30, at)


@pytest.mark.parametrize("window", [0, -96, 96.5, "96", True])
def test_a_window_that_is_not_a_whole_number_of_pixels_is_refused_before_the_run(closeup_clip, tmp_path, window):
    tracks = [track(closeup_clip, "A"), track(closeup_clip, "B", mode="fine")]
    tracks[1].fine_window_px = window
    session = make_session(closeup_clip, tmp_path, tracks)
    segmenter = Watched(ExactFake(closeup_clip))
    with pytest.raises(ValueError, match=r"\bB\b.*window"):
        run(closeup_clip, session, tmp_path, segmenter)
    assert segmenter.made == 0 and list(tmp_path.iterdir()) == []  # no model, no file; A was not started either


@pytest.mark.parametrize("factor", [0, -3.0, float("nan"), "3"])
def test_a_window_factor_that_is_not_a_positive_number_is_refused_before_the_run(closeup_clip, tmp_path, factor):
    session = make_session(closeup_clip, tmp_path, [track(closeup_clip, "B", mode="fine")])
    session.processing.fine_window_factor = factor
    segmenter = Watched(ExactFake(closeup_clip))
    with pytest.raises(ValueError, match="fine_window_factor"):
        run(closeup_clip, session, tmp_path, segmenter)
    assert segmenter.made == 0 and list(tmp_path.iterdir()) == []


def test_the_sessions_window_factor_is_the_one_used(closeup_clip, tmp_path):
    session = make_session(closeup_clip, tmp_path, [track(closeup_clip, "B", mode="fine")])
    session.processing.fine_window_factor = 4.0
    segmenter = Watched(ExactFake(closeup_clip))
    status, _ = run(closeup_clip, session, tmp_path, segmenter)
    assert status == "complete"
    window = session.tracks[0].fine_window_px
    assert 185 <= window <= 192  # F = 47 px: 4 F = 188 (the brief's 139 to 144 for the factor 3, times 4 / 3)
    assert set(segmenter.shapes) == {(window, window, 3)}


# ---------------------------------------------------------------------------------------------
# Frames on which the model finds nothing (review focus 4)


class Blind(Watched):
    """Finds nothing on the given video frames, like a model that loses its object for a while."""

    def __init__(self, inner, frames):
        super().__init__(inner)
        self.blind = set(frames)

    def step(self, image):
        results = super().step(image)
        if self.views[-1][0] in self.blind:
            return [MaskResult(result.obj_id, (0, 0), np.zeros((0, 0), bool), np.zeros((0, 0), np.float32), None)
                    for result in results]
        return results


def test_a_lost_frame_keeps_the_last_crop_center_and_is_marked_lost(closeup_clip, tmp_path):
    gone = [20, 22, 24, 26]
    session = make_session(closeup_clip, tmp_path, [track(closeup_clip, "A", mode="fine")])
    segmenter = Blind(ExactFake(closeup_clip), gone)
    status, seen = run(closeup_clip, session, tmp_path, segmenter)
    assert status == "complete" and seen.finished == ["complete"]

    window = session.tracks[0].fine_window_px
    arrays = ResultsStore.load(tmp_path / "results.npz").arrays("A")
    true_u, true_v, _ = table_truth(closeup_clip, "A", GRID)
    lost = np.isin(GRID, gone)
    assert arrays.frames.tolist() == GRID  # a lost frame keeps its row
    assert arrays.visible.tolist() == (~lost).tolist()
    assert np.isnan(arrays.u[lost]).all() and np.isnan(arrays.v[lost]).all()
    assert (arrays.area_px[lost] == 0).all() and not arrays.edge.any()
    assert set(arrays.mode.tolist()) == {"fine"}
    np.testing.assert_allclose(arrays.cell_px, window / 256)
    # Found again on frame 28: A moves 2.08 px per frame, so 21 px since frame 18, and it is 64 px
    # across: still far from the border of a window of about 193 px.
    np.testing.assert_allclose(arrays.u[~lost], true_u[~lost], rtol=0, atol=0.01)
    np.testing.assert_allclose(arrays.v[~lost], true_v[~lost], rtol=0, atol=0.01)

    views = {view[0]: view for view in segmenter.views[1:]}
    row = {frame: k for k, frame in enumerate(GRID)}
    # frames 20 to 28 are all looked at around where the object was on frame 18, the last time it was found
    for frame in (20, 22, 24, 26, 28):
        assert_centered(views[frame], window, (true_u[row[18]], true_v[row[18]]))
    assert len({views[frame][1] for frame in (20, 22, 24, 26, 28)}) == 1
    # then the crop goes on from where it was found on frame 28, 21 px further
    assert_centered(views[30], window, (true_u[row[28]], true_v[row[28]]))
    assert views[30][1] != views[28][1]
    assert Session.load(tmp_path / "session.json").complete is True


def test_an_empty_preview_mask_gives_a_96_px_window_and_the_run_still_completes(disk_clip, tmp_path):
    # D is a click on the light background of `disk_scene`: no disk comes within 80 px of (250.5, 200.5).
    # A is a disk of radius 12 px at (60.3 + frame / 2, 200.5): 3 F = 72, so its window is 96 px too,
    # and hangs over the frame's bottom border (rows up to 248 of 240).
    tracks = [track(disk_clip, "A", mode="fine"), track_at("D", 0, [(250.5, 200.5)], [1], mode="fine")]
    session = make_session(disk_clip, tmp_path, tracks)
    segmenter = Watched(ThresholdFake())
    status, seen = run(disk_clip, session, tmp_path, segmenter)
    assert status == "complete" and seen.finished == ["complete"]

    saved = Session.load(tmp_path / "session.json")
    assert [a_track.fine_window_px for a_track in saved.tracks] == [96, 96]
    assert len(segmenter.shapes) == 120 and set(segmenter.shapes) == {(96, 96, 3)}
    assert [(r.tracks, r.mode, r.frames_done) for r in saved.runs] == [(["A"], "fine", 60), (["D"], "fine", 60)]
    assert saved.complete is True
    assert any("D" in line and "nothing" in line for line in seen.log)  # the log says the preview was empty

    store = ResultsStore.load(tmp_path / "results.npz")
    lost = store.arrays("D")
    assert lost.frames.tolist() == GRID
    assert not lost.visible.any() and not lost.edge.any() and (lost.area_px == 0).all()
    assert np.isnan(lost.u).all() and np.isnan(lost.v).all() and np.isnan(lost.outline_px).all()
    assert set(lost.mode.tolist()) == {"fine"}
    np.testing.assert_allclose(lost.cell_px, 96 / 256)
    # with no mask to center on, the window lies around the click: the click is at its middle, (48, 48)
    (prompt,) = segmenter.starts[1][1]
    assert prompt.obj_id == "D" and prompt.labels == [1]
    np.testing.assert_allclose(prompt.points_px, [(48.0, 48.0)], rtol=0, atol=0.5)

    # the disk next to it is followed, within the template's tolerance for this stand-in
    found = store.arrays("A")
    true = np.array([center(disk_clip, "A", frame) for frame in GRID])
    assert found.visible.all() and not found.edge.any()
    assert np.hypot(found.u - true[:, 0], found.v - true[:, 1]).max() < 0.25
    assert (found.area_px > 300).all()  # pi 12^2 = 452 px


# ---------------------------------------------------------------------------------------------
# The clicks and the window (SPEC 6.3: prompts are shifted into crop coordinates)


def test_a_point_outside_the_window_is_dropped_and_the_others_are_shifted_into_it(closeup_clip, tmp_path):
    b_u, b_v = center(closeup_clip, "B", 0)  # (42.65, 136.09)
    # a positive click on B, a negative one far to the right (the window of about 141 px around B ends
    # near u = 113), and a negative one 30 px right of B's center
    clicks = [(b_u, b_v), (250.0, 120.0), (b_u + 30.0, b_v)]
    session = make_session(closeup_clip, tmp_path, [track_at("B", 0, clicks, [1, 0, 0], mode="fine")])
    segmenter = Watched(ExactFake(closeup_clip))
    status, _ = run(closeup_clip, session, tmp_path, segmenter)
    assert status == "complete"

    # the preview is on the whole frame and gets all three
    ((shape, (previewed,)),) = segmenter.previews
    assert shape == (240, 320, 3) and previewed.labels == [1, 0, 0]
    np.testing.assert_allclose(previewed.points_px, clicks, rtol=0, atol=1e-9)
    # the run starts on the window with the two that lie in it, in the window's coordinates
    window = session.tracks[0].fine_window_px
    ((shape, (started,)),) = segmenter.starts
    (c0, r0) = segmenter.views[1][1]
    assert shape == (window, window, 3) and (started.obj_id, started.labels) == ("B", [1, 0])
    np.testing.assert_allclose(started.points_px, [(b_u - c0, b_v - r0), (b_u + 30.0 - c0, b_v - r0)], rtol=0,
                               atol=1e-9)
    # the session keeps the clicks as they were made, in px of the full frame
    assert session.tracks[0].prompts[0].points_px == [list(click) for click in clicks]


def test_a_fine_object_without_a_positive_point_in_its_window_is_an_error_before_the_run(closeup_clip, tmp_path):
    # `ExactFake` finds B wherever the click is: here the only positive click is in the far corner
    # of the frame, more than 250 px from B, so it is not in the window around the mask the model returned.
    tracks = [track(closeup_clip, "A"), track_at("B", 0, [(300.5, 20.5)], [1], mode="fine")]
    session = make_session(closeup_clip, tmp_path, tracks)
    segmenter = Watched(ExactFake(closeup_clip))
    with pytest.raises(ValueError, match=r"\bB\b.*positive"):
        run(closeup_clip, session, tmp_path, segmenter)
    assert segmenter.calls == ["preview"]  # no run began, not A's coarse one either
    assert list(tmp_path.iterdir()) == []  # and nothing was written
    assert session.runs == [] and session.tracks[1].fine_window_px is None
