"""Results on the picture (SPEC 10.1; task C5): for the frame the view shows, each track's stored
outline, centroid, id and head mark, in the track's colour, read from results.npz as it is on disk.

Expected values come from the synthetic dish scene at 320 x 240 px and from the store itself:
- `ExactFake` answers with the scene's masks, so the outline stored for frame k encloses the true
  mask of frame k: the centroid of the area inside it (shoelace formula) is the body's center on
  that frame. B is an ellipse that moves 0.643 px per frame to the right (5 mm/s at 0.0324 mm per
  px and 240 frames per second), so the outline of a neighbouring tracked frame is 1.29 px away:
  a drawing of the wrong frame cannot pass a tolerance of 0.3 px;
- B's body has the half length a = 0.235 mm / 0.0324 mm per px = 7.25 px and its head points
  along its path, to the right: its head end is at (u + 7.25, v) on every frame; C moves to the
  left, so its head end is at (u - 7.25, v);
- the colours of the first three objects are yellow, magenta and green (decision X14).
A lost frame, a frame that was not tracked and a track the session no longer has draw nothing.

Coordinates: (u, v) in px of the video frame (SPEC 3.1: u to the right, v downward, pixel centers at
+0.5). Frames are video frame numbers. Nothing is read from a pixel of text.
"""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

import helpers
from outline_tracker import schema, tracking
from outline_tracker.measure import measure_mask
from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.base import MaskResult
from outline_tracker.segmenter.fake import ExactFake
from prompt_helpers import Gate
from session_helpers import body
from track_helpers import NAME, Tracked, own_copy, ready_to_track, results_of, run_to_end
from tracking_helpers import center, table_truth

COLOURS = {"A": "#FFFF00", "B": "#FF00FF", "C": "#00FF00"}
HALF_LENGTH = 0.235 / 0.0324  # of the plain bodies B and C, px


def inside_center(outline) -> tuple[float, float]:
    """The centroid of the area inside a closed polygon [N, 2] (shoelace formula), in its units."""
    x, y = outline[:, 0], outline[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    return (float(((x + np.roll(x, -1)) * cross).sum() / (3 * cross.sum())),
            float(((y + np.roll(y, -1)) * cross).sum() / (3 * cross.sum())))


def line_points(item) -> np.ndarray:
    """The points [N, 2] of a curve drawn in the view, px of the video frame."""
    return np.column_stack(item.getData())


def pen_colour(item) -> str:
    return item.opts["pen"].color().name().upper()


def tracked(window, qtbot, clip, ids="ABC", end=20, heads=None):
    """A window whose objects `ids` of the dish clip were tracked to frame `end` by a job with
    `ExactFake`; `heads` maps an id to its head click (u, v) on frame 0. Returns the `TrackPanel`."""
    panel, objects = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), ids=ids, end=end)
    for track_id, (u, v) in (heads or {}).items():
        objects.prompts.select(track_id)
        assert objects.prompts.set_head(u, v)
    run_to_end(qtbot, panel)
    assert panel.jobs.status == "complete"
    return panel


def found_on(clip, frames, ids) -> ResultsStore:
    """Results as a job would leave them: each object of `ids` found on each video frame of
    `frames`, where the scene of `clip` has it (the masks of `ExactFake`, measured as tracking
    measures them)."""
    store, truth, picture = ResultsStore(), ExactFake(clip), np.zeros((240, 320, 3), np.uint8)
    for frame in frames:
        truth.set_view(frame, (0, 0), (320, 240))
        clicks = [helpers.click(track_id, *center(clip, track_id, frame)) for track_id in ids]
        for track_id, result in zip(ids, truth.preview(picture, clicks)):
            store.put(track_id, measure_mask(result, frame, (0, 0, 320, 240), "coarse"))
    return store


# ---------------------------------------------------------------------------------------------
# What is drawn on a tracked frame


def test_nothing_is_drawn_before_anything_is_tracked(window, qtbot, clip_in_odd_folder):
    panel, _ = ready_to_track(window, qtbot, clip_in_odd_folder, Tracked(ExactFake(clip_in_odd_folder)))
    assert panel.overlays.shown == {}


@pytest.mark.parametrize("frame", [0, 10, 20])
def test_the_outline_drawn_on_frame_k_is_the_stored_outline_of_frame_k(window, qtbot, clip_in_odd_folder, frame):
    clip = clip_in_odd_folder
    panel = tracked(window, qtbot, clip)
    window.show_frame(frame)
    shown, store = panel.overlays.shown, results_of(window)
    assert list(shown) == ["A", "B", "C"]
    for track_id in "ABC":
        arrays = store.arrays(track_id)
        row = arrays.row(frame)
        drawn = shown[track_id]
        stored = arrays.outline_px[row].astype(float)
        assert np.array_equal(drawn.outline, stored)
        # the line in the view is that outline, closed, in the track's colour, on a wider dark casing
        for item in (drawn.line, drawn.casing):
            assert np.array_equal(line_points(item)[:-1], stored)
            assert np.array_equal(line_points(item)[-1], stored[0])
            assert item.scene() is window.view.scene()
        assert pen_colour(drawn.line) == COLOURS[track_id]
        assert drawn.casing.opts["pen"].widthF() > drawn.line.opts["pen"].widthF()
        assert drawn.casing.zValue() < drawn.line.zValue()
        # the centroid is the stored one, and the id is the track's
        assert drawn.center == (arrays.u[row], arrays.v[row])
        assert drawn.label.toPlainText() == track_id
    # and it is this frame's: it encloses the plain bodies where the scene puts them on frame k
    for track_id in "BC":
        cu, cv = center(clip, track_id, frame)
        iu, iv = inside_center(shown[track_id].outline)
        assert abs(iu - cu) < 0.3 and abs(iv - cv) < 0.3
        assert abs(shown[track_id].center[0] - cu) < 0.3 and abs(shown[track_id].center[1] - cv) < 0.3


def test_the_centroid_dots_are_in_the_tracks_colours_at_the_stored_centers(window, qtbot, clip_in_odd_folder):
    panel = tracked(window, qtbot, clip_in_odd_folder)
    window.show_frame(6)
    store = results_of(window)
    dots = {(float(spot.pos().x()), float(spot.pos().y())): spot.brush().color().name().upper()
            for spot in panel.overlays.dots.points()}
    expected = {}
    for track_id in "ABC":
        arrays = store.arrays(track_id)
        row = arrays.row(6)
        expected[(float(arrays.u[row]), float(arrays.v[row]))] = COLOURS[track_id]
    assert dots == expected


def test_a_frame_that_was_not_tracked_draws_nothing(window, qtbot, clip_in_odd_folder):
    panel = tracked(window, qtbot, clip_in_odd_folder, end=10)
    window.show_frame(10)
    assert set(panel.overlays.shown) == {"A", "B", "C"}
    items = [item for drawn in panel.overlays.shown.values() for item in drawn.items()]
    for frame in (11, 12, 30):  # between two grid frames, and after the clip's end
        window.show_frame(frame)
        assert panel.overlays.shown == {}
        assert all(item.scene() is None for item in items)  # taken off the picture
        assert len(panel.overlays.dots.points()) == 0 and len(panel.overlays.heads.points()) == 0
    window.show_frame(8)
    assert set(panel.overlays.shown) == {"A", "B", "C"}


def test_a_lost_frame_draws_nothing_for_that_track(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    panel, objects = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), ids="AB")
    # results as a job would leave them: A is found on frames 0 and 2 and lost on frame 4; B is found on all three
    store, truth, picture = ResultsStore(), ExactFake(clip), np.zeros((240, 320, 3), np.uint8)
    nothing = MaskResult(obj_id="A", offset=(0, 0), mask=np.zeros((0, 0), bool),
                         logits=np.zeros((0, 0), np.float32), score=None)
    for frame in (0, 2, 4):
        truth.set_view(frame, (0, 0), (320, 240))
        clicks = [helpers.click(track_id, *center(clip, track_id, frame)) for track_id in "AB"]
        found = dict(zip("AB", truth.preview(picture, clicks)))
        if frame == 4:
            found["A"] = nothing
        for track_id, result in found.items():
            store.put(track_id, measure_mask(result, frame, (0, 0, 320, 240), "coarse"))
    store.save(window.controller.run_folder / schema.RESULTS_NPZ)
    panel.overlays.reload()
    window.show_frame(2)
    assert set(panel.overlays.shown) == {"A", "B"}
    window.show_frame(4)
    assert set(panel.overlays.shown) == {"B"}
    assert len(panel.overlays.dots.points()) == 1


def test_a_track_the_session_no_longer_has_is_not_drawn(window, qtbot, clip_in_odd_folder):
    panel = tracked(window, qtbot, clip_in_odd_folder, ids="AB", end=10)
    window.show_frame(4)
    assert set(panel.overlays.shown) == {"A", "B"}
    objects = body(window, 6)
    objects.prompts.select("A")
    objects.remove_selected()
    assert set(panel.overlays.shown) == {"B"}


def test_another_video_takes_the_drawings_off(window, qtbot, clip_in_odd_folder, disk_clip, tmp_path):
    panel = tracked(window, qtbot, clip_in_odd_folder, ids="A", end=10)
    window.show_frame(4)
    assert set(panel.overlays.shown) == {"A"}
    window.open_path(own_copy(disk_clip, tmp_path / "other").path)
    assert panel.overlays.shown == {} and len(panel.overlays.dots.points()) == 0


# ---------------------------------------------------------------------------------------------
# The head mark


def test_the_head_mark_is_at_the_head_end_filled_when_the_head_was_clicked(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    bu, bv = center(clip, "B", 0)
    panel = tracked(window, qtbot, clip, heads={"B": (bu + 5.0, bv)})
    for frame in (0, 12, 20):
        window.show_frame(frame)
        shown = panel.overlays.shown
        bu, bv = center(clip, "B", frame)
        cu, cv = center(clip, "C", frame)
        assert shown["B"].head_clicked and not shown["C"].head_clicked  # C's head side is a guess
        assert np.hypot(shown["B"].head[0] - (bu + HALF_LENGTH), shown["B"].head[1] - bv) < 1.0
        assert np.hypot(shown["C"].head[0] - (cu - HALF_LENGTH), shown["C"].head[1] - cv) < 1.0
        marks = {(float(spot.pos().x()), float(spot.pos().y())): spot for spot in panel.overlays.heads.points()}
        assert set(marks) == {shown[track_id].head for track_id in "ABC"}
        filled, hollow = marks[shown["B"].head], marks[shown["C"].head]
        assert filled.brush().color() == QColor("#FF00FF") and filled.brush().style() == Qt.BrushStyle.SolidPattern
        assert hollow.brush().style() == Qt.BrushStyle.NoBrush  # the edge only
        assert hollow.pen().color() == QColor("#00FF00")


def test_a_head_click_against_the_motion_turns_the_mark_to_the_other_end(window, qtbot, clip_in_odd_folder):
    clip = clip_in_odd_folder
    bu, bv = center(clip, "B", 0)
    panel = tracked(window, qtbot, clip, ids="AB", heads={"B": (bu - 5.0, bv)})  # B swims to the right
    window.show_frame(12)
    bu, bv = center(clip, "B", 12)
    head = panel.overlays.shown["B"].head
    assert np.hypot(head[0] - (bu - HALF_LENGTH), head[1] - bv) < 1.0


# ---------------------------------------------------------------------------------------------
# Results are read again after each autosave, and the window stays usable during a run


def test_during_a_run_the_frames_saved_so_far_are_drawn(window, qtbot, clip_in_odd_folder, monkeypatch):
    clip = clip_in_odd_folder
    monkeypatch.setattr(tracking, "AUTOSAVE_EVERY", 5)
    with Gate() as gate:
        segmenter = Tracked(ExactFake(clip), gate=gate, park_at={8})
        panel, _ = ready_to_track(window, qtbot, clip, segmenter)
        panel.track()
        qtbot.waitUntil(gate.parked.is_set)
        qtbot.waitUntil(lambda: panel.jobs.done == 7)
        # seven frames are tracked (0 to 12), and the first five were saved (0 to 8)
        assert results_of(window).arrays("A").frames.tolist() == [0, 2, 4, 6, 8]
        for frame in (0, 8):
            window.show_frame(frame)  # the bottom bar works while the model is in a frame
            assert set(panel.overlays.shown) == {"A"}
            (tu,), (tv,), _ = table_truth(clip, "A", [frame])  # the centroid of the true mask
            assert np.hypot(panel.overlays.shown["A"].center[0] - tu, panel.overlays.shown["A"].center[1] - tv) < 0.01
        window.show_frame(10)  # tracked, not saved yet
        assert panel.overlays.shown == {}
        gate.open()
        qtbot.waitUntil(lambda: not panel.jobs.running)
    assert set(panel.overlays.shown) == {"A"}  # frame 10, once the run has ended
    assert results_of(window).arrays("A").frames.tolist() == list(range(0, 120, 2))
    assert window.controller.run_folder.name == f"dish_tracker_outline_{NAME}"


def test_a_results_file_that_cannot_be_opened_at_this_moment_leaves_the_drawing_until_the_next_reload(
        window, qtbot, clip_in_odd_folder, refuse_open):
    # On Windows the open of results.npz can be refused while the worker replaces the file at the next
    # autosave (CI met it in the test above). The rule: what is drawn stays, nothing is raised, and the
    # next reload draws what the file holds. Here the test writes the files, and `refuse_open` refuses.
    clip = clip_in_odd_folder
    panel, _ = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), ids="AB")
    path = window.controller.run_folder / schema.RESULTS_NPZ
    found_on(clip, [0, 2], "A").save(path)  # A is tracked
    panel.overlays.reload()
    window.show_frame(2)
    assert set(panel.overlays.shown) == {"A"}
    drawn = panel.overlays.shown["A"].outline.copy()
    assert np.array_equal(drawn, results_of(window).arrays("A").outline_px[1].astype(float))
    found_on(clip, [0, 2], "AB").save(path)  # an autosave of a run that tracks B replaces the file
    opens = refuse_open()
    panel.overlays.reload()  # the open is refused: nothing is raised
    assert set(panel.overlays.shown) == {"A"}  # not the new state, and not an empty picture
    assert np.array_equal(panel.overlays.shown["A"].outline, drawn)
    assert panel.overlays.shown["A"].line.scene() is window.view.scene()
    assert len(panel.overlays.dots.points()) == 1
    assert opens.tried == 1
    panel.overlays.reload()  # the next autosave says so again: now the file is read
    assert set(panel.overlays.shown) == {"A", "B"}
    assert np.array_equal(panel.overlays.shown["A"].outline, drawn)
    assert len(panel.overlays.dots.points()) == 2
    assert opens.tried == 2
