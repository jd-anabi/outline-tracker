"""The mask fill (task C8b; SPEC 10.1: "outlines (or translucent mask fill, toggle)").

"Fill", in the bar above the picture, draws each track's stored mask of the frame shown as a
translucent fill in the track's colour, under its outline. It is off by default, and the choice is
remembered between sittings (the application's settings).

What a fill covers is read from the item in the view and held against the mask that results.npz
stores for that frame, pixel by pixel, and against the scene: the pixel under the object's true
center is covered, a corner of the frame is not. That something is really painted is read from the
drawn view: the only pixels that change when Fill is switched on lie inside the boxes of the masks.

Coordinates: px of the video frame (SPEC 3.1): the pixel in column c and row r is the square
[c, c + 1) x [r, r + 1). Positions in the viewport are Qt's device-independent px.
"""

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from gui_helpers import pixels
from outline_tracker.gui.overlays import FILL_KEY
from outline_tracker.gui.panels.time_panel import settings
from outline_tracker.segmenter.fake import ExactFake
from track_helpers import Tracked, ready_to_track, results_of, run_to_end
from tracking_helpers import center

FRAME = 4  # the frame the tests look at: tracked (the clip runs from 0 to 10 at step 2)


@pytest.fixture(autouse=True)
def choice_of_this_test_only():
    """No choice is stored when a test starts, and none is left when it ends."""
    settings().remove(FILL_KEY)
    yield
    settings().remove(FILL_KEY)


@pytest.fixture
def chosen_in_a_window_before():
    """What a window of an earlier sitting stored when its Fill was switched on. Ask for it before
    `window`, which reads the settings when it is made."""
    settings().setValue(FILL_KEY, True)


def tracked(window, qtbot, clip):
    """A and B of the dish clip tracked from frame 0 to 10, and frame `FRAME` shown: the `Overlays`."""
    panel, _ = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), ids="AB", end=10)
    run_to_end(qtbot, panel)
    window.show_frame(FRAME)
    assert set(panel.overlays.shown) == {"A", "B"}
    return panel.overlays


def covered(fill) -> set[tuple[int, int]]:
    """The pixels (column, row) of the video frame that a fill item paints: where its picture is not
    clear, moved to where the item lies in the frame."""
    place = fill.mapRectToParent(fill.boundingRect())
    rows, columns = np.nonzero(fill.image[..., 3] > 0)
    assert place.size().toTuple() == (fill.image.shape[1], fill.image.shape[0])  # one entry per video px
    assert place.left() == int(place.left()) and place.top() == int(place.top())
    return {(int(place.left()) + int(column), int(place.top()) + int(row)) for row, column in zip(rows, columns)}


def stored(window, track_id) -> tuple[set[tuple[int, int]], QRectF]:
    """The pixels (column, row) of the mask results.npz holds for `track_id` on `FRAME`, and the
    box around them in px of the frame."""
    arrays = results_of(window).arrays(track_id)
    crop, (column0, row0) = arrays.mask(arrays.row(FRAME))
    rows, columns = np.nonzero(crop)
    return ({(column0 + int(column), row0 + int(row)) for row, column in zip(rows, columns)},
            QRectF(column0, row0, crop.shape[1], crop.shape[0]))


def grabbed(view) -> np.ndarray:
    QApplication.processEvents()
    return pixels(view.viewport().grab().toImage())


def test_fill_is_a_box_in_the_bar_above_the_picture_and_off_at_first(window):
    box = window.fill_box
    assert box.text() == "Fill" and box.isCheckable() and not box.isChecked() and box.toolTip()
    assert box.parent() is window.fit_button.parent()  # in the bar with Fit and 1:1


def test_fill_on_covers_the_stored_masks_pixels_in_the_tracks_colour_under_the_outline(window, qtbot,
                                                                                     clip_in_odd_folder):
    clip = clip_in_odd_folder
    overlays = tracked(window, qtbot, clip)
    assert not overlays.fill and all(shown.fill is None for shown in overlays.shown.values())
    window.fill_box.click()
    assert overlays.fill
    for track in window.controller.session.tracks:
        shown, (mask, _) = overlays.shown[track.id], stored(window, track.id)
        painted = covered(shown.fill)
        assert painted == mask and len(mask) > 0
        u, v = center(clip, track.id, FRAME)
        assert (int(u), int(v)) in painted and (0, 0) not in painted  # on the animal, not in the frame's corner
        colours = shown.fill.image[shown.fill.image[..., 3] > 0]
        red, green, blue, _ = QColor(track.color).getRgb()
        assert {tuple(int(value) for value in colour[:3]) for colour in colours} == {(red, green, blue)}
        assert {int(alpha) for alpha in colours[:, 3]} == {64}  # translucent: the animal shows through
        assert shown.fill.zValue() < shown.casing.zValue() < shown.line.zValue()  # the outline stays on top


def test_the_fill_is_painted_where_the_masks_are_and_fill_off_takes_it_away(window, qtbot, clip_in_odd_folder):
    overlays = tracked(window, qtbot, clip_in_odd_folder)
    view = window.view
    before = grabbed(view)
    window.fill_box.click()
    changed = (grabbed(view) != before).any(axis=2)
    assert changed.any()
    ratio = view.viewport().devicePixelRatioF()
    boxes = [stored(window, track_id)[1].adjusted(-1, -1, 1, 1) for track_id in overlays.shown]  # 1 px of room
    rows, columns = np.nonzero(changed)
    for row, column in zip(rows, columns):  # every changed pixel of the screen shows a pixel of some mask's box
        at = view.image_item.mapFromScene(view.mapToScene(int(column / ratio), int(row / ratio)))
        assert any(box.contains(at) for box in boxes), (int(column), int(row))
    window.fill_box.click()
    assert not overlays.fill and all(shown.fill is None for shown in overlays.shown.values())
    assert (grabbed(view) == before).all()


def test_the_fill_follows_the_frame_shown(window, qtbot, clip_in_odd_folder):
    overlays = tracked(window, qtbot, clip_in_odd_folder)
    window.fill_box.click()
    on_four = covered(overlays.shown["A"].fill)
    window.show_frame(8)
    arrays = results_of(window).arrays("A")
    crop, (column0, row0) = arrays.mask(arrays.row(8))
    rows, columns = np.nonzero(crop)
    on_eight = {(column0 + int(column), row0 + int(row)) for row, column in zip(rows, columns)}
    assert covered(overlays.shown["A"].fill) == on_eight != on_four
    window.show_frame(3)  # a frame that was not tracked: nothing to fill
    assert overlays.shown == {}


def test_the_choice_goes_to_the_settings(window):
    window.fill_box.click()
    assert settings().value(FILL_KEY, False, type=bool) is True
    window.fill_box.click()
    assert settings().value(FILL_KEY, True, type=bool) is False


def test_a_new_window_has_the_choice_of_the_window_before(chosen_in_a_window_before, window, qtbot,
                                                          clip_in_odd_folder):
    assert window.fill_box.isChecked()
    overlays = tracked(window, qtbot, clip_in_odd_folder)
    assert overlays.fill and all(shown.fill is not None for shown in overlays.shown.values())
