"""Clicks on objects (SPEC 3.1, 3.4, 3.5, 5, 10.1, 10.2; task C4): which click is negative, how a click becomes a
point of the selected object, undo, and where a click may be placed.

The tools Positive, Negative and Head, and what is drawn of the points and of the model's outline, are in
tests/gui/test_prompts_tools_and_drawing.py.

Expected values come from the spec, from geometry and from the synthetic clip:
- which click is negative is the table of SPEC 5 (the physical Control key is `MetaModifier` on
  macOS and `ControlModifier` elsewhere; Command, which Qt calls Control on a Mac, stays positive);
- a click lands at the (u, v) under the cursor (a frame of W x H px that is shown whole in a view
  of w x h px is drawn at min(w / W, h / H) screen px per video px, in the middle), on the frame
  shown, with the hash of that frame as the tracking decoder delivers it (`video.reader.iter_rgb_frames`);
- a frame off the clip's grid is moved forward to the grid (X20).
Nothing is read from a pixel of text.
"""

import copy
import sys

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

import helpers
from gui_helpers import StandInSource, fit_of, wheel
from outline_tracker.tracking import job as tracking
from outline_tracker.gui.view.click_rules import is_negative_click
from outline_tracker.measure.results import ResultsStore
from prompt_helpers import (ALT, LEFT, MIDDLE, NO_KEY, QT_CONTROL, QT_META, RIGHT, SHIFT, Watched, at, clicked_object,
                            double_click, panel_with)
from tracking_helpers import center, frame_identity


def session_of(window):
    return window.controller.session


def points_of(track) -> list[tuple[int, list, list]]:
    """A track's clicks as (frame, points, labels), one entry per stored prompt."""
    return [(prompt.frame, prompt.points_px, prompt.labels) for prompt in track.prompts]


# ---------------------------------------------------------------------------------------------
# Which click is negative (SPEC 5)


@pytest.mark.parametrize("button, modifiers, platform, negative", [
    (LEFT, NO_KEY, "darwin", False), (LEFT, NO_KEY, "win32", False), (LEFT, NO_KEY, "linux", False),
    # a right click; on a Mac a two-finger click and a Control-click arrive as one
    (RIGHT, NO_KEY, "darwin", True), (RIGHT, NO_KEY, "win32", True), (RIGHT, NO_KEY, "linux", True),
    (RIGHT, QT_META, "darwin", True),  # a Control-click that the Mac hands over as a right click
    # Alt, the Option key of a Mac
    (LEFT, ALT, "darwin", True), (LEFT, ALT, "win32", True), (LEFT, ALT, "linux", True),
    # the physical Control key: Qt calls it Meta on a Mac and Control elsewhere
    (LEFT, QT_META, "darwin", True), (LEFT, QT_CONTROL, "win32", True), (LEFT, QT_CONTROL, "linux", True),
    # the Command key of a Mac, which Qt calls Control there: positive
    (LEFT, QT_CONTROL, "darwin", False),
    # the Windows key is no Control key
    (LEFT, QT_META, "win32", False), (LEFT, QT_META, "linux", False),
    (LEFT, SHIFT, "darwin", False), (LEFT, SHIFT, "win32", False),
    (LEFT, SHIFT | ALT, "win32", True), (LEFT, QT_CONTROL | QT_META, "darwin", True),
    (MIDDLE, NO_KEY, "darwin", False), (MIDDLE, NO_KEY, "win32", False),
])
def test_which_click_is_negative(button, modifiers, platform, negative):
    assert is_negative_click(button, modifiers, platform) is negative


# ---------------------------------------------------------------------------------------------
# A click becomes a point of the selected object


def test_a_left_click_adds_a_positive_point_where_it_was_clicked_with_the_frames_hash(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    view = window.view
    window.navigation.step(3)  # the clip's grid is 0, 2, 4, ...: frame 6
    assert view.frame == 6
    panel.add_object()  # object A, and the Positive tool
    QApplication.processEvents()
    scale, left, top = fit_of(view, helpers.SMALL)
    qtbot.mouseClick(view.viewport(), LEFT, NO_KEY, QPoint(200, 150))  # the middle of screen px (200, 150)

    (track,) = session_of(window).tracks
    (prompt,) = track.prompts
    assert (track.id, track.start_frame, prompt.frame, prompt.labels) == ("A", 6, 6, [1])
    ((u, v),) = prompt.points_px
    assert u == pytest.approx((200.5 - left) / scale, abs=0.5)
    assert v == pytest.approx((150.5 - top) / scale, abs=0.5)
    # the hash is that of frame 6 as tracking decodes it: what the guard at a run's start compares
    assert (prompt.frame_hash, prompt.decoder) == frame_identity(disk_clip, 6)
    assert [(mark.track_id, mark.kind) for mark in panel.prompts.marks] == [("A", "positive")]
    assert (panel.prompts.marks[0].u, panel.prompts.marks[0].v) == (u, v)


def test_a_right_click_is_a_negative_point_in_every_point_tool(window, qtbot, disk_clip):
    panel, (u, v) = clicked_object(window, qtbot, disk_clip)
    view, prompts = window.view, panel.prompts
    QApplication.processEvents()
    qtbot.mouseClick(view.viewport(), RIGHT, NO_KEY, at(view, u + 30, v))  # beside the disk
    (track,) = session_of(window).tracks
    assert [labels for _, _, labels in points_of(track)] == [[1, 0]]
    assert track.prompts[0].points_px[1][0] == pytest.approx(u + 30, abs=1)
    prompts.choose_tool("head")
    prompts.tools["head"].click(u - 30, v, RIGHT, NO_KEY)
    assert track.prompts[0].labels == [1, 0, 0] and track.head_px is None
    assert [mark.kind for mark in prompts.marks] == ["positive", "negative", "negative"]


def test_the_keys_of_spec_5_make_a_left_click_negative(window, qtbot, disk_clip):
    panel, (u, v) = clicked_object(window, qtbot, disk_clip)
    tool = panel.prompts.tools["positive"]
    control_key = QT_META if sys.platform == "darwin" else QT_CONTROL  # the physical Control key
    other_key = QT_CONTROL if sys.platform == "darwin" else QT_META    # Command, or the Windows key
    tool.click(u + 30, v, LEFT, ALT)
    tool.click(u - 30, v, LEFT, control_key)
    tool.click(u + 2, v, LEFT, other_key)
    assert session_of(window).tracks[0].prompts[0].labels == [1, 0, 0, 1]
    tool.click(u, v + 30, MIDDLE, NO_KEY)  # the wheel's button is no click on an object
    assert session_of(window).tracks[0].prompts[0].labels == [1, 0, 0, 1]


def test_the_negative_tool_makes_every_click_negative(window, qtbot, disk_clip):
    panel, (u, v) = clicked_object(window, qtbot, disk_clip)
    panel.prompts.choose_tool("negative")
    assert window.view.tool is panel.prompts.tools["negative"]
    window.view.tool.click(u + 30, v, LEFT, NO_KEY)
    window.view.tool.click(u - 30, v, RIGHT, NO_KEY)
    (prompt,) = session_of(window).tracks[0].prompts
    assert prompt.labels == [1, 0, 0]
    assert prompt.points_px == [[u, v], [u + 30, v], [u - 30, v]]


def test_a_head_click_fills_head_px_and_is_not_passed_to_the_segmenter(window, qtbot, disk_clip):
    segmenter = Watched()
    panel, (u, v) = clicked_object(window, qtbot, disk_clip, segmenter=segmenter)
    prompts = panel.prompts
    qtbot.waitUntil(lambda: "A" in prompts.outlines)
    asked = len(segmenter.previews)
    prompts.choose_tool("head")
    window.view.tool.click(u + 6.0, v - 2.0, LEFT, NO_KEY)
    (track,) = session_of(window).tracks
    assert track.head_px == [u + 6.0, v - 2.0]
    assert points_of(track) == [(0, [[u, v]], [1])]  # no point of the object was added
    assert [(mark.kind, mark.u, mark.v) for mark in prompts.marks] == [("positive", u, v), ("head", u + 6.0, v - 2.0)]
    # the next change asks for an outline again: the head click itself asked for none, and no
    # preview was ever given the head's point
    assert prompts.add_point(u + 40.0, v, 0)
    qtbot.waitUntil(lambda: len(segmenter.previews) == asked + 1 and not prompts.busy)
    assert segmenter.previews[asked:] == [[("A", [(u, v), (u + 40.0, v)], [1, 0])]]
    assert all(points == [(u, v)] for call in segmenter.previews[:asked] for _, points, _ in call)


def test_a_head_click_on_another_frame_is_refused_in_the_functions_words(window, qtbot, disk_clip):
    panel, (u, v) = clicked_object(window, qtbot, disk_clip)
    window.navigation.step(1)  # frame 2: the head click belongs to the start frame, frame 0
    panel.prompts.choose_tool("head")
    window.view.tool.click(u, v, LEFT, NO_KEY)
    assert session_of(window).tracks[0].head_px is None
    with pytest.raises(ValueError) as refused:
        tracking.set_head(copy.deepcopy(session_of(window)), "A", 2, (u, v))
    assert panel.prompts.message == ("problem", str(refused.value))
    assert panel.message_label.text() == str(refused.value) and not panel.message_label.isHidden()


# ---------------------------------------------------------------------------------------------
# Undo


def test_undo_removes_the_last_point_of_the_selected_object(window, qtbot, disk_clip):
    panel, (u, v) = clicked_object(window, qtbot, disk_clip)
    prompts = panel.prompts
    assert prompts.add_point(u + 30, v, 0)
    assert prompts.undo() is True
    (track,) = session_of(window).tracks
    assert points_of(track) == [(0, [[u, v]], [1])]
    assert [mark.kind for mark in prompts.marks] == ["positive"]
    assert prompts.undo() is True
    assert track.prompts == [] and prompts.marks == []
    assert prompts.undo() is False  # nothing left to undo: nothing happens
    assert track.prompts == []


def test_the_platforms_undo_key_removes_the_last_point(window, qtbot, disk_clip):
    panel, (u, v) = clicked_object(window, qtbot, disk_clip)
    assert panel.prompts.add_point(u + 30, v, 0)
    undo = QKeySequence(QKeySequence.StandardKey.Undo)[0]  # Ctrl+Z, which is Cmd+Z on a Mac
    qtbot.keyClick(window.view, undo.key(), undo.keyboardModifiers())
    assert points_of(session_of(window).tracks[0]) == [(0, [[u, v]], [1])]
    qtbot.keyClick(window.view, undo.key(), undo.keyboardModifiers())
    assert session_of(window).tracks[0].prompts == []


def test_after_undoing_every_point_the_outline_is_gone(window, qtbot, disk_clip):
    panel, _ = clicked_object(window, qtbot, disk_clip)
    prompts = panel.prompts
    qtbot.waitUntil(lambda: "A" in prompts.outlines)
    assert prompts.undo()
    assert prompts.outlines == {} and not prompts.busy


# ---------------------------------------------------------------------------------------------
# Where a click may be placed


def test_a_click_on_a_frame_off_the_grid_is_moved_forward_and_the_panel_says_so(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    panel.add_object()
    window.view.show_frame(3)  # not a frame of the grid 0, 2, 4, ...
    u, v = center(disk_clip, "A", 3)
    assert panel.prompts.add_point(u, v, 1)
    (track,) = session_of(window).tracks
    assert points_of(track) == [(4, [[u, v]], [1])] and track.start_frame == 4  # forward, to the grid (X20)
    # the stored hash would be frame 3's, so none is stored (tracking.edit.add_prompt)
    assert (track.prompts[0].frame_hash, track.prompts[0].decoder) == (None, None)
    # the panel says so in the function's own words, and the view shows the frame the point is on
    expected = tracking.add_prompt(copy.deepcopy(session_of(window)), ResultsStore(), "A", 3, (u, v), 1, None, None)
    assert expected.moved and "frame 4 is used instead" in expected.note
    assert panel.prompts.message == ("warning", expected.note)
    assert panel.message_label.text() == expected.note and not panel.message_label.isHidden()
    assert window.view.frame == 4 and window.navigation.frame == 4
    assert [(mark.kind, mark.u, mark.v) for mark in panel.prompts.marks] == [("positive", u, v)]


def test_a_click_outside_the_image_adds_nothing(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    view = window.view
    panel.add_object()
    QApplication.processEvents()
    middle = view.viewport().rect().center()
    wheel(view, middle, -4)  # zoom out: the canvas shows around the picture
    edge = at(view, 0.0, 0.0)
    assert edge.x() > 12 and edge.y() > 12
    qtbot.mouseClick(view.viewport(), LEFT, NO_KEY, QPoint(edge.x() - 8, edge.y() - 8))
    assert session_of(window).tracks[0].prompts == [] and panel.prompts.marks == []
    assert panel.prompts.add_point(-3.0, 20.0, 1) is False  # nor through the slot
    assert panel.prompts.add_point(20.0, 240.0, 1) is False  # v = 240 is the frame's lower edge: outside
    assert session_of(window).tracks[0].prompts == []
    qtbot.mouseClick(view.viewport(), LEFT, NO_KEY, at(view, 100.5, 100.5))  # inside: this one counts
    assert len(session_of(window).tracks[0].prompts[0].points_px) == 1


def test_a_refusal_is_shown_in_the_functions_own_words_and_changes_nothing(window, qtbot, disk_clip):
    panel, (u, v) = clicked_object(window, qtbot, disk_clip)
    window.navigation.step(2)  # frame 4, while the clicks of frame 0 are not tracked yet
    session = session_of(window)
    before = copy.deepcopy(session.to_json())
    with pytest.raises(ValueError) as refused:
        tracking.add_prompt(copy.deepcopy(session), ResultsStore(), "A", 4, (u, v), 1, None, None)
    assert "not tracked yet" in str(refused.value)
    assert panel.prompts.add_point(u, v, 1) is False
    assert session.to_json() == before
    assert panel.prompts.message == ("problem", str(refused.value))
    assert panel.message_label.text() == str(refused.value) and not panel.message_label.isHidden()


def test_a_picture_that_is_not_the_videos_frame_takes_no_click(window, qtbot, disk_clip):
    # The hash stored with a click is the hash of the video's frame as tracking reads it. A picture
    # in the view that does not come from the session's video has no such hash: no click on it.
    panel = panel_with(window, qtbot, disk_clip)
    panel.add_object()
    window.view.set_source(StandInSource(helpers.SMALL))
    window.view.show_frame(0)
    assert panel.prompts.add_point(100.5, 80.5, 1) is False
    assert session_of(window).tracks[0].prompts == [] and panel.prompts.marks == []
    kind, text = panel.prompts.message
    assert kind == "problem" and "not stored" in text
    assert str(disk_clip.path.parent) not in text  # a message never shows a folder


def test_a_double_click_is_one_point(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    view = window.view
    panel.add_object()
    QApplication.processEvents()
    double_click(view, at(view, 100.5, 100.5))
    (prompt,) = session_of(window).tracks[0].prompts
    assert prompt.labels == [1] and len(prompt.points_px) == 1
    qtbot.mouseClick(view.viewport(), LEFT, NO_KEY, at(view, 140.5, 100.5))  # the next single click counts
    assert prompt.labels == [1, 1]
    double_click(view, at(view, 180.5, 100.5), RIGHT)
    assert prompt.labels == [1, 1, 0]


def test_without_an_object_a_click_adds_nothing_and_the_panel_says_what_to_do(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    assert panel.prompts.add_point(100.5, 100.5, 1) is False
    assert session_of(window).tracks == []
    assert panel.prompts.message == ("warning", "Click Add first. A point belongs to an object.")
