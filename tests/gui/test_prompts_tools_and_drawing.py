"""Clicks on objects (SPEC 3.1, 3.4, 3.5, 5, 10.1, 10.2; task C4): the tools Positive, Negative and Head, and
what is drawn of the points and of the model's outline.

Which click is negative, how a click becomes a point, undo, and where a click may be placed are in
tests/gui/test_prompts.py.

Expected values come from the spec, from geometry and from the synthetic clip:
- a click lands at the (u, v) under the cursor (a frame of W x H px that is shown whole in a view
  of w x h px is drawn at min(w / W, h / H) screen px per video px, in the middle);
- the disks of `synthetic.disk_scene` have radius 12 px and known centers, so the outline of a
  clicked disk lies on that circle;
- the texts and cursors of the tools are the design note's.
Nothing is read from a pixel of text; the one test that reads the drawn view looks for the track's
colour where the click was made.
"""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui_helpers import drawn
from prompt_helpers import DISK_RADIUS, LEFT, NO_KEY, at, clicked_object, panel_with
from tracking_helpers import center

YELLOW, MAGENTA = "#FFFF00", "#FF00FF"  # the colours of the tracks A and B (X14)
BACKGROUND = (250.5, 60.5)  # a point of frame 0 of the disk clip that is far from every disk


# ---------------------------------------------------------------------------------------------
# The tools


def test_a_point_tool_shows_the_cross_and_says_what_a_click_does(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    view, prompts = window.view, panel.prompts
    panel.add_object()
    assert view.tool is prompts.tools["positive"] and prompts.tool_kind == "positive"
    assert view.viewport().cursor().shape() == Qt.CursorShape.CrossCursor
    assert window.tool_text.text() == "Positive: click on animal A."
    prompts.choose_tool("negative")
    assert window.tool_text.text() == "Negative: click on what is not animal A."
    prompts.choose_tool("head")
    assert window.tool_text.text() == "Head: click on the head of A."
    assert view.viewport().cursor().shape() == Qt.CursorShape.CrossCursor
    panel.add_object()  # B is selected now, and Add chooses Positive
    assert window.tool_text.text() == "Positive: click on animal B."
    prompts.select("A")
    assert window.tool_text.text() == "Positive: click on animal A."
    prompts.choose_tool(None)
    assert view.tool is None and prompts.tool_kind is None
    assert window.tool_text.text() == "Pan: drag to move the picture. Scroll to zoom."


def test_escape_returns_to_pan_and_releases_the_tool_button(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    panel.add_object()
    assert panel.positive_button.isChecked()
    panel.negative_button.click()
    assert window.view.tool is panel.prompts.tools["negative"]
    assert (panel.positive_button.isChecked(), panel.negative_button.isChecked()) == (False, True)
    qtbot.keyClick(window.view, Qt.Key.Key_Escape)
    assert window.view.tool is None
    assert not any(button.isChecked() for button in (panel.positive_button, panel.negative_button, panel.head_button))
    panel.head_button.click()
    assert window.view.tool is panel.prompts.tools["head"] and panel.head_button.isChecked()
    panel.head_button.click()  # a second click on the checked button: back to Pan
    assert window.view.tool is None and not panel.head_button.isChecked()


# ---------------------------------------------------------------------------------------------
# What is drawn


def test_a_point_is_drawn_where_it_was_clicked_in_the_objects_colour(window, qtbot, disk_clip):
    panel = panel_with(window, qtbot, disk_clip)
    view = window.view
    panel.add_object()
    QApplication.processEvents()
    assert drawn(view, (255, 255, 0)) is None  # nothing of the clip is yellow
    aimed = at(view, *BACKGROUND)
    qtbot.mouseClick(view.viewport(), LEFT, NO_KEY, aimed)  # on the background: the model finds nothing there
    qtbot.waitUntil(lambda: not panel.prompts.busy)
    disc = drawn(view, (255, 255, 0))  # A is yellow; the disc is 12 screen px across with a black edge
    assert disc is not None
    assert (disc.left + disc.right) / 2 == pytest.approx(aimed.x() + 0.5, abs=1.5)
    assert (disc.top + disc.bottom) / 2 == pytest.approx(aimed.y() + 0.5, abs=1.5)
    assert 6 <= disc.width <= 12 and 6 <= disc.height <= 12
    assert panel.prompts.outlines == {}
    assert panel.prompts.message == ("warning", "The model found nothing at the points of object A. "
                                                "Click on the animal itself.")


def test_the_outline_of_a_clicked_disk_lies_on_the_disk(window, qtbot, disk_clip):
    panel, _ = clicked_object(window, qtbot, disk_clip, "B")
    prompts = panel.prompts
    assert prompts.busy  # the request has returned, and the outline is on its way
    qtbot.waitUntil(lambda: "A" in prompts.outlines)
    assert not prompts.busy
    outline = prompts.outlines["A"]  # the object is A; it was placed on the scene's disk B
    cu, cv = center(disk_clip, "B", 0)
    radii = np.hypot(outline[:, 0] - cu, outline[:, 1] - cv)
    assert len(outline) >= 64
    assert np.abs(radii - DISK_RADIUS).max() < 1.5  # px: the disk's edge is one blurred pixel wide
    assert radii.mean() == pytest.approx(DISK_RADIUS, abs=0.75)
    drawn_as = prompts.graphics["A"]
    assert drawn_as.line.opts["pen"].color().name().upper() == YELLOW
    assert drawn_as.casing.opts["pen"].color().name() == "#000000"
    assert drawn_as.casing.opts["pen"].widthF() > drawn_as.line.opts["pen"].widthF()
    assert drawn_as.label.toPlainText() == "A"
    assert prompts.message == ("", "")


def test_every_object_clicked_on_the_frame_gets_its_outline_in_its_colour(window, qtbot, disk_clip):
    panel, _ = clicked_object(window, qtbot, disk_clip, "A")
    prompts = panel.prompts
    panel.add_object()
    assert prompts.add_point(*center(disk_clip, "B", 0), 1)
    qtbot.waitUntil(lambda: set(prompts.outlines) == {"A", "B"})
    for track_id, colour in (("A", YELLOW), ("B", MAGENTA)):
        cu, cv = center(disk_clip, track_id, 0)
        outline = prompts.outlines[track_id]
        assert np.abs(np.hypot(outline[:, 0] - cu, outline[:, 1] - cv) - DISK_RADIUS).max() < 1.5
        assert prompts.graphics[track_id].line.opts["pen"].color().name().upper() == colour
    # the selected object (B) is drawn with the wider line
    assert prompts.graphics["B"].line.opts["pen"].widthF() > prompts.graphics["A"].line.opts["pen"].widthF()
    prompts.select("A")
    assert prompts.graphics["A"].line.opts["pen"].widthF() > prompts.graphics["B"].line.opts["pen"].widthF()


def test_opening_another_video_returns_to_pan_and_clears_points_and_outlines(window, qtbot, disk_clip, dish_clip):
    panel, _ = clicked_object(window, qtbot, disk_clip)
    prompts = panel.prompts
    qtbot.waitUntil(lambda: "A" in prompts.outlines)
    assert window.view.tool is prompts.tools["positive"] and prompts.selected == "A"
    window.open_path(dish_clip.path)
    assert window.view.tool is None and prompts.tool_kind is None
    assert window.tool_text.text() == "Pan: drag to move the picture. Scroll to zoom."
    assert not panel.positive_button.isChecked()
    assert prompts.marks == [] and prompts.outlines == {} and prompts.graphics == {}
    assert prompts.selected is None and prompts.message == ("", "") and not prompts.busy
    assert panel.table.rowCount() == 0
    assert drawn(window.view, (255, 255, 0)) is None
