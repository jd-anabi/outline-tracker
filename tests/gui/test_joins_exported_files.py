"""The joins between the panels (4 to 6): exported files older than their sources, the Quickstart, the split files.

Task C9; SPEC 6.4, 8.1, 10.1, 10.2; decision X7. Panels 5 to 9 were built side by side. These tests hold the places
where two of them meet, numbered as that build step numbered them. This file holds the last three:

4. Panel 9 needs attention when session.json or results.npz is newer than positions.csv.
5. The Quickstart marks the buttons of panels 8 and 9 in bold.
6. gui/view/prompts.py and gui/panels/video_panel.py were split: what they offered is still there.

The other joins are in tests/gui/test_joins.py (1: Export all and the flags need no model) and in
tests/gui/test_joins_lock.py (2 and 3: the one lock, the model after results).

Expected values are the brief's sentences and SPEC 8.1's list of files, never the panels' own
output. A factory, a run or an export is parked on a gate (tests/support/prompt_helpers.py), so
"during" is a state the test holds; where a file's time decides, the test sets it. Frames are
video frame numbers; (u, v) is in px of the video frame (SPEC 3.1).
"""

import re
from datetime import datetime
from pathlib import Path

import pytest
from PySide6.QtWidgets import QAbstractButton

from export_panel_helpers import export_panel, export_to_end, is_on, state, tracked
from export_panel_helpers import hint as export_hint
from gui_helpers import show
from joins_helpers import DAY, written_at
from last_controls_helpers import control_texts, marked_names, quickstart
from layout_helpers import REPO, module
from outline_tracker.core import schema
from outline_tracker.gui.worker import worker_of
from outline_tracker.segmenter.fake import ExactFake
from session_helpers import body, read_json

RESULTS_NEWER = "The results changed after the last export. Click Export all again."
SESSION_NEWER = "The session changed after the last export. Click Export all again."
# What the two modules offered before they were split (as of the commit before task C9), private names too.
PROMPTS_OFFERED = [
    "BLACK", "BUSY_TEXT", "CASING", "DoubleClickWatch", "Drawn", "Found", "KINDS", "LINE_WIDTHS", "LOADING_TEXT",
    "MARK_SIZES", "MARK_SYMBOLS", "Mark", "NOTHING_FOUND", "NO_MODEL", "NO_OBJECT", "NO_OUTLINE", "PLUS_SIZE",
    "PointTool", "Prompts", "QColor", "QKeySequence", "QObject", "QShortcut", "Qt", "ResultsOnDisk", "Signal",
    "TOOL_TEXTS", "WAIT_TEXT", "WHITE", "_Shown", "colour_of", "crop_box", "dataclass", "decoder_tag", "found_in",
    "frame_hash", "frame_shown", "jobs_of", "marks_on", "np", "pg", "point_kind", "preview_input", "sys", "tracking"]
VIDEO_PANEL_OFFERED = [
    "CHECK_ERRORS", "CONTROL_HEIGHT", "CheckThread", "ElidedLabel", "HINT_DONE", "HINT_NAME", "HINT_VIDEO",
    "HINT_WARNINGS", "LABEL_WIDTH", "LARGEST_FRAME", "Message", "NAME_MISSING", "NOT_CHECKED", "NOT_WRITTEN",
    "NO_VALUE", "Path", "QEvent", "QGridLayout", "QHBoxLayout", "QLabel", "QLineEdit", "QObject", "QPushButton",
    "QSizePolicy", "QSpinBox", "QThread", "QVBoxLayout", "QWidget", "Qt", "RUNNING", "SESSION_FILTERS", "SPACING",
    "Signal", "VideoPanel", "_WheelGuard", "build", "clip_problem", "cv2", "dialogs", "field_label", "field_rows",
    "guard_wheel", "jobs_of", "size_text", "theme", "video"]


# ---------------------------------------------------------------------------------------------
# 4. Exported files that are older than what they come from


def test_a_session_saved_after_the_export_needs_attention_and_a_new_export_is_done(window, qtbot,
                                                                                  clip_in_odd_folder):
    panel = tracked(window, qtbot, clip_in_odd_folder)
    folder = window.controller.run_folder
    export_to_end(qtbot, panel)
    assert state(window) == "done"  # a fresh export is never older: it saves the session first
    written_at(folder, results=(14, 0), session=(14, 1), positions=(14, 5))
    panel.refresh()
    assert state(window) == "done" and export_hint(window) == "Exported at 14:05. Export all replaces these files."
    window.controller.session.calibration.stick["length_mm"] = 25.0  # every position in mm is another one now
    window.controller.touch()
    window.controller.save_now()  # session.json is written now, long after 14:05 of that day
    assert state(window) == "attention" and export_hint(window) == SESSION_NEWER
    assert is_on(window) and panel.export_button.property("kind") == "primary"
    assert panel.open_button.property("kind") != "primary"
    export_to_end(qtbot, panel)
    assert state(window) == "done" and panel.message.kind == "success"
    assert export_hint(window).startswith("Exported at ") and panel.open_button.property("kind") == "primary"
    assert read_json(folder / schema.SESSION_JSON)["calibration"]["stick"]["length_mm"] == 25.0


def test_results_written_after_the_export_need_attention_and_a_new_export_is_done(window, qtbot, clip_in_odd_folder):
    """The panel's state comes from the times of three files: session.json counts as well as
    results.npz and positions.csv, so the test sets the time of each."""
    panel = tracked(window, qtbot, clip_in_odd_folder)
    folder = window.controller.run_folder
    assert state(window) == "todo" and panel.export_button.property("kind") == "primary"
    export_to_end(qtbot, panel)
    assert state(window) == "done"
    written_at(folder, results=(14, 0), session=(14, 1), positions=(14, 5))
    panel.refresh()
    assert state(window) == "done" and export_hint(window) == "Exported at 14:05. Export all replaces these files."
    written_at(folder, results=(14, 10))  # tracked again since
    panel.refresh()
    assert state(window) == "attention" and export_hint(window) == RESULTS_NEWER
    assert is_on(window) and panel.export_button.property("kind") == "primary"
    assert panel.message.isHidden() and panel.files_label.isHidden()  # what the older export wrote is not listed
    export_to_end(qtbot, panel)  # positions.csv is written now, long after 14:10 of that day
    assert state(window) == "done" and panel.message.kind == "success"


def test_an_export_on_the_disk_is_done_when_its_session_is_opened(window, second_window, qtbot,
                                                                  clip_in_odd_folder):
    """A session that is opened again: panel 9 is done when neither session.json nor results.npz
    is newer than positions.csv. A window that closes does not write a session that is on the
    disk as it is, and opening a session does not write it either."""
    panel = tracked(window, qtbot, clip_in_odd_folder)
    folder = window.controller.run_folder
    export_to_end(qtbot, panel)
    written_at(folder, results=(9, 30), session=(9, 30), positions=(9, 31))
    window.close()
    assert datetime.fromtimestamp((folder / schema.SESSION_JSON).stat().st_mtime) == datetime(*DAY, 9, 30)
    second_window.segmenter_factory = lambda model, device: ExactFake(clip_in_odd_folder)
    second_window.open_path(folder / schema.SESSION_JSON)
    show(second_window, qtbot)
    qtbot.waitUntil(lambda: worker_of(second_window).ready)
    again = export_panel(second_window)
    assert state(second_window) == "done" and is_on(second_window)
    assert export_hint(second_window) == "Exported at 09:31. Export all replaces these files."
    assert again.message.isHidden() and again.files_label.isHidden() and again.open_button.isEnabled()
    assert not second_window.controller.save_timer.isActive()  # nothing is due to be saved: nothing changed


# ---------------------------------------------------------------------------------------------
# 5. The Quickstart


def steps_of_the_quickstart() -> dict[int, str]:
    """The numbered steps of the README's Quickstart, by number (each is one line there)."""
    section = quickstart((REPO / "README.md").read_text(encoding="utf-8"))
    return {int(number): text for number, text in re.findall(r"^(\d+)\. +(.*)$", section, flags=re.MULTILINE)}


def test_the_quickstart_names_the_buttons_of_panels_8_and_9_in_bold_and_each_is_the_windows_own(window):
    steps, texts = steps_of_the_quickstart(), control_texts(window)
    for number, needed in ((8, "Re-track from here"), (9, "Export all")):
        buttons = {button.text() for button in body(window, number).findChildren(QAbstractButton)
                   if button.text()}  # a table has a corner button without a text
        marked = marked_names(steps[number])
        assert needed in buttons and needed in marked, (number, marked)
        assert [name for name in marked if name not in texts] == []
        unmarked = re.sub(r"\*\*[^*]+\*\*", "", steps[number])  # the step without what is in bold
        assert [text for text in buttons if text in unmarked] == [], number


# ---------------------------------------------------------------------------------------------
# 6. The two files that were split


@pytest.mark.parametrize("name, offered", [(module("prompts"), PROMPTS_OFFERED),
                                           (module("video panel"), VIDEO_PANEL_OFFERED)],
                         ids=["prompts.py", "video_panel.py"])
def test_every_name_a_split_module_offered_is_still_importable_from_it_and_it_is_under_400_lines(name, offered):
    import importlib

    module = importlib.import_module(name)
    assert [wanted for wanted in offered if not hasattr(module, wanted)] == []
    lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
    assert len(lines) < 400, f"{Path(module.__file__).name} has {len(lines)} lines"


def test_the_parts_that_moved_are_the_same_objects_wherever_they_are_imported_from():
    from outline_tracker.gui.widgets import panel_parts
    from outline_tracker.gui.view import prompt_drawing, prompts
    from outline_tracker.gui.panels import video_panel

    for name in ("Message", "ElidedLabel", "guard_wheel", "field_label", "field_rows", "_WheelGuard"):
        assert getattr(video_panel, name) is getattr(panel_parts, name), name
    for name in ("Drawn", "colour_of", "BLACK", "CASING", "LINE_WIDTHS", "MARK_SIZES"):
        assert getattr(prompts, name) is getattr(prompt_drawing, name), name
