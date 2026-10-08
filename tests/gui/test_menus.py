"""The menus and the keys 1 to 9 (SPEC 10.1, 10.2; task C8a).

Expected values come from the spec: File has Open video, Open session, Save session, Save session
as and Quit (Export is added by the export panel's task); Help has Quickstart and About. An item
does what its button or key does elsewhere, and is off while it cannot work: saving needs a video
and the student's name, which names the run folder (SPEC 8.1). The Save key still says what is
missing then. Keys 1 to 9 select the first nine objects of panel 6, whose ids are A, B, C, ... in
the order they were added (SPEC 3.4, 5).
"""

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog

from finish_helpers import named, never_blocking, record_every_dialog  # noqa: F401 (fixture)
from gui_helpers import picture, show
from outline_tracker.gui import about, dialogs
from prompt_helpers import objects_panel
from session_helpers import FOLDER_OF_NAME, NAME, body, type_into

LEFT = Qt.MouseButton.LeftButton
CONTROL = Qt.KeyboardModifier.ControlModifier  # Qt's Ctrl is the Mac's Cmd


def save_items(window):
    return window.menus.save_action, window.menus.save_as_action


# ---------------------------------------------------------------------------------------------
# What is in the menus


def test_open_video_and_open_session_are_the_file_menus_items_and_panel_1s_buttons(window, qtbot, monkeypatch,
                                                                                   clip_in_odd_folder):
    # follows test_open_video_and_open_session_are_the_file_menus_actions of tests/gui/test_session_panels.py
    asked = record_every_dialog(monkeypatch)
    show(window, qtbot)
    panel = body(window, 1)
    assert (panel.open_video_button.text(), panel.open_session_button.text()) == ("Open video", "Open session")
    assert panel.open_session_action is window.menus.open_session_action
    assert panel.save_action is window.menus.save_action

    qtbot.mouseClick(panel.open_video_button, LEFT)
    qtbot.mouseClick(panel.open_session_button, LEFT)
    window.menus.open_action.trigger()
    window.menus.open_session_action.trigger()
    assert [(parent, title) for parent, title, _, _ in asked.files] == [
        (window, "Open video"), (window, "Open session"), (window, "Open video"), (window, "Open session")]
    filters = asked.files[3][2]
    assert "*.json" in filters[0] and filters[-1] == "All files (*)"
    asked.files[2][3](clip_in_odd_folder.path)  # the file chosen for the menu's Open video
    assert window.controller.video_path == clip_in_odd_folder.path


# ---------------------------------------------------------------------------------------------
# An item that cannot work yet is off


def test_the_items_that_save_are_off_until_there_is_a_video_and_a_name(window, qtbot, clip_in_odd_folder):
    made = window.menus
    always = (made.open_action, made.open_session_action, made.quit_action, made.quickstart_action,
              made.about_action)
    assert [item.isEnabled() for item in save_items(window)] == [False, False]  # no video
    assert all(item.isEnabled() for item in always)
    window.open_path(clip_in_odd_folder.path)
    assert [item.isEnabled() for item in save_items(window)] == [False, False]  # a video, no name
    type_into(qtbot, body(window, 1).name_edit, NAME)
    assert [item.isEnabled() for item in save_items(window)] == [True, True]
    type_into(qtbot, body(window, 1).name_edit, "")
    assert [item.isEnabled() for item in save_items(window)] == [False, False]
    assert all(item.isEnabled() for item in always)


def test_a_name_typed_before_the_video_counts_when_the_video_is_open(window, qtbot, clip_in_odd_folder):
    type_into(qtbot, body(window, 1).name_edit, NAME)
    assert [item.isEnabled() for item in save_items(window)] == [False, False]  # a name, no video
    window.open_path(clip_in_odd_folder.path)
    assert [item.isEnabled() for item in save_items(window)] == [True, True]


def test_the_save_key_says_what_is_missing_while_the_item_is_off(window, qtbot, clip_in_odd_folder):
    show(window, qtbot)
    qtbot.keyClick(window, Qt.Key.Key_S, CONTROL)
    assert window.statusBar().currentMessage() == "Type your name in panel 1 first."  # first of all the name
    window.open_path(clip_in_odd_folder.path)
    window.statusBar().clearMessage()
    assert not window.menus.save_action.isEnabled()
    qtbot.keyClick(window, Qt.Key.Key_S, CONTROL)
    assert window.statusBar().currentMessage() == "Type your name in panel 1 first."
    assert [entry.name for entry in clip_in_odd_folder.path.parent.iterdir()] == ["dish_tracker.mp4"]


def test_the_save_key_saves_once_while_the_item_is_on(window, qtbot, clip_in_odd_folder):
    path = named(window, qtbot, clip_in_odd_folder)
    show(window, qtbot)
    saves = []
    window.controller.saved.connect(lambda: saves.append(window.controller.saved_path))
    qtbot.keyClick(window, Qt.Key.Key_S, CONTROL)
    assert saves == [path.parent / FOLDER_OF_NAME / "session.json"]  # the key is the item's, and no second key's


# ---------------------------------------------------------------------------------------------
# Each item calls its function


def test_save_session_saves_as_the_save_key_does(window, qtbot, clip_in_odd_folder):
    path = named(window, qtbot, clip_in_odd_folder)
    target = path.parent / FOLDER_OF_NAME / "session.json"
    assert not target.exists()
    window.menus.save_action.trigger()
    assert target.is_file() and window.controller.saved_path == target
    assert FOLDER_OF_NAME in window.statusBar().currentMessage()


def test_save_session_as_asks_for_a_folder_and_goes_on_there(window, qtbot, monkeypatch, clip_in_odd_folder, tmp_path):
    asked = record_every_dialog(monkeypatch)
    named(window, qtbot, clip_in_odd_folder)
    window.menus.save_as_action.trigger()
    assert [(parent, title) for parent, title, _ in asked.folders] == [(window, "Save session as")]
    assert window.controller.saved_path is None  # nothing is written until a folder is chosen

    chosen = tmp_path / "my runs" / "dish, second try"
    asked.folders[0][2](chosen)
    controller = window.controller
    assert controller.run_folder == chosen and controller.saved_path == chosen / "session.json"
    assert (chosen / "session.json").is_file()
    assert window.statusBar().currentMessage() == "The session is saved in the run folder dish, second try."
    assert body(window, 1).folder_label.full_text == "dish, second try"  # panel 1 says where the files are
    assert body(window, 1).folder_label.toolTip() == str(chosen)
    assert asked.messages == []


def test_quit_closes_the_window(window, qtbot):
    show(window, qtbot)
    window.menus.quit_action.trigger()
    assert not window.isVisible()


def test_about_shows_the_versions(window, monkeypatch):
    asked = record_every_dialog(monkeypatch)
    window.menus.about_action.trigger()
    assert asked.messages == [(window, "info", about.about_text())]


# ---------------------------------------------------------------------------------------------
# The dialog that asks for a folder


def test_choose_folder_asks_for_one_folder_and_hands_it_on(window, qtbot, never_blocking, tmp_path):  # noqa: F811
    show(window, qtbot)
    chosen = []
    dialogs.choose_folder(window, "Save session as", chosen.append)
    (asking,) = window.findChildren(QFileDialog)
    assert asking.isVisible() and asking.isModal()
    assert asking.windowTitle() == "Save session as"
    assert asking.fileMode() == QFileDialog.FileMode.Directory
    assert asking.testOption(QFileDialog.Option.ShowDirsOnly)
    assert chosen == []
    folder = tmp_path / "runs"
    folder.mkdir()
    asking.setDirectory(str(tmp_path))
    asking.selectFile(str(folder))  # the user picks the folder and presses Choose
    asking.accept()
    assert chosen == [folder] and isinstance(chosen[0], Path)
    assert not asking.isVisible()


def test_choose_folder_hands_on_nothing_when_the_user_cancels(window, qtbot, never_blocking):  # noqa: F811
    show(window, qtbot)
    chosen = []
    dialogs.choose_folder(window, "Save session as", chosen.append)
    (asking,) = window.findChildren(QFileDialog)
    asking.reject()
    assert chosen == [] and not asking.isVisible()


# ---------------------------------------------------------------------------------------------
# Keys 1 to 9


@pytest.fixture
def nine_objects(window, qtbot, clip_in_odd_folder):
    """The window with a clip open (a copy in this test's folder) and nine objects, A to I, in
    panel 6. Returns its `ObjectsPanel`."""
    picture(window, qtbot, clip_in_odd_folder)
    panel = objects_panel(window)
    for _ in range(9):
        panel.add_object()
    assert [track.id for track in window.controller.session.tracks] == list("ABCDEFGHI")
    window.view.setFocus()
    QApplication.processEvents()
    return panel


def test_keys_1_to_9_select_objects_1_to_9(window, qtbot, nine_objects):
    panel = nine_objects
    assert panel.prompts.selected == "I"  # the object added last
    for number in (1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 1):
        qtbot.keyClick(window, Qt.Key(Qt.Key.Key_0.value + number))
        assert panel.prompts.selected == "ABCDEFGHI"[number - 1], number
        assert panel.table.currentRow() == number - 1  # and panel 6 shows which one


def test_a_key_without_an_object_changes_nothing(window, qtbot, disk_clip):
    show(window, qtbot)
    qtbot.keyClick(window, Qt.Key.Key_1)  # no video
    picture(window, qtbot, disk_clip)
    panel = objects_panel(window)
    qtbot.keyClick(window, Qt.Key.Key_1)  # no object
    assert panel.prompts.selected is None
    for _ in range(3):
        panel.add_object()
    qtbot.keyClick(window, Qt.Key.Key_2)
    assert panel.prompts.selected == "B"
    for key in (Qt.Key.Key_4, Qt.Key.Key_9, Qt.Key.Key_0):  # there are three objects; 0 is no key of an object
        qtbot.keyClick(window, key)
        assert panel.prompts.selected == "B"


def test_the_keys_work_wherever_the_focus_is_except_in_a_text_field(window, qtbot, nine_objects):
    panel = nine_objects
    for number, part in enumerate((window.view, window.navigation.slider, window.scroll, panel.table,
                                   panel.add_button), start=2):
        part.setFocus()
        QApplication.processEvents()
        qtbot.keyClick(part, Qt.Key(Qt.Key.Key_0.value + number))
        assert panel.prompts.selected == "ABCDEFGHI"[number - 1], part
    name, fps, frame_box = body(window, 1).name_edit, body(window, 2).fps_edit, window.navigation.frame_box
    for field in (name, fps, frame_box):
        field.setFocus()
        QApplication.processEvents()
        qtbot.keyClick(field, Qt.Key.Key_1)
        assert panel.prompts.selected == "F"  # a text field keeps the digit for itself
    assert name.text() == "1" and fps.text() == "1"
