"""Opening a video: the controller that owns the video and its session, the window's ways to open
one, and the two dialogs everything asks through (SPEC 8.1, 8.10, 10.1, 10.2; task C1).

Expected values come from the files and the spec: a synthetic clip's size, frame count and frame
rate are its scene's; the size in bytes and the SHA-256 are computed here from the file (the clips
are far smaller than 64 MiB, so the hash of the first 64 MiB is the hash of the file); a new
session's clip is frame 0 to the last frame at the default step of SPEC 8.10 (2). The texts are the
design note's: the empty video area, and the message for a video that cannot be opened.
"""

import hashlib
import os
import shutil
from pathlib import Path

import cv2
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox, QPushButton

import helpers
from gui_helpers import picture, record_dialogs, show
from outline_tracker import video
from outline_tracker.gui import dialogs
from outline_tracker.gui.session_controller import SessionController
from outline_tracker.session import Session

LEFT = Qt.MouseButton.LeftButton
NOT_OPENED = "The video could not be opened"
WHAT_TO_DO = ("Check that the file is on this computer, not only in a cloud folder, and that it is a "
              "_tracker.mp4 file. For a .MOV file, run “outline-tracker convert” on it first.")


@pytest.fixture
def controller(qtbot):
    """A controller of its own, with the signals it emits collected in `opened` and `changed`.
    Its video is released after the test."""
    made = SessionController()
    made.opened, made.changed = [], []
    made.video_opened.connect(lambda: made.opened.append(made.video_path))
    made.session_changed.connect(lambda: made.changed.append(True))
    yield made
    made.close()


@pytest.fixture
def no_video(tmp_path):
    """A text file with a video's name, in a folder of its own."""
    path = tmp_path / helpers.ODD_FOLDER / "notes_tracker.mp4"
    path.parent.mkdir()
    path.write_text("These are notes, not a video.\n", encoding="utf-8")
    return path


def own_copy(clip, tmp_path) -> Path:
    """A copy of the clip's file that this test may delete."""
    return Path(shutil.copyfile(clip.path, tmp_path / clip.path.name))


def is_released(source, path: Path) -> bool:
    """Whether a `FrameSource` has let go of its file: it reads no frame any more, and the file can
    be deleted (Windows refuses that while a file is open)."""
    with pytest.raises(ValueError, match="closed"):
        source.get(0)
    path.unlink()
    return not path.exists()


# ---------------------------------------------------------------------------------------------
# The controller


def test_a_new_controller_holds_nothing(controller):
    assert (controller.session, controller.video_path, controller.info) == (None, None, None)
    assert (controller.source, controller.run_folder) == (None, None)
    controller.close()  # nothing open: nothing happens


@pytest.mark.parametrize("name", ["dish_clip", "disk_clip", "closeup_clip"])
def test_open_video_makes_a_session_for_the_file(controller, request, name):
    clip = request.getfixturevalue(name)
    scene, path = clip.scene, clip.path
    controller.open_video(path)
    assert controller.opened == [path] and controller.changed == []
    assert controller.video_path == path and controller.run_folder is None

    info = controller.info
    assert (info.width, info.height, info.n_frames) == (*scene.size, scene.n_frames) == (320, 240, 120)
    session = controller.session
    assert isinstance(session, Session)
    assert (session.clip.start, session.clip.end, session.clip.step) == (0, 119, 2)
    block = session.video
    assert Path(block.abspath).name == path.name and os.path.isabs(block.abspath)
    assert Path(block.abspath) == Path(os.path.abspath(path))
    assert block.relpath is None  # relative to a run folder, and there is none yet
    assert block.size == path.stat().st_size
    assert block.sha256_first_64mib == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (block.width, block.height, block.n_frames) == (320, 240, 120)
    assert block.fps_container == pytest.approx(scene.fps)
    # nothing else is known yet
    assert session.student == "" and session.time.fps_true is None
    assert session.tracks == [] and session.calibration.stick is None and session.circle is None
    # the frames come from the open file, one at a time
    frame = controller.source.get(5)
    assert frame.shape == (240, 320, 3) and frame.dtype == "uint8"
    assert controller.source.info is info


def test_open_video_reads_a_file_in_a_folder_with_a_space_and_accents(controller, clip_in_odd_folder):
    path = clip_in_odd_folder.path
    assert helpers.ODD_FOLDER in str(path)
    controller.open_video(str(path))  # a path given as text
    assert controller.video_path == path
    assert controller.session.video.abspath == os.path.abspath(path)
    assert controller.source.get(119).shape == (240, 320, 3)


def test_a_second_open_video_replaces_the_first_and_releases_its_file(controller, dish_clip, disk_clip, tmp_path):
    first_path = own_copy(dish_clip, tmp_path)
    controller.open_video(first_path)
    first_source, first_session = controller.source, controller.session
    controller.open_video(disk_clip.path)
    assert controller.video_path == disk_clip.path
    assert controller.session is not first_session and controller.source is not first_source
    assert Path(controller.session.video.abspath).name == disk_clip.path.name
    assert controller.opened == [first_path, disk_clip.path]
    assert is_released(first_source, first_path)
    assert controller.source.get(0).shape == (240, 320, 3)  # the second one is open


def test_a_text_file_named_mp4_is_refused_and_nothing_changes(controller, dish_clip, no_video, tmp_path):
    with pytest.raises(ValueError) as refused:  # with nothing open
        controller.open_video(no_video)
    assert (controller.session, controller.video_path, controller.info, controller.source) == (None,) * 4
    message = str(refused.value)
    assert message == f"notes_tracker.mp4 could not be read. {WHAT_TO_DO}"
    assert helpers.ODD_FOLDER not in message and str(tmp_path) not in message  # the name, never the folder

    controller.open_video(dish_clip.path)
    before = (controller.session, controller.video_path, controller.info, controller.source)
    with pytest.raises(ValueError, match="notes_tracker.mp4 could not be read"):
        controller.open_video(no_video)
    assert all(now is then for now, then in zip(
        (controller.session, controller.video_path, controller.info, controller.source), before))
    assert controller.source.get(7).shape == (240, 320, 3)  # the video that was open still is
    assert controller.opened == [dish_clip.path]


def test_a_file_that_is_not_there_is_refused_in_the_same_words(controller, tmp_path):
    with pytest.raises(ValueError) as refused:
        controller.open_video(tmp_path / "gone_tracker.mp4")
    assert str(refused.value) == f"gone_tracker.mp4 could not be read. {WHAT_TO_DO}"
    assert controller.opened == []


def test_a_file_that_opencv_stumbles_over_is_refused_in_the_same_words(controller, dish_clip, monkeypatch):
    # OpenCV reports some damaged files with an error of its own kind, which is no OSError
    def stumble(path):
        raise cv2.error("the stream is damaged")

    controller.open_video(dish_clip.path)
    session = controller.session
    monkeypatch.setattr(video, "probe", stumble)
    with pytest.raises(ValueError) as refused:
        controller.open_video(dish_clip.path.with_name("damaged_tracker.mp4"))
    assert str(refused.value) == f"damaged_tracker.mp4 could not be read. {WHAT_TO_DO}"
    assert controller.session is session and controller.source.get(3).shape == (240, 320, 3)


def test_touch_says_that_the_session_changed(controller, dish_clip):
    controller.open_video(dish_clip.path)
    controller.session.student = "Ada"
    controller.touch()
    controller.touch()
    assert controller.changed == [True, True] and controller.opened == [dish_clip.path]


def test_close_releases_the_video_file(controller, dish_clip, tmp_path):
    path = own_copy(dish_clip, tmp_path)
    controller.open_video(path)
    source = controller.source
    controller.close()
    assert controller.source is None
    assert is_released(source, path)
    controller.close()  # again: nothing happens


# ---------------------------------------------------------------------------------------------
# The window: its parts, and its ways to open a video


def test_the_window_offers_its_controller_view_navigation_and_panels(window):
    assert isinstance(window.controller, SessionController)
    assert window.controller.session is None
    assert window.view.parent() is not None and window.navigation.parent() is not None
    assert len(window.panels) == 9


def test_the_empty_video_area_says_how_to_start_and_offers_open_video(window, qtbot):
    # what the video area's two lines of text were in the empty window, now around the button
    show(window, qtbot)
    title, button, hint = window.start_title, window.open_button, window.start_hint
    assert title.text() == "Open a video to start."
    assert button.text() == "Open video" and button.property("kind") == "primary"
    assert hint.text() == "Then follow panels 1 to 9 at the right."
    area = window.video_area
    assert all(part.isVisible() for part in (title, button, hint)) and not window.view.isVisible()
    parts = [part.geometry().translated(part.parentWidget().mapTo(area, QPoint(0, 0)))
             for part in (title, button, hint)]  # each part's place in the video area
    assert parts[0].bottom() < parts[1].top() and parts[1].bottom() < parts[2].top()  # in this order
    for part in parts:  # each in the middle of the area, left to right (QRect.center() is a whole px: not used)
        assert part.left() + part.width() / 2 == pytest.approx(area.width() / 2, abs=1)
    assert parts[1].top() + 16 == pytest.approx(area.height() / 2, abs=20)  # the button near the middle
    assert button.height() == 32  # a primary button


def test_the_open_video_button_asks_for_a_file_and_opens_the_one_chosen(window, qtbot, monkeypatch,
                                                                         clip_in_odd_folder):
    asked = record_dialogs(monkeypatch)
    show(window, qtbot)
    qtbot.mouseClick(window.open_button, LEFT)
    assert len(asked.files) == 1 and asked.messages == []
    parent, title, filters, on_chosen = asked.files[0]
    assert parent is window and title == "Open video"
    assert "*.mp4" in filters[0] and "*.mov" in filters[0]  # videos first
    assert filters[-1] == "All files (*)"
    assert window.controller.session is None  # nothing is opened until a file is chosen

    on_chosen(clip_in_odd_folder.path)
    assert window.controller.video_path == clip_in_odd_folder.path
    assert window.view.isVisible() and not window.open_button.isVisible()
    assert window.view.frame == 0
    assert window.view.hasFocus()  # no button has it: Space or Enter presses nothing by accident
    status = window.statusBar().currentMessage()
    assert "dish_tracker.mp4" in status and helpers.ODD_FOLDER not in status
    assert asked.messages == []


def test_the_file_menu_has_open_video_and_quit(window, qtbot, monkeypatch, dish_clip, tmp_path):
    asked = record_dialogs(monkeypatch)
    menus = [action for action in window.menuBar().actions() if action.menu() is not None]
    assert [action.text() for action in menus][:1] == ["File"]
    entries = [action for action in menus[0].menu().actions() if not action.isSeparator()]
    assert (entries[0].text(), entries[-1].text()) == ("Open video", "Quit")  # the first and the last entry
    assert (entries[0], entries[-1]) == (window.open_action, window.quit_action)
    assert window.open_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Open)  # the system's own

    window.open_action.trigger()
    assert [(parent, title) for parent, title, _, _ in asked.files] == [(window, "Open video")]

    path = own_copy(dish_clip, tmp_path)
    window.open_path(path)
    source = window.controller.source
    show(window, qtbot)
    window.quit_action.trigger()
    assert not window.isVisible()
    assert is_released(source, path)


def test_a_file_that_is_no_video_gives_a_message_and_the_open_video_stays(window, qtbot, monkeypatch,
                                                                         dish_clip, no_video, tmp_path):
    asked = record_dialogs(monkeypatch)
    view = picture(window, qtbot, dish_clip)
    window.navigation.slider.setValue(20)
    assert view.frame == 40
    session = window.controller.session

    window.open_path(no_video)
    assert len(asked.messages) == 1
    parent, kind, text = asked.messages[0]
    assert parent is window and kind == "problem"
    # what happened, then what to do; the file's name and never its folder
    assert text == f"{NOT_OPENED}\nnotes_tracker.mp4 could not be read. {WHAT_TO_DO}"
    assert helpers.ODD_FOLDER not in text and str(tmp_path) not in text
    status = window.statusBar().currentMessage()
    assert "notes_tracker.mp4" in status and helpers.ODD_FOLDER not in status
    # the video that was open stays as it was
    assert window.controller.session is session and window.controller.video_path == dish_clip.path
    assert view.isVisible() and view.frame == 40 and window.navigation.frame == 40
    window.navigation.slider.setValue(21)
    assert view.frame == 42


def test_a_file_that_is_no_video_leaves_the_empty_window_empty(window, qtbot, monkeypatch, no_video):
    asked = record_dialogs(monkeypatch)
    show(window, qtbot)
    window.open_path(no_video)
    assert [(kind, text.splitlines()[0]) for _, kind, text in asked.messages] == [("problem", NOT_OPENED)]
    assert window.controller.session is None
    assert window.open_button.isVisible() and not window.view.isVisible()
    assert not window.navigation.isEnabled()


def test_closing_tells_the_parts_first_while_the_video_is_still_open(window, qtbot, dish_clip):
    # a panel saves and a worker stops on `closing`: both may still need the session and the video
    window.open_path(dish_clip.path)
    source, seen = window.controller.source, []
    window.closing.connect(lambda: seen.append((window.controller.session is not None, source.get(0).shape)))
    assert window.close()
    assert seen == [(True, (240, 320, 3))]  # told once, before the video was released (frame: rows, columns, RGB)
    with pytest.raises(ValueError):
        source.get(0)
    assert window.close() and len(seen) == 1  # a second close (the test's own teardown does one) tells nobody


def test_closing_the_window_releases_the_video(window, qtbot, dish_clip, tmp_path):
    path = own_copy(dish_clip, tmp_path)
    window.open_path(path)
    show(window, qtbot)
    source = window.controller.source
    assert source.get(0).shape == (240, 320, 3)  # open and readable
    assert window.close()
    assert is_released(source, path)


# ---------------------------------------------------------------------------------------------
# The dialogs themselves


@pytest.fixture
def never_blocking(monkeypatch):
    """Make `exec`, which would wait for the user and never return here, an error."""
    def blocked(*_):
        raise AssertionError("A dialog was run with exec(): it blocks the window. Use open().")

    for kind in (QDialog, QMessageBox, QFileDialog):
        monkeypatch.setattr(kind, "exec", blocked)


@pytest.mark.parametrize("kind, icon", [("info", QMessageBox.Icon.Information),
                                        ("warning", QMessageBox.Icon.Warning),
                                        ("problem", QMessageBox.Icon.Critical)])
def test_message_opens_a_dialog_that_says_what_happened_and_what_to_do(window, qtbot, never_blocking, kind, icon):
    show(window, qtbot)
    dialogs.message(window, kind, "The video could not be opened\nIt could not be read. Check the file.")
    (box,) = window.findChildren(QMessageBox)
    assert box.isVisible() and box.isModal()
    assert box.text() == "The video could not be opened"
    assert box.informativeText() == "It could not be read. Check the file."
    assert box.icon() == icon
    buttons = box.findChildren(QPushButton)
    assert [button.text() for button in buttons] == ["Close"]
    qtbot.mouseClick(buttons[0], LEFT)
    assert not box.isVisible()


def test_message_takes_a_text_of_one_line_and_refuses_an_unknown_kind(window, qtbot, never_blocking):
    show(window, qtbot)
    dialogs.message(window, "info", "The session was saved.")
    (box,) = window.findChildren(QMessageBox)
    assert (box.text(), box.informativeText()) == ("The session was saved.", "")
    box.close()
    with pytest.raises(ValueError, match="oops"):
        dialogs.message(window, "oops", "Something.")


def test_open_file_asks_for_one_existing_file_and_hands_it_on(window, qtbot, never_blocking, dish_clip):
    show(window, qtbot)
    chosen = []
    dialogs.open_file(window, "Open video", ["Videos (*.mp4 *.mov)", "All files (*)"], chosen.append)
    (asking,) = window.findChildren(QFileDialog)
    assert asking.isVisible() and asking.isModal()
    assert asking.windowTitle() == "Open video"
    assert asking.nameFilters() == ["Videos (*.mp4 *.mov)", "All files (*)"]
    assert asking.fileMode() == QFileDialog.FileMode.ExistingFile
    assert asking.acceptMode() == QFileDialog.AcceptMode.AcceptOpen
    assert chosen == []
    asking.selectFile(str(dish_clip.path))  # the user picks the file and presses Open
    asking.accept()
    assert chosen == [dish_clip.path] and isinstance(chosen[0], Path)
    assert not asking.isVisible()


def test_open_file_hands_on_nothing_when_the_user_cancels(window, qtbot, never_blocking):
    show(window, qtbot)
    chosen = []
    dialogs.open_file(window, "Open video", ["All files (*)"], chosen.append)
    (asking,) = window.findChildren(QFileDialog)
    asking.reject()
    assert chosen == [] and not asking.isVisible()
