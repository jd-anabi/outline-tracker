"""Saving the session and opening it again (task C2; SPEC 2, 8.1, 8.10, 10.2).

Expected values come from the spec: session.json is in the run folder
`<video folder>/<video stem>_outline_<student>` and names its video by a path relative to that
folder (SPEC 8.1), so the video beside the run folder is `../<video name>`; it is written 0.75 s
after the last change, on the platform's Save key, when the window closes and through `save_now`;
a session of another schema version gives a plain message (SPEC 8.10). The debounce is tested by
its timer (interval, single shot, restarted by a change) and by waiting for the file, never by a
fixed sleep. Every clip is a copy in the test's own folder.
"""

import shutil
import threading
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence

import helpers
from gui_helpers import record_dialogs, show
from outline_tracker import fileio
from outline_tracker.gui.session_controller import SessionController
from outline_tracker.session import Session, VideoNotFoundError
from session_helpers import (FOLDER_OF_NAME, NAME, body, hd_clip, hd_scene_clip, own_settings,  # noqa: F401 (fixtures)
                             read_json, second_spelling, settle, type_into, wait_for_check)

NOT_WRITTEN = "A file could not be written"
NOT_OPENED = "The session could not be opened"
HELD = "The folder {folder} holds a session already. Open it with Open session, or type another name."


@pytest.fixture
def named(window, qtbot, clip_in_odd_folder):
    """The window with the dish clip (a copy in this test's folder) open and the name typed; returns
    (window, the clip's path, the run folder's session.json, which is not written yet)."""
    path = clip_in_odd_folder.path
    window.open_path(path)
    type_into(qtbot, body(window, 1).name_edit, NAME)
    return window, path, path.parent / FOLDER_OF_NAME / "session.json"


@pytest.fixture
def controller(qtbot):
    """A controller of its own, without a window. Its video is released after the test."""
    made = SessionController()
    yield made
    made.close()


# ---------------------------------------------------------------------------------------------
# Writing session.json


def test_a_change_is_saved_by_a_single_shot_timer_of_750_ms_that_a_new_change_restarts(named, qtbot, monkeypatch):
    window, path, target = named
    controller, timer = window.controller, window.controller.save_timer
    threads = []
    real = Session.save
    monkeypatch.setattr(Session, "save", lambda session, where: (
        threads.append(threading.current_thread() is threading.main_thread()), real(session, where))[1])
    assert timer.interval() == 750 and timer.isSingleShot()
    assert timer.isActive() and not target.exists()  # the name was a change: the save is due, not done

    qtbot.waitUntil(lambda: timer.remainingTime() < 400)  # nearly half of the 750 ms have gone by
    body(window, 1).step_box.setValue(4)  # another change: the wait starts again
    # (Qt may move a timer of this kind by 5% of its interval: 600 ms is far from both 400 and 750)
    assert timer.isActive() and timer.remainingTime() > 600 and not target.exists()

    qtbot.waitUntil(target.is_file, timeout=10_000)
    assert not timer.isActive() and threads == [True]  # written once, from the GUI thread
    saved = read_json(target)
    assert saved["format"] == "outline-tracker-session" and saved["schema_version"] == 1
    assert saved["student"] == NAME and saved["clip"] == {"start": 0, "end": 119, "step": 4}
    assert saved["video"]["relpath"] == "../dish_tracker.mp4"  # relative to the run folder (SPEC 8.1)
    assert Path(saved["video"]["abspath"]) == path and saved["video"]["size"] == path.stat().st_size
    assert controller.saved_path == target and controller.save_problem is None
    assert sorted(entry.name for entry in target.parent.iterdir()) == ["session.json"]  # no file left half written


def test_the_timers_own_signal_writes_the_file(named):
    window, _, target = named
    window.controller.save_timer.timeout.emit()
    assert target.is_file() and read_json(target)["student"] == NAME


def test_save_now_writes_at_once_and_returns_the_file(named):
    window, _, target = named
    controller = window.controller
    seen = []
    controller.saved.connect(lambda: seen.append(controller.saved_path))
    assert controller.save_now() == target and target.is_file()
    assert not controller.save_timer.isActive() and seen == [target]  # nothing is due any more
    assert Session.load(target).to_json() == controller.session.to_json()


def test_nothing_is_saved_while_the_name_is_empty(window, qtbot, clip_in_odd_folder):
    path = clip_in_odd_folder.path
    window.open_path(path)
    controller = window.controller
    body(window, 1).step_box.setValue(3)
    type_into(qtbot, body(window, 2).fps_edit, "240")
    assert controller.run_folder is None and not controller.save_timer.isActive()
    assert controller.save_now() is None and controller.saved_path is None
    assert window.close()
    assert [entry.name for entry in path.parent.iterdir()] == [path.name]  # no folder, no file


def test_the_save_key_saves_what_is_being_typed(named, qtbot):
    window, _, target = named
    show(window, qtbot)
    panel = body(window, 1)
    type_into(qtbot, body(window, 2).fps_edit, "239.6", enter=False)  # typed, and Enter not pressed
    assert panel.save_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Save)
    qtbot.keyClick(window, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)  # Qt's Ctrl is the Mac's Cmd
    assert target.is_file()
    assert read_json(target)["time"] == {"fps_true": 239.6, "source": "typed", "manifest_path": None,
                                         "stopwatch": None}
    assert FOLDER_OF_NAME in window.statusBar().currentMessage()


def test_the_save_key_without_a_name_says_what_is_missing(window, qtbot, clip_in_odd_folder):
    window.open_path(clip_in_odd_folder.path)
    show(window, qtbot)
    qtbot.keyClick(window, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
    assert window.statusBar().currentMessage() == "Type your name in panel 1 first."
    assert [entry.name for entry in clip_in_odd_folder.path.parent.iterdir()] == ["dish_tracker.mp4"]


def test_closing_the_window_saves_also_a_name_that_was_typed_and_not_entered(window, qtbot, clip_in_odd_folder):
    path = clip_in_odd_folder.path
    window.open_path(path)
    panel = body(window, 1)
    panel.end_box.setValue(99)
    type_into(qtbot, panel.name_edit, NAME, enter=False)
    assert window.controller.session.student == ""
    assert window.close()
    saved = read_json(path.parent / FOLDER_OF_NAME / "session.json")
    assert saved["student"] == NAME and saved["clip"] == {"start": 0, "end": 99, "step": 2}
    path.unlink()  # the video was released after the save (Windows refuses this while a file is open)


def test_a_controller_that_is_closed_writes_nothing_later(controller, clip_in_odd_folder):
    path = clip_in_odd_folder.path
    controller.open_video(path)
    controller.session.student = NAME
    controller.touch()
    assert controller.run_folder == path.parent / FOLDER_OF_NAME and controller.save_timer.isActive()
    controller.close()
    assert not controller.save_timer.isActive() and not controller.run_folder.exists()


def test_opening_another_video_first_saves_what_was_due(named, hd_clip):
    window, _, target = named
    assert window.controller.save_timer.isActive() and not target.exists()
    window.open_path(hd_clip.path)
    assert read_json(target)["student"] == NAME
    # the new video is another session: the name stays in its field, and names the new run folder
    assert window.controller.session.student == NAME
    assert window.controller.run_folder == hd_clip.path.parent / "hd_tracker_outline_Ada_Lovelace"


# ---------------------------------------------------------------------------------------------
# The run folder follows the name; an earlier session is never written over


def test_after_a_new_name_the_session_is_saved_in_the_new_names_folder_and_the_old_one_is_left(named, qtbot):
    window, path, first = named
    controller, panel = window.controller, body(window, 1)
    assert controller.save_now() == first
    before = first.read_bytes()

    type_into(qtbot, panel.name_edit, "Grace")
    second = path.parent / "dish_tracker_outline_Grace" / "session.json"
    assert controller.run_folder == second.parent
    assert panel.folder_label.full_text == "dish_tracker_outline_Grace"  # where the files are now
    assert controller.save_now() == second
    assert read_json(second)["student"] == "Grace" and read_json(second)["video"]["relpath"] == "../dish_tracker.mp4"
    assert first.read_bytes() == before  # the old folder is left as it is

    type_into(qtbot, panel.name_edit, NAME)  # back to the first name: that folder is this window's own
    assert controller.save_now() == first and controller.save_problem is None


def test_a_session_that_is_in_the_run_folder_already_is_not_written_over(window, qtbot, monkeypatch,
                                                                         clip_in_odd_folder):
    # the same video opened again and the same name typed: the earlier work must survive
    asked = record_dialogs(monkeypatch)
    path = clip_in_odd_folder.path
    earlier = path.parent / FOLDER_OF_NAME / "session.json"
    earlier.parent.mkdir()
    earlier.write_text('{"earlier": "work"}\n', encoding="utf-8")
    window.open_path(path)
    panel, controller = body(window, 1), window.controller
    type_into(qtbot, panel.name_edit, NAME)
    assert controller.save_now() is None and controller.save_now() is None
    assert earlier.read_text(encoding="utf-8") == '{"earlier": "work"}\n'
    problem = (f"The folder {FOLDER_OF_NAME} holds a session already. Open it with Open session, or type "
               "another name.")
    assert controller.save_problem == problem
    assert not panel.save_message.isHidden() and panel.save_message.text() == problem
    assert panel.save_message.property("kind") == "problem" and window.panels[0].state == "attention"
    assert [(parent, kind, text) for parent, kind, text in asked.messages] == [
        (window, "problem", f"{NOT_WRITTEN}\n{problem}")]  # told once, however often the save is tried

    type_into(qtbot, panel.name_edit, "Grace")  # another name: saved, and the message goes
    assert controller.save_now() == path.parent / "dish_tracker_outline_Grace" / "session.json"
    assert panel.save_message.isHidden() and len(asked.messages) == 1


@pytest.mark.parametrize("another_video_between", [False, True], ids=["at once", "after another video"])
def test_the_same_video_opened_again_does_not_write_over_its_saved_session(named, qtbot, monkeypatch, hd_clip,
                                                                           another_video_between):
    # Open video starts a new session (default clip, no fps_true) whose run folder is the saved session's
    window, path, target = named
    asked = record_dialogs(monkeypatch)
    controller = window.controller
    body(window, 1).step_box.setValue(5)
    type_into(qtbot, body(window, 2).fps_edit, "239.6")
    assert controller.save_now() == target
    before = target.read_bytes()
    assert read_json(target)["clip"]["step"] == 5 and read_json(target)["time"]["fps_true"] == 239.6

    if another_video_between:
        window.open_path(hd_clip.path)
    window.open_path(path)
    assert controller.session.clip.step == 2 and controller.session.time.fps_true is None  # a new session
    assert controller.run_folder == target.parent and controller.save_timer.isActive()
    controller.save_timer.timeout.emit()  # the delayed save of the new session
    assert target.read_bytes() == before
    problem = HELD.format(folder=FOLDER_OF_NAME)
    assert controller.save_problem == problem and controller.saved_path is None
    assert controller.save_now() is None and target.read_bytes() == before  # the Save key, closing: the same
    assert [(parent, kind, text) for parent, kind, text in asked.messages] == [
        (window, "problem", f"{NOT_WRITTEN}\n{problem}")]

    window.open_path(target)  # what the message says to do: Open session, and the work goes on
    assert controller.session.clip.step == 5 and controller.session.time.fps_true == 239.6
    assert controller.save_problem is None and body(window, 1).save_message.isHidden()
    body(window, 1).step_box.setValue(6)
    assert controller.save_now() == target and read_json(target)["clip"]["step"] == 6


def test_a_new_name_does_not_lead_a_new_session_over_an_earlier_one_of_this_window(named, qtbot, monkeypatch):
    window, path, first = named
    controller, panel = window.controller, body(window, 1)
    panel.step_box.setValue(5)
    assert controller.save_now() == first
    type_into(qtbot, panel.name_edit, "Grace")
    second = path.parent / "dish_tracker_outline_Grace" / "session.json"
    assert controller.save_now() == second
    kept = {first: first.read_bytes(), second: second.read_bytes()}

    asked = record_dialogs(monkeypatch)
    window.open_path(path)  # a new session, named Grace as the field still says
    assert controller.session.student == "Grace" and controller.session.clip.step == 2
    controller.save_timer.timeout.emit()
    assert controller.save_problem == HELD.format(folder="dish_tracker_outline_Grace")
    type_into(qtbot, panel.name_edit, NAME)  # the first name's folder holds the first session
    assert controller.save_timer.isActive()
    controller.save_timer.timeout.emit()
    assert controller.save_problem == HELD.format(folder=FOLDER_OF_NAME)

    type_into(qtbot, panel.name_edit, "Mary")  # a folder of its own: the new session is saved there
    assert controller.save_now() == path.parent / "dish_tracker_outline_Mary" / "session.json"
    type_into(qtbot, panel.name_edit, NAME)  # having a file of its own does not make the others its own
    assert controller.save_now() is None and controller.save_problem == HELD.format(folder=FOLDER_OF_NAME)
    assert {file: file.read_bytes() for file in kept} == kept
    assert [text.split("\n", 1)[0] for _, _, text in asked.messages] == [NOT_WRITTEN] * 3


@pytest.mark.parametrize("through_open_session", [False, True], ids=["after a save", "after Open session"])
def test_a_name_typed_again_in_another_case_stays_in_this_sessions_folder(window, qtbot, monkeypatch,
                                                                          clip_in_odd_folder, through_open_session):
    # macOS and Windows know a folder by every case of its name: the folder of "ada" is the folder of "Ada",
    # and the session.json in it is this session's own under both spellings
    asked = record_dialogs(monkeypatch)
    path = clip_in_odd_folder.path
    window.open_path(path)
    panel, controller = body(window, 1), window.controller
    type_into(qtbot, panel.name_edit, "ada")
    first = path.parent / "dish_tracker_outline_ada" / "session.json"
    assert controller.save_now() == first
    again = second_spelling(first.parent, "dish_tracker_outline_Ada") / "session.json"
    assert str(again) != str(first) and again.samefile(first)  # one file, two spellings
    if through_open_session:
        window.open_path(first)

    type_into(qtbot, panel.name_edit, "Ada")  # the name, corrected
    assert controller.run_folder == again.parent
    assert controller.save_now() == again and controller.save_problem is None
    assert read_json(first)["student"] == "Ada"  # the one file holds the corrected name
    assert panel.save_message.isHidden() and asked.messages == []
    panel.step_box.setValue(7)  # and the session goes on being saved there
    assert controller.save_now() == again and read_json(first)["clip"]["step"] == 7


def test_a_locked_session_file_is_saved_beside_it_and_the_user_is_told_once(named, monkeypatch):
    window, _, target = named
    asked = record_dialogs(monkeypatch)
    controller, panel = window.controller, body(window, 1)
    assert controller.save_now() == target
    real, locked = fileio.os.replace, [True]

    def replace(source, destination):
        if locked[0] and Path(destination).name == "session.json":
            raise PermissionError("the file is open in another program")  # what Windows says of a locked file
        return real(source, destination)

    monkeypatch.setattr(fileio.os, "replace", replace)
    monkeypatch.setattr(fileio.time, "sleep", lambda seconds: None)  # the 5 s of retries, without the wait
    body(window, 1).step_box.setValue(6)
    beside = target.with_name("session.new.json")
    assert controller.save_now() == beside and controller.save_now() == beside
    assert read_json(beside)["clip"]["step"] == 6 and read_json(target)["clip"]["step"] == 2
    told = ("session.json is open in another program. Close it there. Until then the session is saved as "
            "session.new.json in the run folder.")
    assert [(parent, kind, text) for parent, kind, text in asked.messages] == [
        (window, "problem", f"{NOT_WRITTEN}\n{told}")]
    assert not panel.save_message.isHidden() and panel.save_message.text() == told

    locked[0] = False  # the other program let go
    assert controller.save_now() == target and read_json(target)["clip"]["step"] == 6
    assert panel.save_message.isHidden() and len(asked.messages) == 1


def test_a_folder_that_cannot_be_written_is_told_once_and_never_raised(named, monkeypatch):
    window, _, target = named
    asked = record_dialogs(monkeypatch)
    controller = window.controller

    def refuse(session, where):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(Session, "save", refuse)
    assert controller.save_now() is None
    controller.save_timer.timeout.emit()  # the timer's save does not raise either
    problem = (f"session.json could not be written to the folder {FOLDER_OF_NAME}. Check that the folder is not "
               "read-only and that the disk is not full. Nothing is saved until this works.")
    assert controller.save_problem == problem and not target.exists()
    assert [(kind, text) for _, kind, text in asked.messages] == [("problem", f"{NOT_WRITTEN}\n{problem}")]
    assert helpers.ODD_FOLDER not in problem


# ---------------------------------------------------------------------------------------------
# Opening a session


def set_up(window, qtbot):
    """Fill panels 1 and 2 of a window that has the dish clip open: the name, a clip, fps_true."""
    first, second = body(window, 1), body(window, 2)
    type_into(qtbot, first.name_edit, NAME)
    first.start_box.setValue(10)
    first.end_box.setValue(50)
    first.step_box.setValue(5)
    type_into(qtbot, second.fps_edit, "239.6")


def test_opening_the_saved_session_brings_every_field_back(window, qtbot, clip_in_odd_folder, hd_clip):
    path = clip_in_odd_folder.path
    window.open_path(path)
    set_up(window, qtbot)
    target = window.controller.save_now()
    written = read_json(target)

    window.open_path(hd_clip.path)  # another video: nothing of the first session is in the window any more
    first, second, controller = body(window, 1), body(window, 2), window.controller
    type_into(qtbot, first.name_edit, "")
    assert controller.session.clip.end == 5 and controller.session.time.fps_true is None

    show(window, qtbot)
    window.open_path(target)
    session = controller.session
    assert controller.video_path == path and controller.run_folder == target.parent
    assert session.to_json() == written
    assert first.name_edit.text() == NAME and first.file_label.full_text == "dish_tracker.mp4"
    assert (first.start_box.value(), first.end_box.value(), first.step_box.value()) == (10, 50, 5)
    assert first.folder_label.full_text == FOLDER_OF_NAME and first.name_message.isHidden()
    assert second.fps_edit.text() == "239.6" and second.source_label.text() == "typed in"
    assert window.panels[1].state == "done"
    assert window.panels[1].hint.text() == "fps_true = 239.60 fps (typed in)."
    bar = window.navigation
    assert bar.slider.maximum() == 8 and bar.frame == 10 and window.view.frame == 10  # the grid 10, 15, ..., 50
    assert bar.time_label.text() == "t = 0.042 s"  # 10 / 239.6 s
    assert window.statusBar().currentMessage() == "session.json is open."
    assert not controller.save_timer.isActive()  # opened as it was saved: nothing to write
    wait_for_check(qtbot, window)
    assert window.panels[0].state == "attention"  # the dish clip's warning is back as well

    first.end_box.setValue(40)  # and it is saved where it was opened
    assert controller.save_now() == target and read_json(target)["clip"]["end"] == 40


def test_a_run_folder_moved_with_its_video_opens_through_the_relative_path(window, qtbot, monkeypatch,
                                                                           clip_in_odd_folder, tmp_path):
    asked = record_dialogs(monkeypatch)
    window.open_path(clip_in_odd_folder.path)
    set_up(window, qtbot)
    target = window.controller.save_now()
    settle(qtbot, window)
    window.controller.close()  # release the video: Windows moves no open file
    moved = tmp_path / "moved to here"
    shutil.move(clip_in_odd_folder.path.parent, moved)

    window.open_path(moved / target.parent.name / "session.json")
    controller = window.controller
    assert asked.files == [] and asked.messages == []
    assert controller.video_path == moved / "dish_tracker.mp4" and controller.source.get(10).shape == (240, 320, 3)
    assert controller.run_folder == moved / FOLDER_OF_NAME
    assert Path(controller.session.video.abspath) == moved / "dish_tracker.mp4"
    assert controller.save_timer.isActive()  # the video's new place is to be written


def test_a_video_that_moved_is_asked_for_and_another_file_is_refused(window, qtbot, monkeypatch, clip_in_odd_folder,
                                                                    hd_clip, tmp_path):
    asked = record_dialogs(monkeypatch)
    path = clip_in_odd_folder.path
    window.open_path(path)
    set_up(window, qtbot)
    target = window.controller.save_now()
    window.open_path(hd_clip.path)  # the window is on another video now
    settle(qtbot, window)
    elsewhere = tmp_path / "elsewhere" / "dish_tracker.mp4"
    elsewhere.parent.mkdir()
    shutil.move(path, elsewhere)

    window.open_path(target)
    controller = window.controller
    assert controller.video_path == hd_clip.path  # nothing is opened until the video is found
    assert window.statusBar().currentMessage() == "The video dish_tracker.mp4 was not found. Choose where it is now."
    (parent, title, filters, on_chosen), = asked.files
    assert parent is window and title == "Find the video of this session" and "*.mp4" in filters[0]
    assert asked.messages == []

    on_chosen(hd_clip.path)  # another file than the session's video
    assert controller.video_path == hd_clip.path and controller.run_folder != target.parent
    (_, kind, text), = asked.messages
    assert kind == "problem" and text.splitlines()[0] == NOT_OPENED
    assert "hd_tracker.mp4 is not the video of this session" in text
    assert str(tmp_path) not in text  # the file's name, never its folder

    on_chosen(elsewhere)
    assert controller.video_path == elsewhere and controller.session.student == NAME
    assert controller.run_folder == target.parent and len(asked.messages) == 1
    assert controller.session.video.relpath == "../../elsewhere/dish_tracker.mp4"
    assert controller.save_now() == target and read_json(target)["video"]["relpath"] == "../../elsewhere/dish_tracker.mp4"


def test_the_controller_asks_for_a_video_that_is_at_none_of_the_sessions_places(controller, named, qtbot):
    window, path, _ = named
    target = window.controller.save_now()
    settle(qtbot, window)
    window.controller.close()
    path.rename(path.with_name("renamed_tracker.mp4"))
    with pytest.raises(VideoNotFoundError) as missing:
        controller.open_session(target)
    assert str(missing.value) == "The video dish_tracker.mp4 was not found. Choose where it is now."
    assert controller.session is None
    controller.open_session(target, path.with_name("renamed_tracker.mp4"))  # the user's answer
    assert controller.session.student == NAME and controller.video_path.name == "renamed_tracker.mp4"


def test_a_session_of_another_schema_version_gives_its_plain_message(window, monkeypatch, named):
    asked = record_dialogs(monkeypatch)
    _, path, _ = named
    target = window.controller.save_now()
    newer = target.with_name("from the future.json")
    newer.write_text(target.read_text(encoding="utf-8").replace('"schema_version": 1', '"schema_version": 2'),
                     encoding="utf-8")
    session = window.controller.session
    window.open_path(newer)
    (parent, kind, text), = asked.messages
    assert parent is window and kind == "problem"
    heading, rest = text.split("\n", 1)
    assert heading == NOT_OPENED
    assert rest.startswith("from the future.json has session format version 2, but this outline-tracker")
    assert "reads only version 1" in rest and "update yours" in rest
    assert helpers.ODD_FOLDER not in text and str(path.parent) not in text
    assert window.controller.session is session and window.controller.video_path == path  # what was open stays
    assert window.statusBar().currentMessage() == "from the future.json could not be opened."


@pytest.mark.parametrize("content", ["These are notes, not a session.\n", '{"format": "something else"}\n', "[1, 2]\n"])
def test_a_file_that_is_no_session_gives_a_message_and_nothing_changes(window, monkeypatch, tmp_path, content):
    asked = record_dialogs(monkeypatch)
    path = tmp_path / helpers.ODD_FOLDER / "notes.json"
    path.parent.mkdir()
    path.write_text(content, encoding="utf-8")
    window.open_path(path)
    assert [(kind, text) for _, kind, text in asked.messages] == [("problem", (
        f"{NOT_OPENED}\nnotes.json is not a session file of Outline Tracker. Choose the session.json of a run folder."))]
    assert window.controller.session is None and asked.files == []


def test_a_session_file_that_is_not_there_gives_a_message_and_names_only_the_file(window, monkeypatch, tmp_path):
    asked = record_dialogs(monkeypatch)
    window.open_path(tmp_path / helpers.ODD_FOLDER / "session.json")
    status = window.statusBar().currentMessage()
    assert status == "session.json could not be opened."
    (parent, kind, text), = asked.messages
    assert parent is window and kind == "problem"
    assert text == (f"{NOT_OPENED}\nsession.json could not be read. Check that the file is on this computer, not "
                    "only in a cloud folder.")
    assert helpers.ODD_FOLDER not in text and str(tmp_path) not in text
    assert window.controller.session is None


def test_the_sessions_video_that_cannot_be_decoded_is_refused_like_any_video(window, monkeypatch, named):
    asked = record_dialogs(monkeypatch)
    _, path, _ = named
    target = window.controller.save_now()
    session = window.controller.session
    from outline_tracker.gui import session_controller

    def stumble(video_path):
        raise OSError("the stream is damaged")

    monkeypatch.setattr(session_controller, "FrameSource", stumble)
    window.open_path(target)
    (_, kind, text), = asked.messages
    assert kind == "problem" and text.startswith(f"{NOT_OPENED}\ndish_tracker.mp4 could not be read. ")
    assert window.controller.session is session
