"""Opening a saved session (task C2; SPEC 2, 8.1, 8.10): its fields come back, and a video that moved is asked for.

A session that cannot be opened gives a plain message. Saving it is in tests/gui/test_session_saving.py. The
fixtures `named` and `controller` are in tests/gui/conftest.py, as both files use them.

Expected values come from the spec: session.json is in the run folder
`<video folder>/<video stem>_outline_<student>` and names its video by a path relative to that
folder (SPEC 8.1), so the video beside the run folder is `../<video name>`; a session of another
schema version gives a plain message (SPEC 8.10). Every clip is a copy in the test's own folder.
"""

import shutil
from pathlib import Path

import pytest

import helpers
from gui_helpers import record_dialogs, show
from outline_tracker.core.session import VideoNotFoundError
from session_helpers import FOLDER_OF_NAME, NAME, body, read_json, settle, type_into, wait_for_check

pytestmark = pytest.mark.usefixtures("own_settings")  # every test here has a settings folder of its own

NOT_OPENED = "The session could not be opened"


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
