"""The run folder when the name changes, and Save session as (SPEC 8.1, 8.10, 10.1; task C8a).

Expected values come from the spec and the task: the default run folder is
`<video folder>/<video stem>_outline_<student>` (SPEC 8.1); the run folder is self-contained, so
session.json names its video by a path relative to the folder it is in; results.npz is the store
every other file is rebuilt from (SPEC 8.12), so the folder that holds it is not left behind when
the name is typed again, and it is taken along, as a copy, when the session goes on in another
folder. Every clip is a copy in the test's own folder; results.npz is an empty store written by
`ResultsStore.save`, a real file of that kind.
"""

import os
import shutil

import pytest

from finish_helpers import named, record_every_dialog
from outline_tracker.gui import session_controller
from outline_tracker.results import ResultsStore
from session_helpers import FOLDER_OF_NAME, body, read_json, type_into

NOT_WRITTEN = "A file could not be written"
GRACE = "dish_tracker_outline_Grace"
HELD = "The folder {folder} holds a session already. Choose another folder, or open that session with Open session."
TRACKER_FILES = ("The folder {folder} holds other .csv or .txt files. It looks like a folder of Tracker files, and "
                 "the run's files are not written into one. Choose another folder.")
NOT_COPIED = ("results.npz could not be copied to the folder {folder}. Check that the folder is not read-only and "
              "that the disk is not full. The session stays in the run folder {kept}.")


@pytest.fixture
def saved(window, qtbot, clip_in_odd_folder):
    """The window with the dish clip (a copy) open, the name typed and the session saved. Returns
    (window, the clip's path, the run folder)."""
    path = named(window, qtbot, clip_in_odd_folder)
    folder = path.parent / FOLDER_OF_NAME
    assert window.controller.save_now() == folder / "session.json"
    return window, path, folder


def with_results(folder) -> bytes:
    """Put a results.npz into `folder`, as a run does. Returns the file's bytes."""
    return ResultsStore().save(folder / "results.npz").read_bytes()


def files_of(folder) -> list[str]:
    return sorted(entry.name for entry in folder.iterdir())


# ---------------------------------------------------------------------------------------------
# A new name


def test_with_results_in_the_run_folder_a_new_name_keeps_the_folder(saved, qtbot):
    window, path, folder = saved
    controller, panel = window.controller, body(window, 1)
    results = with_results(folder)
    type_into(qtbot, panel.name_edit, "Grace")
    assert controller.session.student == "Grace" and controller.student == "Grace"  # the name does change
    assert controller.run_folder == folder  # and the folder with the results is not left
    assert controller.save_now() == folder / "session.json" and controller.save_problem is None
    assert read_json(folder / "session.json")["student"] == "Grace"
    assert panel.folder_label.full_text == FOLDER_OF_NAME and panel.folder_label.toolTip() == str(folder)
    assert not (path.parent / GRACE).exists()  # no second folder beside the video
    assert (folder / "results.npz").read_bytes() == results
    assert controller.refusal("track") is None and controller.refusal("export") is None


def test_with_results_a_name_that_is_cleared_and_typed_again_stays_in_the_folder_too(saved, qtbot):
    window, path, folder = saved
    controller, name = window.controller, body(window, 1).name_edit
    with_results(folder)
    type_into(qtbot, name, "")  # the field is emptied first, and the new name typed after that
    assert controller.run_folder == folder and controller.session.student == ""
    assert controller.refusal("track") == controller.refusal("export") == "Type your name in panel 1 first."
    type_into(qtbot, name, "Grace")
    assert controller.run_folder == folder and controller.save_now() == folder / "session.json"
    assert read_json(folder / "session.json")["student"] == "Grace"
    assert not (path.parent / GRACE).exists()


def test_without_results_a_new_name_moves_to_its_default_folder(saved, qtbot):
    window, path, folder = saved
    controller, panel = window.controller, body(window, 1)
    before = (folder / "session.json").read_bytes()
    type_into(qtbot, panel.name_edit, "Grace")
    assert controller.run_folder == path.parent / GRACE  # today's rule
    assert controller.save_now() == path.parent / GRACE / "session.json"
    assert panel.folder_label.full_text == GRACE
    assert (folder / "session.json").read_bytes() == before  # the old folder is left as it is


def test_results_that_arrive_later_hold_the_folder_from_then_on(saved, qtbot):
    window, path, folder = saved
    controller = window.controller
    type_into(qtbot, body(window, 1).name_edit, "Grace")  # no results yet: the session moves
    moved = path.parent / GRACE
    assert controller.save_now() == moved / "session.json"
    with_results(moved)  # tracked under this name
    type_into(qtbot, body(window, 1).name_edit, "Grace Hopper")
    assert controller.run_folder == moved and controller.session.student == "Grace Hopper"
    assert controller.save_now() == moved / "session.json"
    assert not (path.parent / "dish_tracker_outline_Grace_Hopper").exists()


def test_a_session_opened_from_a_folder_with_results_stays_there_under_a_new_name(window, qtbot, saved):
    _, path, folder = saved
    with_results(folder)
    window.open_path(folder / "session.json")
    type_into(qtbot, body(window, 1).name_edit, "Grace")
    assert window.controller.run_folder == folder
    assert window.controller.save_now() == folder / "session.json"


# ---------------------------------------------------------------------------------------------
# Save session as


def test_save_session_as_leaves_a_complete_run_folder_and_the_next_save_goes_there(saved, qtbot, tmp_path):
    window, path, folder = saved
    controller = window.controller
    results = with_results(folder)
    before = (folder / "session.json").read_bytes()
    chosen = tmp_path / "other place" / "run 2"

    assert controller.save_as(chosen) == chosen / "session.json"
    assert files_of(chosen) == ["results.npz", "session.json"]  # a complete run folder at the new place
    assert (chosen / "results.npz").read_bytes() == results
    assert controller.run_folder == chosen and controller.saved_path == chosen / "session.json"
    assert controller.save_problem is None
    # the folder is self-contained: its session names the video by the path from the new folder
    assert read_json(chosen / "session.json")["video"]["relpath"] == os.path.relpath(path, chosen).replace(os.sep, "/")
    # the folder that was left stays as it is
    assert files_of(folder) == ["results.npz", "session.json"]
    assert (folder / "session.json").read_bytes() == before and (folder / "results.npz").read_bytes() == results

    body(window, 1).step_box.setValue(5)  # a change: the delayed save, and the Save key, go to the new folder
    assert controller.save_timer.isActive()
    controller.save_timer.timeout.emit()
    assert read_json(chosen / "session.json")["clip"]["step"] == 5
    assert controller.save_now() == chosen / "session.json"
    assert (folder / "session.json").read_bytes() == before
    type_into(qtbot, body(window, 1).name_edit, "Grace")  # the new folder holds results: a new name stays in it
    assert controller.run_folder == chosen


def test_save_session_as_without_results_writes_the_session_alone(saved, tmp_path):
    window, _, folder = saved
    chosen = tmp_path / "elsewhere"
    assert window.controller.save_as(chosen) == chosen / "session.json"
    assert files_of(chosen) == ["session.json"] and files_of(folder) == ["session.json"]
    assert read_json(chosen / "session.json")["student"] == "Ada Lovelace"


def test_save_session_as_saves_what_is_typed_and_not_entered(saved, qtbot, tmp_path):
    window, _, _ = saved
    type_into(qtbot, body(window, 2).fps_edit, "239.6", enter=False)
    chosen = tmp_path / "elsewhere"
    window.controller.save_as(chosen)
    assert read_json(chosen / "session.json")["time"]["fps_true"] == 239.6


def test_if_the_copy_fails_nothing_changes_and_the_reason_is_told(saved, monkeypatch, tmp_path):
    window, _, folder = saved
    controller = window.controller
    asked = record_every_dialog(monkeypatch)
    with_results(folder)
    kept = {name: (folder / name).read_bytes() for name in files_of(folder)}
    chosen = tmp_path / "full disk"

    def no_room(source, target):
        raise OSError(28, "No space left on device")

    copyfile = shutil.copyfile
    monkeypatch.setattr(session_controller.shutil, "copyfile", no_room)
    assert controller.save_as(chosen) is None
    assert controller.run_folder == folder and controller.saved_path == folder / "session.json"
    assert not chosen.exists()  # not even the folder is left at the new place
    assert {name: (folder / name).read_bytes() for name in files_of(folder)} == kept
    reason = NOT_COPIED.format(folder="full disk", kept=FOLDER_OF_NAME)
    assert asked.messages == [(window, "problem", f"{NOT_WRITTEN}\n{reason}")]
    assert controller.save_problem is None  # the session is still saved where it was
    assert body(window, 1).folder_label.full_text == FOLDER_OF_NAME

    monkeypatch.setattr(session_controller.shutil, "copyfile", copyfile)
    assert controller.save_now() == folder / "session.json"  # and goes on being saved there
    assert controller.save_as(chosen) == chosen / "session.json"  # the same folder again, once the disk has room
    assert files_of(chosen) == ["results.npz", "session.json"]


@pytest.mark.parametrize("held", ["session.json", "results.npz"])
def test_a_folder_that_holds_another_run_is_not_written_into(saved, monkeypatch, tmp_path, held):
    window, _, folder = saved
    controller = window.controller
    asked = record_every_dialog(monkeypatch)
    with_results(folder)
    chosen = tmp_path / "taken"
    chosen.mkdir()
    (chosen / held).write_bytes(b"earlier work")
    assert controller.save_as(chosen) is None
    assert files_of(chosen) == [held] and (chosen / held).read_bytes() == b"earlier work"
    assert controller.run_folder == folder
    assert asked.messages == [(window, "problem", f"{NOT_WRITTEN}\n{HELD.format(folder='taken')}")]


def test_a_folder_of_tracker_files_is_not_made_a_run_folder(saved, monkeypatch, tmp_path):
    # a student's folder of Tracker exports: their notebooks read every .csv and .txt in it as a track
    window, _, folder = saved
    asked = record_every_dialog(monkeypatch)
    chosen = tmp_path / "tracker data"
    chosen.mkdir()
    (chosen / "A.txt").write_text("t\tx\ty\n", encoding="utf-8")
    assert window.controller.save_as(chosen) is None
    assert files_of(chosen) == ["A.txt"] and window.controller.run_folder == folder
    assert asked.messages == [(window, "problem", f"{NOT_WRITTEN}\n{TRACKER_FILES.format(folder='tracker data')}")]


def test_save_session_as_into_the_run_folder_itself_is_a_plain_save(saved, monkeypatch):
    window, _, folder = saved
    asked = record_every_dialog(monkeypatch)
    results = with_results(folder)
    body(window, 1).step_box.setValue(7)
    assert window.controller.save_as(folder) == folder / "session.json"
    assert read_json(folder / "session.json")["clip"]["step"] == 7
    assert (folder / "results.npz").read_bytes() == results and asked.messages == []


def test_save_session_as_needs_a_video_and_a_name(window, qtbot, clip_in_odd_folder, tmp_path):
    chosen = tmp_path / "elsewhere"
    assert window.controller.save_as(chosen) is None  # no video
    window.open_path(clip_in_odd_folder.path)
    assert window.controller.save_as(chosen) is None  # no name
    assert not chosen.exists() and window.controller.run_folder is None
