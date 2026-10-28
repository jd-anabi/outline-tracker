"""A session.json of version 1 in the window (W3 step 1; rule 5 of docs/ROADMAP.md section 2, decisions 36
to 39; docs/design/formats.md, "Writing over a version 1 file").

Opening and closing it writes nothing, also when its video has moved; the first real change writes version 2
and keeps the old file, byte for byte, as session.v1.json, and the window says so once; a file of version 1
that holds a key of version 2 is refused with the key's name; a copy that fails leaves session.json as it was.
Each old file is example 1 of formats.md, fitted to the dish clip by literal values (tests/support/
session_v1_helpers.py). Every clip is a copy in the test's own folder.
"""

import errno
import os

import pytest

import helpers
from gui_helpers import record_dialogs
from outline_tracker.core import session as session_module
from session_helpers import body, read_json, settle
from session_v1_helpers import OLD_NS, snapshot, window_session, write_v1

pytestmark = pytest.mark.usefixtures("own_settings")  # every test here has a settings folder of its own

RUN = "dish_tracker_outline_ana"  # the folder of a session of "ana" (SPEC 8.1); formats.md's example 1 is hers
NOT_OPENED = "The session could not be opened"
NOT_WRITTEN = "A file could not be written"
MOVED = "../old place/dish_tracker.mp4"  # where the video was when the file was written


def old_session(clip, relpath=None):
    """`<clip folder>/<RUN>/session.json` of version 1 for the clip; returns its path."""
    run_folder = clip.path.parent / RUN
    return write_v1(run_folder / "session.json", window_session(clip.path, run_folder, relpath))


@pytest.mark.parametrize("relpath", [None, MOVED], ids=["in place", "video moved"])
def test_opening_and_closing_a_version_1_session_writes_nothing(window, qtbot, monkeypatch, clip_in_odd_folder,
                                                                 relpath):
    asked = record_dialogs(monkeypatch)
    target = old_session(clip_in_odd_folder, relpath)
    before = snapshot(clip_in_odd_folder.path.parent)

    window.open_path(target)
    controller = window.controller
    assert controller.session.student == "ana" and controller.session.read_version == 1
    assert controller.video_path == clip_in_odd_folder.path and controller.run_folder == target.parent
    assert controller.session.video.relpath == "../dish_tracker.mp4"  # found again, and not written (decision 37)
    assert body(window, 2).fps_edit.text() == "239.6" and body(window, 2).source_label.text() == "typed in"
    assert not controller.save_timer.isActive()
    settle(qtbot, window)
    assert window.close()

    assert snapshot(clip_in_odd_folder.path.parent) == before  # the same files, bytes and times
    assert asked.messages == []


def test_the_first_change_writes_version_2_and_keeps_the_old_file_and_says_so_once(window, qtbot, monkeypatch,
                                                                                    clip_in_odd_folder):
    asked = record_dialogs(monkeypatch)
    target = old_session(clip_in_odd_folder, MOVED)
    old = target.read_bytes()
    window.open_path(target)
    controller = window.controller
    settle(qtbot, window)

    controller.session.notes = "third try"  # a real change
    controller.touch()
    kept = target.with_name("session.v1.json")
    qtbot.waitUntil(kept.is_file, timeout=10_000)  # the delayed save
    assert kept.read_bytes() == old and kept.stat().st_mtime_ns == OLD_NS
    saved = read_json(target)
    assert saved["schema_version"] == 2 and saved["label"] == "ana" and saved["notes"] == "third try"
    assert saved["video"]["relpath"] == "../dish_tracker.mp4"  # the video's new place, with the change
    assert sorted(entry.name for entry in target.parent.iterdir()) == ["session.json", "session.v1.json"]
    (parent, kind, text), = asked.messages
    assert parent is window and kind == "info"
    assert "session.v1.json" in text and "0.1.0" in text and "renamed to session.json" in text
    assert helpers.ODD_FOLDER not in text and str(target.parent) not in text  # names, never folders

    controller.session.notes = "fourth try"
    controller.touch()
    assert controller.save_now() == target and read_json(target)["notes"] == "fourth try"
    assert len(asked.messages) == 1 and kept.read_bytes() == old  # told once; the kept file stays as it was
    assert window.close()
    assert sorted(entry.name for entry in target.parent.iterdir()) == ["session.json", "session.v1.json"]


def test_a_version_1_file_that_holds_a_key_of_version_2_is_refused_with_its_name(window, monkeypatch,
                                                                                 clip_in_odd_folder):
    asked = record_dialogs(monkeypatch)
    run_folder = clip_in_odd_folder.path.parent / RUN
    data = window_session(clip_in_odd_folder.path, run_folder)
    data["label"] = "ana"  # "student" in version 1, "label" in version 2: which counts cannot be told
    target = write_v1(run_folder / "session.json", data)
    before = snapshot(clip_in_odd_folder.path.parent)

    window.open_path(target)
    (parent, kind, text), = asked.messages
    assert parent is window and kind == "problem" and text.splitlines()[0] == NOT_OPENED
    assert '"label"' in text and "version 1" in text and "version 2" in text
    assert helpers.ODD_FOLDER not in text
    assert window.controller.session is None and window.controller.run_folder is None  # nothing was opened
    assert window.close()
    assert snapshot(clip_in_odd_folder.path.parent) == before


def test_when_the_old_file_cannot_be_kept_session_json_stays_and_the_window_says_why(window, qtbot, monkeypatch,
                                                                                     clip_in_odd_folder):
    asked = record_dialogs(monkeypatch)
    target = old_session(clip_in_odd_folder)
    window.open_path(target)
    settle(qtbot, window)
    before = snapshot(clip_in_odd_folder.path.parent)
    kept = str(target.with_name("session.v1.json"))

    def disk_full(path, tag, data, mtime_ns=None):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), kept)

    monkeypatch.setattr(session_module, "keep_beside", disk_full)
    controller = window.controller
    controller.session.notes = "third try"
    controller.touch()
    assert controller.save_now() is None
    problem = controller.save_problem
    assert "session.v1.json" in problem and "session.json was not replaced" in problem
    assert os.strerror(errno.ENOSPC) in problem and helpers.ODD_FOLDER not in problem
    (parent, kind, text), = asked.messages
    assert kind == "problem" and text == f"{NOT_WRITTEN}\n{problem}"
    assert snapshot(clip_in_odd_folder.path.parent) == before
