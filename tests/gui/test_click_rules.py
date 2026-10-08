"""`ResultsOnDisk` of outline_tracker/gui/click_rules.py: the results of the run folder as results.npz
holds them now, for the parts of the window that only read them.

During a run the worker thread replaces results.npz at every autosave, and the window reads the file
again after each one. On Windows the file can be refused to a reader at that moment (CI met it in
tests/gui/test_overlays.py). The rule: a results file that cannot be opened at this moment leaves what
was read before in place, and is read at the next call.

Expected values: the frames the test wrote, and the counts that follow from the rule: the file is
opened once per call for as long as the file on disk is not the one that was read, and not at all
once it is. No real lock is needed: the fixture `refuse_open` (tests/gui/conftest.py) refuses the open
as Windows does, and one test refuses the look at the file (`Path.stat`) the same way.

The records are made up (`made_up` of tests/results_helpers.py): only their frames are asked.
Frames are video frame numbers.
"""

import errno
from pathlib import Path
from types import SimpleNamespace

from outline_tracker.gui.click_rules import ResultsOnDisk
from outline_tracker.results import ResultsStore
from outline_tracker.schema import RESULTS_NPZ
from results_helpers import made_up


def reader(run_folder) -> ResultsOnDisk:
    """A `ResultsOnDisk` for `run_folder`. All it asks of a controller is where the run folder is."""
    return ResultsOnDisk(SimpleNamespace(run_folder=run_folder))


def save_frames(run_folder, frames) -> Path:
    """Put a results.npz with track A on the video frames `frames` into `run_folder`, in place of
    the one that is there, as an autosave does. Returns the file."""
    store = ResultsStore()
    for frame in frames:
        store.put("A", made_up(frame, seed=frame))
    return store.save(Path(run_folder) / RESULTS_NPZ)


def frames_of(store: ResultsStore) -> list[int]:
    """The video frames on which track A has a record in `store`."""
    return store.arrays("A").frames.tolist()


def test_a_file_that_cannot_be_opened_at_this_moment_leaves_what_was_read_and_is_read_at_the_next_call(
        tmp_path, refuse_open):
    results = reader(tmp_path)
    save_frames(tmp_path, [0, 2])
    before = results.now()
    assert frames_of(before) == [0, 2]
    save_frames(tmp_path, [0, 2, 4, 6])  # the next autosave: another file under the same name
    opens = refuse_open()                # and the worker is replacing it again: the open is refused once
    assert results.now() is before       # no error, and what was read before
    assert frames_of(before) == [0, 2]
    assert opens.tried == 1
    assert frames_of(results.now()) == [0, 2, 4, 6]
    assert opens.tried == 2              # the new file was opened twice: refused once, read once
    assert frames_of(results.now()) == [0, 2, 4, 6]
    assert opens.tried == 2              # the file is the one that was read: it is not opened again


def test_a_refusal_at_the_very_first_read_gives_no_records_and_the_next_call_reads_the_file(tmp_path, refuse_open):
    results = reader(tmp_path)
    save_frames(tmp_path, [0, 2])
    opens = refuse_open()
    assert results.now().track_ids == []  # nothing was read before: no records, and no error
    assert opens.tried == 1
    assert frames_of(results.now()) == [0, 2]
    assert opens.tried == 2


def test_a_file_that_cannot_be_looked_at_is_treated_as_one_that_cannot_be_opened(tmp_path, monkeypatch):
    results = reader(tmp_path)
    save_frames(tmp_path, [0, 2])
    before = results.now()
    assert frames_of(before) == [0, 2]
    path = save_frames(tmp_path, [0, 2, 4, 6])
    stat, refused = Path.stat, []

    def refuse_once(self, **how):
        """In place of `Path.stat`: the first look at results.npz is refused, every other is the real one."""
        if self == path and not refused:
            refused.append(self)
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return stat(self, **how)

    monkeypatch.setattr(Path, "stat", refuse_once)
    assert results.now() is before  # no error, and what was read before
    assert refused == [path]
    assert frames_of(results.now()) == [0, 2, 4, 6]


def test_what_was_read_in_another_run_folder_is_not_what_was_read_before(tmp_path, refuse_open):
    first, second = tmp_path / "first", tmp_path / "second"
    save_frames(first, [0, 2])
    save_frames(second, [0, 2, 4, 6])
    controller = SimpleNamespace(run_folder=first)
    results = ResultsOnDisk(controller)
    assert frames_of(results.now()) == [0, 2]
    controller.run_folder = second  # another video is opened, with results of its own
    opens = refuse_open()
    assert results.now().track_ids == []  # of this file nothing was read: not the other folder's records
    assert frames_of(results.now()) == [0, 2, 4, 6]
    assert opens.tried == 2
