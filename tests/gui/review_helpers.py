"""What the tests of panel 8 share (task C7): a tracked run in a window, the answer to the question
before a correction, and the frames on which the dish scene's B and C touch.

Imported by name from tests/gui/test_review.py and test_review_fix.py. Nothing here imports torch,
and no helper waits with a delay: the flags are listed in the worker thread, and `listed` waits
for the panel to say that its table is up to date.

Two ways to a tracked run:
- `opened_run` opens a copy of a run folder that the core made once (`dish_run`: A, B and C of the
  dish clip on every 2nd frame, with a scale): for the table and for going through it;
- `tracked_window` tracks in the window itself, with a student's name: for the corrections, which
  start a run.

Coordinates: (u, v) in px of the video frame (SPEC 3.1: u to the right, v downward, pixel centers at
+0.5). Frames are video frame numbers; times are s.
"""

from __future__ import annotations

import shutil

import pytest
from PySide6.QtCore import Qt

from export_helpers import ellipse_gap
from gui_helpers import show
from outline_tracker import schema
from outline_tracker.segmenter.fake import ExactFake
from results_helpers import read_npz
from track_helpers import Tracked, ready_to_track, run_to_end
from tracking_helpers import abc_session, run

FPS = 240.0                     # of every synthetic scene
GRID = list(range(0, 120, 2))   # the dish clip has 120 frames; a new clip takes every 2nd
# 100 px for 3.24 mm: the dish scene's 0.0324 mm per px; the origin in the middle of a 320 x 240 px frame
STICK = {"p1_px": [10.5, 20.5], "p2_px": [110.5, 20.5], "length_mm": 3.24}
ORIGIN = [160.5, 120.5]
# CONTACT is "within 3 px" here (2 grid cells are 2.5 px on a 320 px frame, decision X6). The stored
# outline has 256 points, so frames whose true gap is this close to 3 px may go either way.
SURELY_WITHIN, SURELY_BEYOND = 2.7, 3.3


def review_panel(window):
    """The controls of panel 8 (Review and fix) in the window: its `ReviewPanel`."""
    from outline_tracker.gui.panels.review_panel import ReviewPanel

    return window.panels[7].findChild(ReviewPanel)


def hint(window) -> str:
    """The hint line of panel 8."""
    return window.panels[7].hint.text()


def listed(qtbot, review) -> None:
    """Wait until the panel's table is that of session.json and results.npz as they are on disk."""
    qtbot.waitUntil(lambda: not review.listing.pending, timeout=30_000)


def give_scale(window, save: bool = True) -> None:
    """Give the open session `STICK` and `ORIGIN`: the flags need the scale. The session is saved
    at once unless `save` is false; then the controller saves it a moment after the change."""
    session = window.controller.session
    session.calibration.stick = dict(STICK)
    session.axes.origin_px = list(ORIGIN)
    window.controller.touch()
    if save:
        window.controller.save_now()


@pytest.fixture(scope="session")
def dish_run(dish_clip, tmp_path_factory):
    """A run folder in which the core tracked A, B and C of the dish clip on every 2nd frame with
    `ExactFake`, with a scale. Tests open a copy of it (`opened_run`), never the folder itself."""
    folder = tmp_path_factory.mktemp("review") / "run"
    status, _ = run(dish_clip, abc_session(dish_clip, folder), folder, ExactFake(dish_clip))
    assert status == "complete"
    return folder


def opened_run(window, qtbot, run_folder, own_folder, model: bool = True):
    """Open a copy of `run_folder`, made as `own_folder`, in the window; show the window with
    panel 8 open; wait until the model is ready and the flags are listed. With `model` false the
    window has no model, as when it could not be loaded. Returns the `ReviewPanel`."""
    if not model:
        window.segmenter_factory = None
    shutil.copytree(run_folder, own_folder)
    window.open_path(own_folder / schema.SESSION_JSON)
    window.panels[7].set_expanded(True)
    show(window, qtbot)
    review = review_panel(window)
    qtbot.waitUntil(lambda: review.worker.state == ("ready" if model else "failed"))
    listed(qtbot, review)
    return review


def tracked_window(window, qtbot, clip, segmenter=None, ids="ABC", end=80, scale: bool = True):
    """Track the objects `ids` of `clip` (a copy in the test's own folder) to frame `end` in the
    window, with `segmenter` (`ExactFake` in a `Tracked` if None), and wait for the flags. The
    session has `STICK` unless `scale` is false. Returns (the `ReviewPanel`, the `ObjectsPanel`,
    the `TrackPanel`)."""
    stand_in = Tracked(ExactFake(clip)) if segmenter is None else segmenter
    track, objects = ready_to_track(window, qtbot, clip, stand_in, ids=tuple(ids), end=end)
    if scale:
        give_scale(window)
    window.panels[7].set_expanded(True)
    run_to_end(qtbot, track)
    assert track.jobs.status == "complete", track.jobs.reason
    review = review_panel(window)
    listed(qtbot, review)
    return review, objects, track


def track_again(qtbot, review, timeout: int = 30_000) -> None:
    """Wait for the run that a correction started, and for the flags listed after it."""
    assert review.jobs.running, review.message.text() or hint(review.window())
    qtbot.waitUntil(lambda: not review.jobs.running, timeout=timeout)
    assert review.jobs.status == "complete", review.jobs.reason
    listed(qtbot, review)


class Confirming:
    """In place of `dialogs.confirm`: `asked` holds (text, action) of every question, and each is
    answered at once, with yes (`on_confirmed` is called) while `answer` is true."""

    def __init__(self, answer: bool = True):
        self.answer, self.asked = answer, []

    def confirm(self, parent, text, action, on_confirmed):
        self.asked.append((text, action))
        if self.answer:
            on_confirmed()


def confirming(monkeypatch, answer: bool = True) -> Confirming:
    """Replace `dialogs.confirm` by a `Confirming` that answers every question with `answer`."""
    from outline_tracker.gui import dialogs

    recorder = Confirming(answer)
    monkeypatch.setattr(dialogs, "confirm", recorder.confirm)
    return recorder


def frames_with(rows, track_id: str, code: str) -> list[int]:
    """The frames of the rows (track, frame, t in s, code) that are of `track_id` with `code`."""
    return [frame for name, frame, _, row_code in rows if (name, row_code) == (track_id, code)]


def contact_frames(clip, frames) -> tuple[list[int], list[int]]:
    """Of `frames`, (those on which the true outlines of B and C are surely within the 3 px of
    CONTACT, those on which they are surely beyond), from the two true ellipses."""
    gaps = {frame: ellipse_gap(clip, "B", "C", frame) for frame in frames}
    return ([frame for frame in frames if gaps[frame] <= SURELY_WITHIN],
            [frame for frame in frames if gaps[frame] >= SURELY_BEYOND])


def shown_rows(review) -> list[tuple[str, ...]]:
    """The texts of the table's cells, row by row."""
    model = review.table.model()
    return [tuple(model.data(model.index(row, column), Qt.ItemDataRole.DisplayRole)
                  for column in range(model.columnCount())) for row in range(model.rowCount())]


def current_row(review) -> int:
    """The table's current row, -1 for none."""
    return review.table.currentIndex().row()


def object_cells(objects, track_id: str) -> tuple[str, ...]:
    """The texts of the row of `track_id` in panel 6's table."""
    table = objects.table
    (row,) = [row for row in range(table.rowCount()) if table.item(row, 0).text() == track_id]
    return tuple(table.item(row, column).text() for column in range(table.columnCount()))


def results_file(window) -> dict:
    """Every array of results.npz of the window's run folder, by key, read now."""
    return read_npz(window.controller.run_folder / schema.RESULTS_NPZ)


def assert_tracks_identical(after: dict, before: dict, track_ids) -> None:
    """Every array of the named tracks is the same in two readings of results.npz (`results_file`)."""
    import numpy as np

    for track_id in track_ids:
        for key in schema.RESULTS_KEYS:
            name = schema.npz_key(track_id, key.name)
            assert after[name].dtype == before[name].dtype and after[name].shape == before[name].shape, name
            np.testing.assert_array_equal(after[name], before[name], err_msg=name)
