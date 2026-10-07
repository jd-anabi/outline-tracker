"""What the tests of panel 9 (Export) and the smoke test of the whole flow share (task C6).

Imported by name from tests/gui/test_export_panel.py and tests/gui/test_smoke.py, the fixture too (a
fixture imported into a test module is that module's own). Nothing here imports torch.

- `second_window` is a second window for one test, made and closed by the function of the `window`
  fixture of tests/helpers.py: a saved session reopens in it.
- `ExportWatch` stands where panel 9 calls `export.export_all`: it keeps how it was called, can say
  lines and wait at a `Gate` (tests/gui/prompt_helpers.py) first, and then calls the real one. No
  helper waits with a delay.
- `tracked` is a window whose run folder holds the results of a run with `ExactFake`.
- `is_off` and `is_on` read Export all and File > Export together; `set_times` says when the two
  files that decide the panel's state were written; `lock` lets a file be open in another program.
- `walk_through` is SPEC 13.5's flow, through the panels' public slots and one simulated click.

Coordinates: (u, v) in px of the video frame (SPEC 3.1: u to the right, v downward, pixel centers at
+0.5). mm are in the session's axes, y up. Frames are video frame numbers; times are s.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pytest

import helpers
from calibration_helpers import on_circle, place
from export_helpers import STICK, calibrate
from gui_helpers import show
from outline_tracker import export, fileio, schema
from outline_tracker.gui.worker import worker_of
from outline_tracker.segmenter.fake import ExactFake
from prompt_helpers import LEFT, NO_KEY, at, this_thread
from session_helpers import body, type_into
from track_helpers import NAME, Tracked, ready_to_track, run_log, run_to_end, track_panel
from tracking_helpers import center

END = 20                     # the last frame of the clips tracked here: frames 0, 2, ..., 20
FRAMES = list(range(0, END + 1, 2))  # 11 frames at the default step of 2
STICK_ENDS = (STICK["p1_px"], STICK["p2_px"])  # 300 px apart
STICK_MM = STICK["length_mm"]                  # 30 mm: 0.1 mm per px
# SPEC 8.1: what a run folder holds after Export all, without probes (the Tracker-format files are
# in a folder named after the model)
P0_FILES = {schema.SESSION_JSON, schema.POSITIONS_CSV, "edgetam", schema.SHAPES_CSV, schema.RADIAL_CSV,
            schema.OUTLINES_NPZ, schema.OVERLAY_MP4, schema.RESULTS_NPZ, schema.RUN_LOG, schema.README_TXT}
# what Export all itself writes of them, by name relative to the run folder, for one track A
WRITTEN = {schema.POSITIONS_CSV, "edgetam/A.csv", schema.SHAPES_CSV, schema.RADIAL_CSV, schema.OUTLINES_NPZ,
           schema.README_TXT, schema.OVERLAY_MP4, schema.RUN_LOG}
EXPORT_TIP = "Write the CSV files, the overlay video, the log and README.txt to the run folder"  # while it is on


@pytest.fixture
def second_window(qtbot):
    """A second `MainWindow` for the same test, not shown yet: made and closed by the very function
    of the `window` fixture (tests/helpers.py), so it has the stand-in's factory, and after the
    test its worker is stopped and its video released."""
    yield from helpers.window.__wrapped__(qtbot)


def export_panel(window):
    """The controls of panel 9 (Export) in the window: its `ExportPanel`."""
    from outline_tracker.gui.panels.export_panel import ExportPanel

    return window.panels[8].findChild(ExportPanel)


def hint(window) -> str:
    """The hint line of panel 9."""
    return window.panels[8].hint.text()


def state(window) -> str:
    """The state of panel 9: "todo", "done" or "attention"."""
    return window.panels[8].state


def is_off(window, reason: str, hint_line: str | None = None) -> bool:
    """Whether Export all and File > Export are off, with `reason` as the button's tooltip and as
    the hint line (or `hint_line` there)."""
    panel = export_panel(window)
    return (not panel.export_button.isEnabled() and not window.menus.export_action.isEnabled()
            and panel.export_button.toolTip() == reason and hint(window) == (hint_line or reason))


def is_on(window) -> bool:
    """Whether Export all and File > Export can be pressed, and the button's tooltip says what it does."""
    panel = export_panel(window)
    return (panel.export_button.isEnabled() and window.menus.export_action.isEnabled()
            and panel.export_button.toolTip() == EXPORT_TIP)


def set_times(run_folder: Path, results: datetime, positions: datetime) -> None:
    """Say when results.npz and positions.csv of a run folder were written (the computer's local time)."""
    for name, when in ((schema.RESULTS_NPZ, results), (schema.POSITIONS_CSV, positions)):
        os.utime(run_folder / name, (when.timestamp(), when.timestamp()))


def lock(monkeypatch, name: str) -> None:
    """Let the file `name` be open in another program, as Windows reports it: the rename onto it
    is refused. The 5 s of tries are made without the waits."""
    real = fileio.os.replace

    def replace(source, destination):
        if Path(destination).name == name:
            raise PermissionError("the file is open in another program")
        return real(source, destination)

    monkeypatch.setattr(fileio.os, "replace", replace)
    monkeypatch.setattr(fileio.time, "sleep", lambda seconds: None)


class ExportWatch:
    """In place of `export_all` where panel 9 calls it: keeps each call, then does the real export.

    `calls`: for every call (run folder, overlay, the thread it ran in, the text of the folder's
    session.json at that moment). Before it exports, a call gives each line of `says` to `log`,
    waits at `gate` if there is one, and raises `error` if there is one. `reports`: what the real
    `export_all` returned, call by call.
    """

    def __init__(self, gate=None, says=(), error: Exception | None = None):
        self.gate, self.says, self.error = gate, list(says), error
        self.calls: list[tuple] = []
        self.reports: list = []
        self._real = export.export_all

    def __call__(self, run_folder, overlay=False, log=print):
        saved = (Path(run_folder) / schema.SESSION_JSON).read_text(encoding="utf-8")
        self.calls.append((Path(run_folder), overlay, this_thread(), saved))
        for line in self.says:
            log(line)
        if self.gate is not None:
            self.gate.park()
        if self.error is not None:
            raise self.error
        self.reports.append(self._real(run_folder, overlay=overlay, log=log))
        return self.reports[-1]


def watch_export(monkeypatch, **how) -> ExportWatch:
    """Put an `ExportWatch` where panel 9 calls `export_all`, and return it."""
    from outline_tracker.gui.panels import export_panel as module

    watch = ExportWatch(**how)
    monkeypatch.setattr(module, "export_all", watch)
    return watch


class Clock:
    """A clock for panel 9 in place of the computer's: each call gives the next of `times` (s)."""

    def __init__(self, *times: float):
        self._times = list(times)

    def __call__(self) -> float:
        return self._times.pop(0) if len(self._times) > 1 else self._times[0]


def tracked(window, qtbot, clip, scale: bool = True, end: int = END, ids=("A",)):
    """A window whose run folder holds results: the student is `NAME`, `clip` (a copy in the test's
    own folder) is open, fps_true is the scene's, each object of `ids` has a click, and a run with
    `ExactFake` has tracked them on `FRAMES`. With `scale` the session has the stick and the axes of
    tests/export_helpers.py (0.1 mm per px). Returns the `ExportPanel`."""
    panel, _ = ready_to_track(window, qtbot, clip, Tracked(ExactFake(clip)), ids=ids, end=end)
    if scale:
        calibrate(window.controller.session)
        window.controller.touch()
    run_to_end(qtbot, panel)
    assert panel.jobs.status == "complete", panel.jobs.reason
    return export_panel(window)


def export_to_end(qtbot, panel, timeout: int = 30_000) -> None:
    """Press Export all and wait until the export has ended and reported it."""
    panel.export_button.click()
    assert panel.exporting, panel.message.text() or panel.export_button.toolTip()
    qtbot.waitUntil(lambda: not panel.exporting, timeout=timeout)


def listed(panel) -> dict[str, str]:
    """What panel 9 lists as written: {the file's name relative to the run folder: its size as text}."""
    lines = [line for line in panel.files_label.text().split("\n") if line]
    return dict(line.split(" · ") for line in lines)


def walk_through(window, qtbot, clip, end: int = END):
    """SPEC 13.5's flow in `window`, on `clip` (a copy in the test's own folder): the window opens,
    the name is typed, the clip loads and is cut to frames 0 to `end`, fps_true is typed, the stick
    (`STICK_ENDS`, `STICK_MM`), eight points on the scene's dish wall and Origin to Center are set
    through the panels' slots, object A is added and clicked with the mouse on its true center, a
    run with `ExactFake` tracks it, and Export all writes the files. Returns the run folder."""
    window.segmenter_factory = lambda model, device: ExactFake(clip)
    show(window, qtbot)
    first = body(window, 1)
    type_into(qtbot, first.name_edit, NAME)
    window.open_path(clip.path)
    qtbot.waitUntil(lambda: worker_of(window).ready)
    first.end_box.setValue(end)
    type_into(qtbot, body(window, 2).fps_edit, f"{clip.scene.fps:g}")

    calibration, dish = body(window, 3), body(window, 4)
    place(calibration.stick_tool, STICK_ENDS)
    calibration.set_stick_length(STICK_MM)
    cu, cv, radius = clip.scene.dish
    place(dish.circle_tool, on_circle((cu, cv), radius, range(0, 360, 45)))
    dish.origin_button.click()

    objects = body(window, 6)
    objects.add_object()  # chooses the Positive tool: the next click on the picture is A's first point
    view = window.view
    qtbot.mouseClick(view.viewport(), LEFT, NO_KEY, at(view, *center(clip, "A", 0)))
    qtbot.waitUntil(lambda: "A" in objects.prompts.outlines)

    tracking = track_panel(window)
    run_to_end(qtbot, tracking)
    assert tracking.jobs.status == "complete", tracking.jobs.reason
    export_to_end(qtbot, export_panel(window))
    return window.controller.run_folder


def no_error_recorded(window, asked, caplog) -> bool:
    """Whether nothing went wrong in `window` that anybody was told of or that was noted: no trace
    kept by the worker, no dialog (`asked`: the recorder of tests/gui/finish_helpers.py), no trace
    in run.log, and nothing logged by the package from WARNING up (pytest's `caplog`)."""
    noted = [record for record in caplog.records
             if record.name.startswith("outline_tracker") and record.levelno >= logging.WARNING]
    return (worker_of(window).traces == [] and asked.messages == [] and not noted
            and "Traceback" not in run_log(window.controller.run_folder))
