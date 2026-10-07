"""What the tests of task C9 share (tests/gui/test_joins.py): a run folder that was tracked before
the window opened, the parts of the window that a task which writes files switches off, and the
times of the three files that decide the state of panel 9.

Imported by name from tests/gui/test_joins.py. Nothing here imports torch, and no helper waits
with a delay: a factory, a run or an export is parked on a `Gate` (tests/gui/prompt_helpers.py).

Frames are video frame numbers; times of files are the computer's local time.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from export_helpers import coarse_run
from gui_helpers import show
from outline_tracker import schema
from session_helpers import body, type_into
from track_helpers import NAME

RUNNING = "Tracking is running."        # why a part is off during a tracking run
EXPORTING = "An export is running."     # and during an export (the brief's sentence)
WAIT_FOR_EXPORT = "Wait until the export has ended."  # a point tool's line during an export
DAY = (2026, 2, 3)  # a day that is over, whenever the tests run


def earlier_run(window, qtbot, clip, factory) -> Path:
    """A run folder that the core tracked before the window opened (A of `clip`, a copy in the
    test's own folder, on every 2nd frame, with fps_true and a scale), opened in `window` with
    `factory` as the window's model factory and the student's name typed. The window is shown, so
    its worker has begun to load the model. Returns the run folder."""
    folder = clip.path.parent / "a run"
    coarse_run(clip, folder, ["A"])
    window.segmenter_factory = factory
    window.open_path(folder / schema.SESSION_JSON)
    type_into(qtbot, body(window, 1).name_edit, NAME)  # the folder was made without a name
    show(window, qtbot)
    return folder


def lockable(window) -> dict:
    """The parts that change what a tracking run or an export works on, by a name for the failure
    message. Panel 6: Add, Remove, the three tools, the mode and the window (its box and Auto).
    Panel 1 and the File menu: the name, the clip, Open video, Open session, Save session as.
    Panel 7: the device (the model is in `model_box`: it is also off once there are results)."""
    one, six, seven, menus = body(window, 1), body(window, 6), body(window, 7), window.menus
    return {"Add": six.add_button, "Remove": six.remove_button, "Positive": six.positive_button,
            "Negative": six.negative_button, "Head": six.head_button, "mode": six.mode_box,
            "window": six.window_box, "Auto": six.auto_button, "name": one.name_edit, "start": one.start_box,
            "end": one.end_box, "step": one.step_box, "Open video": one.open_video_button,
            "Open session": one.open_session_button, "File > Open video": menus.open_action,
            "File > Open session": menus.open_session_action, "File > Save session as": menus.save_as_action,
            "device": seven.device_box}


def corrections(window) -> dict:
    """The three corrections of panel 8, by name."""
    eight = body(window, 8)
    return {"Re-track from here": eight.retrack_button, "End track here": eight.end_button,
            "Continue as new track": eight.continue_button}


def off(parts: dict) -> list[str]:
    """The names of the parts that are off."""
    return [name for name, part in parts.items() if not part.isEnabled()]


def not_saying(parts: dict, reason: str) -> list[str]:
    """The names of the parts whose tooltip is not `reason`."""
    return [name for name, part in parts.items() if part.toolTip() != reason]


def written_at(folder: Path, **times: tuple[int, int]) -> None:
    """Say when files of the run folder `folder` were written: `results`, `session` and `positions`
    (results.npz, session.json, positions.csv), each as (hour, minute) of `DAY`."""
    names = {"results": schema.RESULTS_NPZ, "session": schema.SESSION_JSON, "positions": schema.POSITIONS_CSV}
    for which, (hour, minute) in times.items():
        when = datetime(*DAY, hour, minute).timestamp()
        os.utime(folder / names[which], (when, when))
