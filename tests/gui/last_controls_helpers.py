"""What the tests of task C8b share: a run parked inside the model, the question before Remove,
a stand-in that names its device, and the README's Quickstart read as data.

Imported by name from tests/gui/test_run_lock.py, test_model_choice.py, test_fill.py and
test_finish.py. Nothing here imports torch.

- `parked` gives a stand-in whose tracked frame `park_at` waits at a `Gate` (prompt_helpers), and
  `press_track` presses Track and waits until the worker thread stands there: the run is going,
  and stays so until the test opens the gate. No test waits with a delay.
- `record_confirm` replaces `dialogs.confirm` by a recorder: the test gives the answer itself.

Frames are video frame numbers; "tracked frame n" counts the frames a job gave the model, from 1.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QAbstractButton

from outline_tracker.segmenter.fake import ExactFake, ThresholdFake
from track_helpers import Tracked

REPO = Path(__file__).resolve().parents[2]
RUNNING = "Tracking is running."  # why a part is off during a run


class Questions:
    """What the window asked `dialogs.confirm`, in place of the dialog: `asked` holds (parent,
    text, action, on_confirmed). Nothing is answered: a test calls `on_confirmed` for a yes."""

    def __init__(self):
        self.asked = []

    def confirm(self, parent, text, action, on_confirmed):
        self.asked.append((parent, text, action, on_confirmed))


def record_confirm(monkeypatch) -> Questions:
    """Replace `confirm` of outline_tracker/gui/dialogs.py by a recorder, and return it."""
    from outline_tracker.gui import dialogs

    recorder = Questions()
    monkeypatch.setattr(dialogs, "confirm", recorder.confirm)
    return recorder


def parked(clip, gate, park_at: int = 3) -> Tracked:
    """A stand-in for `clip` (an `ExactFake`) whose tracked frame `park_at` waits at `gate`."""
    return Tracked(ExactFake(clip), gate=gate, park_at={park_at})


def press_track(qtbot, track, gate) -> None:
    """Press Track in panel 7 (`track`, its `TrackPanel`) and wait until the job stands at `gate`."""
    track.track_button.click()
    assert track.jobs.running, track.message.text()
    qtbot.waitUntil(gate.parked.is_set)


def let_run_end(qtbot, track, gate) -> None:
    """Open `gate` and wait until the job has ended and reported it."""
    gate.open()
    qtbot.waitUntil(lambda: not track.jobs.running, timeout=30_000)


class OnDevice(ThresholdFake):
    """A stand-in that says which device it runs on, as the real segmenter does, and how often it
    was closed."""

    def __init__(self, device: str):
        super().__init__()
        self.device, self.closed = device, 0

    def close(self):
        self.closed += 1
        super().close()


def choose(box, key: str) -> None:
    """Choose the entry of a combo box whose data is `key`, as a user does with the mouse."""
    index = box.findData(key)
    assert index >= 0, f"{key!r} is not among {[box.itemData(i) for i in range(box.count())]}"
    box.setCurrentIndex(index)
    box.activated.emit(index)


# ---------------------------------------------------------------------------------------------
# README.md and docs/DEVELOPER.md as data


def quickstart(text: str) -> str:
    """The `## Quickstart` section of a README text, up to the next `## ` heading, without its
    fenced code blocks; "" when there is none."""
    kept, inside = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            inside = not inside
        elif not inside:
            kept.append(line)
    parts = [part.partition("\n") for part in re.split(r"^## ", "\n".join(kept), flags=re.MULTILINE)]
    return next((body for head, _, body in parts if head.strip() == "Quickstart"), "")


def marked_names(section: str) -> list[str]:
    """What a text names in backticks or in bold, in order: the README marks a control so."""
    return [first or second for first, second in re.findall(r"\*\*([^*]+)\*\*|`([^`]+)`", section)]


def control_texts(window) -> set[str]:
    """The text of every button (check boxes and panel titles too) and of every menu item of the
    window, as the window's own widgets have them."""
    buttons = {button.text() for button in window.findChildren(QAbstractButton)}
    items = {item.text() for item in window.findChildren(QAction)}
    return {text for text in buttons | items if text}
