"""`dialogs.confirm`: the question before an action that cannot be undone (remove an object that has
results, end a track, re-track from a frame). SPEC 10.2 and the design note: the safe button is the
default one, the other button is named after the action, and the dialog never blocks the window.
"""

import pytest
from gui_helpers import show
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QMessageBox, QPushButton

from outline_tracker.gui import dialogs

LEFT = Qt.MouseButton.LeftButton
TEXT = "Remove object B?\nIts 120 tracked frames are removed from the results. This cannot be undone."


@pytest.fixture
def never_blocking(monkeypatch):
    """Make `exec`, which would wait for the user and never return here, an error."""
    def blocked(*_):
        raise AssertionError("A dialog was run with exec(): it blocks the window. Use open().")

    for kind in (QDialog, QMessageBox):
        monkeypatch.setattr(kind, "exec", blocked)


def asked(window, qtbot, done):
    show(window, qtbot)
    dialogs.confirm(window, TEXT, "Remove", lambda: done.append("removed"))
    (box,) = (box for box in window.findChildren(QMessageBox) if box.isVisible())  # a closed one goes later
    return box, {button.text(): button for button in box.findChildren(QPushButton)}


def test_confirm_asks_with_the_action_and_cancel_and_cancel_is_the_safe_default(window, qtbot, never_blocking):
    done = []
    box, buttons = asked(window, qtbot, done)
    assert box.isVisible() and box.isModal()
    assert box.text() == "Remove object B?"
    assert box.informativeText() == "Its 120 tracked frames are removed from the results. This cannot be undone."
    assert box.icon() == QMessageBox.Icon.Warning
    assert sorted(buttons) == ["Cancel", "Remove"]
    assert box.defaultButton() is buttons["Cancel"] and box.escapeButton() is buttons["Cancel"]
    assert done == []  # asking has done nothing yet


def test_the_action_button_does_it_once(window, qtbot, never_blocking):
    done = []
    box, buttons = asked(window, qtbot, done)
    qtbot.mouseClick(buttons["Remove"], LEFT)
    assert done == ["removed"] and not box.isVisible()


def test_cancel_and_closing_the_dialog_do_nothing(window, qtbot, never_blocking):
    done = []
    box, buttons = asked(window, qtbot, done)
    qtbot.mouseClick(buttons["Cancel"], LEFT)
    assert done == [] and not box.isVisible()
    box, _ = asked(window, qtbot, done)
    box.reject()  # Esc, or the window's close button
    assert done == []
