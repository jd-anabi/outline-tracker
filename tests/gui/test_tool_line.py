"""The line above the picture that says what the chosen tool does (gui/main_window.py).

A tool's sentence can be longer than the room beside the Fit and 1:1 buttons. The line then shows
what fits and keeps the whole sentence as its tooltip; it never makes room for itself by pushing
the dock. Widths are Qt's device-independent px; the check compares the dock with itself, so it
does not depend on a font.
"""

from gui_helpers import show
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

LONG = "Circle: " + "click a point on the inner wall of the dish, far from the points before it. " * 8


class LongTool:
    """A tool whose sentence is far wider than any window."""

    cursor = Qt.CursorShape.CrossCursor
    text = LONG

    def click(self, u, v, button, modifiers):
        pass


def test_a_long_tool_line_does_not_squeeze_the_dock(window, qtbot, dish_clip):
    window.resize(960, 600)  # the smallest window
    window.open_path(dish_clip.path)
    show(window, qtbot)
    dock, picture = window.dock.width(), window.view.width()
    window.view.set_tool(LongTool())
    QApplication.processEvents()  # let the layout settle
    assert (window.dock.width(), window.view.width()) == (dock, picture)
    assert window.tool_text.text() == LONG and window.tool_text.toolTip() == LONG  # the whole sentence on hover
    window.view.set_tool(None)
    QApplication.processEvents()
    assert (window.dock.width(), window.view.width()) == (dock, picture)
    assert window.tool_text.toolTip() == window.tool_text.text()  # Pan's own line
