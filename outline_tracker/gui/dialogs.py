"""The three dialogs of the window (SPEC 10.2): a message, the choice of a file to open, and the
choice of a folder.

Every module of the window asks through `message`, `open_file` and `choose_folder`, so that there is
one place for how a dialog looks and behaves, and one set of functions for a test to replace. Each
shows its dialog with `open()`: the call returns at once and the window's event loop goes on. None
uses `exec()`, which would start a second event loop inside the call.

A dialog is for an action the user asked for that could not start or was stopped, and for what the
spec names (file dialogs, About); a warning, a field's error or a success is shown in the panel it
belongs to, not here. No quantities in this module, so no units and no coordinate frame.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

WINDOW_TITLE = "Outline Tracker"
KINDS = ("info", "warning", "problem")


def message(parent: QWidget, kind: str, text: str) -> None:
    """Show `text` in a dialog over `parent` with one button, "Close", and return at once.

    `kind` is `info`, `warning` or `problem` and chooses the icon; any other value is a
    `ValueError`. The first line of `text` says what happened and is shown as the heading; the
    lines after it say what to do next. A text of one line is the heading alone. The text names a
    file by its name, never by its folder, and holds no trace of the program.
    """
    icons = dict(zip(KINDS, (QMessageBox.Icon.Information, QMessageBox.Icon.Warning, QMessageBox.Icon.Critical)))
    if kind not in icons:
        raise ValueError(f"A message is one of {', '.join(KINDS)}, not {kind!r}.")
    heading, _, rest = text.partition("\n")
    box = QMessageBox(parent)
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)  # one dialog, one use: no heap of hidden ones
    box.setWindowTitle(WINDOW_TITLE)
    box.setIcon(icons[kind])
    box.setText(heading)
    box.setInformativeText(rest.strip())
    close = box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(close)
    box.setEscapeButton(close)
    box.open()


def open_file(parent: QWidget, title: str, filters: list[str], on_chosen) -> None:
    """Ask for one existing file in a dialog over `parent`, and return at once.

    `title` is the dialog's title. `filters` are Qt's name filters, the first one preselected, for
    example `["Videos (*.mp4 *.mov)", "All files (*)"]`. When the user chooses a file,
    `on_chosen(path)` is called with its `pathlib.Path`; when the user cancels, nothing is called.
    """
    dialog = QFileDialog(parent, title)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
    dialog.setNameFilters(list(filters))
    dialog.fileSelected.connect(lambda name: on_chosen(Path(name)))
    dialog.open()


def choose_folder(parent: QWidget, title: str, on_chosen) -> None:
    """Ask for one folder in a dialog over `parent`, and return at once.

    `title` is the dialog's title. The dialog shows folders only, and a new folder can be made in
    it. When the user chooses a folder, `on_chosen(path)` is called with its `pathlib.Path`; when
    the user cancels, nothing is called.
    """
    dialog = QFileDialog(parent, title)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
    dialog.fileSelected.connect(lambda name: on_chosen(Path(name)))
    dialog.open()
