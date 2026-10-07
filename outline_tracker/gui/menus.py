"""The menus of the window (SPEC 10.1): File with Open video, Open session, Save session, Save
session as and Quit; Help with Quickstart and About.

An item does what its button or key does elsewhere, through the same function: Open video is the
window's `choose_video`; Open session and Save session are connected by panel 1, which has their
functions and buttons; Save session as asks for a folder and hands it to the controller's
`save_as`. An item that cannot work yet is off: the two that save need a video and the student's
name. The Save key belongs to Save session; while that item is off the key still calls its
function, which then says in the status bar what is missing.

File > Export is added by the export panel. No quantities here, so no units and no coordinate
frame.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut

from outline_tracker.gui import about, dialogs

QUICKSTART_URL = "https://github.com/jd-anabi/outline-tracker#readme"  # the repository's README


class Menus(QObject):
    """The menu bar of `window` (a `MainWindow`). The items: `open_action`, `open_session_action`,
    `save_action`, `save_as_action` and `quit_action` in File; `quickstart_action` and
    `about_action` in Help. `save_key` is the Save key while Save session is off."""

    def __init__(self, window):
        super().__init__(window)
        self._window, self._controller = window, window.controller
        standard = QKeySequence.StandardKey
        self.open_action = QAction("Open video", window)
        self.open_action.setShortcut(QKeySequence(standard.Open))
        self.open_session_action = QAction("Open session", window)
        self.save_action = QAction("Save session", window)
        self.save_action.setShortcut(QKeySequence(standard.Save))
        self.save_as_action = QAction("Save session as", window)
        self.quit_action = QAction("Quit", window)
        self.quit_action.setShortcut(QKeySequence(standard.Quit))
        self.quit_action.setMenuRole(QAction.MenuRole.QuitRole)
        self.quickstart_action = QAction("Quickstart", window)
        self.about_action = QAction("About", window)
        self.about_action.setMenuRole(QAction.MenuRole.NoRole)  # in Help on every system, as the spec lists it

        file_menu = window.menuBar().addMenu("File")
        file_menu.addActions([self.open_action, self.open_session_action, self.save_action, self.save_as_action])
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        help_menu = window.menuBar().addMenu("Help")
        help_menu.addActions([self.quickstart_action, self.about_action])

        self.save_key = QShortcut(QKeySequence(standard.Save), window)
        self.save_key.activated.connect(self.save_action.triggered)  # the item's own function, whoever gave it
        self.open_action.triggered.connect(window.choose_video)
        self.save_as_action.triggered.connect(self.save_session_as)
        self.quit_action.triggered.connect(window.close)
        self.quickstart_action.triggered.connect(self.open_quickstart)
        self.about_action.triggered.connect(lambda: about.show(window))
        for changed in (self._controller.video_opened, self._controller.session_changed, self._controller.saved):
            changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """Switch each item on or off: Save session and Save session as need a video and the
        student's name; the others always work."""
        can_save = self._controller.session is not None and bool(self._controller.student.strip())
        self.save_action.setEnabled(can_save)
        self.save_as_action.setEnabled(can_save)
        self.save_key.setEnabled(not can_save)  # never both: two holders of one key would both stay silent

    def save_session_as(self) -> None:
        """Ask for the folder to go on in; the chosen one goes to the controller's `save_as`, and
        the status bar says where the session is then. A folder that cannot be used is told by the
        controller, and the session stays where it was."""
        dialogs.choose_folder(self._window, "Save session as", self._save_as)

    def _save_as(self, folder: Path) -> None:
        if self._controller.save_as(folder) is not None:
            self._window.statusBar().showMessage(f"The session is saved in the run folder {Path(folder).name}.")

    def open_quickstart(self) -> None:
        """Open the quickstart, the README of the repository, in the system's browser."""
        QDesktopServices.openUrl(QUrl(QUICKSTART_URL))
