"""Panel 2, "Time" (SPEC 2 step 4, 3.3, 4.1, 10.1): fps_true, typed, read from the course's
manifest or measured with a filmed stopwatch, with its source.

fps_true is the true frame rate in frames per second; the time of a frame is t = frame / fps_true
in s (SPEC 3.3), which the bottom bar shows as soon as the session has the value. A typed value
that is not a positive, finite number is refused in the field and the session keeps what it had; a
value under 100 frames per second is taken with a warning, since for the course's slow-motion
videos it usually means a re-timed copy.

The manifest (`data/manifest.csv`, one row per video with its fps_true) is looked for when a video
is opened whose session has no fps_true yet: in the current folder and upward from the video's
folder (`geometry.find_manifest`), then at the manifest the user pointed to last time, which is
remembered between sessions in the application's settings (`settings`). The settings are read only
when a video is opened and written only when the user chooses a manifest that lists the video.

Stopwatch… opens the dialog of gui/stopwatch_dialog.py; on OK its frame rate goes into the session
with the two frames and the two readings it came from. The window cannot be used while the dialog
is open, so what was entered is kept for the next time it opens, also after Cancel: the two frames
can be looked for one after the other. No px or mm here.
"""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from outline_tracker.geometry import find_manifest, fps_from_manifest
from outline_tracker.gui import dialogs, theme
from outline_tracker.gui.panels.video_panel import SPACING, Message, field_rows
from outline_tracker.gui.stopwatch_dialog import StopwatchDialog

WARN_BELOW = 100.0  # frames per second (SPEC 3.3)
MANIFEST_KEY = "manifest_path"  # in the settings: the manifest the user pointed to last
MANIFEST_FILTERS = ["Manifest (*.csv)", "All files (*)"]
SOURCE_WORDS = {"manifest": "from the manifest", "stopwatch": "from the stopwatch", "typed": "typed in",
                "tracker-export": "from the Tracker export"}

HINT_DONE = "fps_true = {fps:.2f} fps ({source})."
HINT_UNDER = "fps_true is under 100 fps. This usually means a re-timed copy of the video. Check the value."
NO_NUMBER = "fps_true must be a number, for example 240."
NOT_POSITIVE = "fps_true must be more than 0 fps."
NOT_LISTED = "{manifest} has no fps_true for {video}. Check the row of this video in the manifest, or type fps_true."


def settings() -> QSettings:
    """The application's settings, kept between sessions: an INI file in the user's own settings
    folder, the same kind of file on every system."""
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "outline-tracker", "outline-tracker")


def usable_fps(video_path, manifest) -> float | None:
    """fps_true in frames per second from the row of the video at `video_path` in the manifest file
    `manifest` (both paths), or None when the file, the row or a positive, finite number is missing."""
    fps = fps_from_manifest(Path(video_path), Path(manifest))
    return float(fps) if fps is not None and math.isfinite(fps) and fps > 0 else None


def fps_text(fps: float) -> str:
    """fps_true (frames per second) as the field shows it: up to 6 decimals, no zeros at the end."""
    return f"{fps:.6f}".rstrip("0").rstrip(".")


class TimePanel(QWidget):
    """The controls of panel 2. Parts: `fps_edit` (fps_true in frames per second), `source_label`
    (where the value came from), `message` (why a typed value or a manifest was refused),
    `stopwatch_button` and `manifest_button`."""

    def __init__(self, window):
        super().__init__()
        self._window, self._controller, self._panel = window, window.controller, window.panels[1]
        self._start_hint = self._panel.hint.text()
        self._refused = ""  # why what is in the field, or the manifest chosen last, was not taken
        self._stopwatch_draft: dict | None = None  # what the Stopwatch dialog held when it was closed last

        self.fps_edit = QLineEdit()
        self.fps_edit.setPlaceholderText("frames per second")
        self.fps_edit.setToolTip("The true frame rate of the recording in frames per second (fps_true). "
                                 "The time of a frame is its number divided by fps_true.")
        self.source_label = QLabel()
        self.message = Message()
        self.stopwatch_button = QPushButton("Stopwatch…")
        self.stopwatch_button.setAutoDefault(False)
        self.stopwatch_button.setToolTip("Measure fps_true from two frames that show a stopwatch")
        self.manifest_button = QPushButton("Choose manifest")
        self.manifest_button.setAutoDefault(False)
        self.manifest_button.setToolTip("Read fps_true from a manifest.csv file that lists this video")

        rows = QVBoxLayout(self)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(SPACING)
        buttons = QHBoxLayout()
        buttons.setSpacing(SPACING)
        buttons.addWidget(self.stopwatch_button)
        buttons.addWidget(self.manifest_button)
        buttons.addStretch(1)
        rows.addLayout(field_rows((("fps_true", self.fps_edit), ("Source", self.source_label), (None, self.message))))
        rows.addLayout(buttons)

        controller = self._controller
        self.fps_edit.editingFinished.connect(self._typed)
        self.stopwatch_button.clicked.connect(self.open_stopwatch)
        self.manifest_button.clicked.connect(self.choose_manifest)
        controller.video_opened.connect(self._video_opened)
        controller.session_changed.connect(self._show)
        controller.about_to_save.connect(self._typed)
        self._show()

    def choose_manifest(self) -> None:
        """Ask for a manifest file; fps_true is then read from its row for the open video."""
        dialogs.open_file(self._window, "Choose manifest", MANIFEST_FILTERS, self.use_manifest)

    def use_manifest(self, manifest) -> None:
        """Take fps_true from the manifest file at `manifest` (a path the user pointed to) and
        remember the file for the videos opened later. A manifest that does not give this video a
        positive number of frames per second changes nothing and is said in the panel."""
        manifest, video_path = Path(manifest), self._controller.video_path
        fps = usable_fps(video_path, manifest)
        if fps is None:
            self._refused = NOT_LISTED.format(manifest=manifest.name, video=video_path.name)
            self._show()
            return
        settings().setValue(MANIFEST_KEY, str(manifest))
        self._take(fps, "manifest", manifest)

    def open_stopwatch(self) -> None:
        """Open the Stopwatch dialog over the window, and return at once. It opens with what it
        held last time for this video, else with the numbers of the session's stopwatch. On OK its
        frame rate becomes fps_true, from the stopwatch. Without a video there is nothing to time."""
        if self._controller.session is None:
            return
        start = self._stopwatch_draft or self._controller.session.time.stopwatch
        dialog = StopwatchDialog(self._window, self._window.view.frame, start)
        dialog.accepted.connect(lambda: self._take(dialog.fps, "stopwatch", stopwatch=dialog.values()))
        dialog.finished.connect(lambda: setattr(self, "_stopwatch_draft", dialog.values()))
        dialog.open()

    def _take(self, fps: float, source: str, manifest: Path | None = None, stopwatch: dict | None = None) -> None:
        """Put fps_true (frames per second) and where it came from into the session: the manifest's
        path, or the stopwatch's two frames and two readings in s; what belongs to another source goes."""
        time = self._controller.session.time
        time.fps_true, time.source, time.stopwatch = fps, source, stopwatch
        time.manifest_path = None if manifest is None else str(manifest)
        self._refused = ""
        self.fps_edit.setModified(False)
        self._controller.touch()

    def _typed(self) -> None:
        """What is in the field counts (Enter, leaving the field, a save): a positive, finite
        number goes into the session as typed in; anything else is refused here."""
        session = self._controller.session
        if session is None or not self.fps_edit.isModified():
            return
        fps = float_or_none(self.fps_edit.text())
        if fps is None and not self.fps_edit.text().strip() and session.time.fps_true is None:
            self._refused = ""  # nothing typed, and nothing known: as before
        elif fps is None or not math.isfinite(fps):
            self._refused = NO_NUMBER
        elif fps <= 0:
            self._refused = NOT_POSITIVE
        else:
            if fps != session.time.fps_true or session.time.source != "typed":
                self._take(fps, "typed")
            self._refused = ""
        self.fps_edit.setModified(False)
        self._show()

    def _video_opened(self) -> None:
        """A video without fps_true yet: look for it in the manifest above the video, then in the
        manifest the user pointed to last."""
        self._refused, self._stopwatch_draft = "", None
        self.fps_edit.setModified(False)  # what was being typed was for the video before
        video_path = self._controller.video_path
        if self._controller.session.time.fps_true is None:
            manifest = find_manifest(video_path)
            fps = None if manifest is None else usable_fps(video_path, manifest)
            if fps is None:
                manifest = settings().value(MANIFEST_KEY)
                fps = usable_fps(video_path, manifest) if isinstance(manifest, str) and manifest else None
            if fps is not None:
                self._take(fps, "manifest", Path(manifest))
                return
        self._show()

    def _show(self) -> None:
        """Bring every part in line with the session and with what was refused."""
        session = self._controller.session
        for part in (self.fps_edit, self.stopwatch_button, self.manifest_button):
            part.setEnabled(session is not None)
        fps = None if session is None else session.time.fps_true
        self.message.show_text("problem", self._refused)
        typed_wrong = self._refused in (NO_NUMBER, NOT_POSITIVE)
        # what is being typed, and what was refused, stays to be seen; so does "240" for 240.0
        if not (typed_wrong or self.fps_edit.isModified()) and float_or_none(self.fps_edit.text()) != fps:
            self.fps_edit.setText("" if fps is None else fps_text(fps))
        under = fps is not None and fps < WARN_BELOW
        theme.set_property(self.fps_edit, "check", "error" if typed_wrong else "warn" if under else "")
        source = "" if fps is None else SOURCE_WORDS.get(session.time.source, session.time.source)
        self.source_label.setText(source)
        if fps is None:
            hint = self._start_hint
        else:
            hint = HINT_UNDER if under else HINT_DONE.format(fps=fps, source=source)
        self._panel.set_state("attention" if typed_wrong or under else "todo" if fps is None else "done")
        self._panel.set_hint(hint)


def float_or_none(text: str) -> float | None:
    """The number a text stands for, or None when it is none."""
    try:
        return float(text)
    except ValueError:
        return None


def build(window) -> TimePanel:
    """The controls of panel 2 for `window` (a `MainWindow`), connected to its controller. fps_true
    is in frames per second."""
    return TimePanel(window)
