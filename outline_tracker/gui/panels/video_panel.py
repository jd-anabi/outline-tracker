"""Panel 1, "Student and video" (SPEC 2 steps 1 to 3, 8.1, 10.1): the student's name, Open video
and Open session, what the file says about itself, the warnings of last week's check of the video,
the clip (start, end, step), and where the session is saved.

`build(window)` makes the panel's controls and connects them to the window's `SessionController`:
a value typed here goes into the session (`controller.touch()` then tells everyone), and whatever
changes the session is shown here. It also gives the window what belongs to saving: the File menu's
Open session and Save session, the save when the window closes, and the one dialog that tells of a
save that went wrong. Three small parts are shared with the other panels of this task: `Message`
(a line of text in a tinted box), `ElidedLabel` (a file's name, cut in the middle when it is too
long) and `guard_wheel` (a box that the mouse wheel changes only while it has the keyboard).

`video.check_video` reads about 120 frames of a long video (measured: 0.6 s for 10 s of 1080p at
240 frames per second), so it runs in a thread of its own; its result comes back as a signal. A
check that could not be made is said like a warning: the panel is never done for a video that was
not checked.

Frames are video frame numbers counted from 0; the clip's step is in frames; the frame size is in
px of the decoded frame; the file's frame rate is in frames per second and is what the file states,
never fps_true. Lengths of the layout are Qt's device-independent px.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QEvent, QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QSpinBox,
                               QVBoxLayout, QWidget)

from outline_tracker import video
from outline_tracker.gui import dialogs, theme
from outline_tracker.gui.navigation import LARGEST_FRAME

LABEL_WIDTH = 120     # the column of the field labels
SPACING = 8           # between two rows, and between two buttons
CONTROL_HEIGHT = 28   # a field, a box
SESSION_FILTERS = ["Session (*.json)", "All files (*)"]
NOT_WRITTEN = "A file could not be written"  # the heading of the dialog; what to do follows it

NAME_MISSING = "Type your name in panel 1 first. The run folder is named after you. Nothing is saved until then."
HINT_NAME = "Type your name. The run folder is named after you."
HINT_VIDEO = "Open your video (a _tracker.mp4 file)."
HINT_DONE = "{file} · {width} × {height} px · {frames} frames · clip {start} to {end}, step {step}."
HINT_WARNINGS = ("This video has 1 warning. Read it below before you continue.",
                 "This video has {n} warnings. Read them below before you continue.")
# In place of the check's warnings, when the check itself could not be made.
NOT_CHECKED = ("This video could not be checked. Check that the file is still on this computer, not only in a cloud "
               "folder, and open it again.")
# What `video.check_video` raises for a file that went away or cannot be decoded.
CHECK_ERRORS = (OSError, cv2.error, ValueError)
NO_VALUE = "–"


def size_text(n_bytes: int) -> str:
    """A file's size in bytes as text in decimal units: "999 bytes", "601 kB", "12.3 MB", "2.50 GB"."""
    if n_bytes < 1_000:
        return f"{n_bytes} bytes"
    if n_bytes < 1_000_000:
        return f"{n_bytes / 1e3:.0f} kB"
    if n_bytes < 1_000_000_000:
        return f"{n_bytes / 1e6:.1f} MB"
    return f"{n_bytes / 1e9:.2f} GB"


def clip_problem(start: int, end: int, step: int, n_frames: int) -> tuple[str, str] | None:
    """Why a clip cannot be used on a video of `n_frames` frames: (the field at fault: `start`,
    `end` or `step`, the sentence for the user), or None for a clip that can be used. `start` and
    `end` are video frame numbers counted from 0 (the end is part of the clip), `step` is in frames."""
    last = n_frames - 1
    if start > last:
        return "start", f"The start frame must be {last} or less. The video has {n_frames} frames."
    if end < start:
        return "end", f"The end frame ({end}) is before the start frame ({start})."
    if end > last:
        return "end", f"The end frame must be {last} or less. The video has {n_frames} frames."
    if step < 1:
        return "step", "The step must be 1 frame or more."
    return None


class Message(QLabel):
    """A line of text in a tinted box, under the control it is about: `show_text(kind, text)` with
    kind `problem` or `warning`; an empty text hides it. It wraps, and is never wider than its room."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "msg")
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.hide()

    def show_text(self, kind: str, text: str) -> None:
        self.setText(text)
        theme.set_property(self, "kind", kind)
        self.setVisible(bool(text))


class ElidedLabel(QLabel):
    """A label for the name of a file or a folder: `full_text` is the name, and what shows is cut
    in the middle when the label is too narrow for it. The tooltip holds the name, or what
    `set_full_text` was given as `tip` (the whole path)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.full_text = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def set_full_text(self, text: str, tip: str | None = None) -> None:
        self.full_text = text
        self.setToolTip(text if tip is None else tip)
        self._cut()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._cut()

    def _cut(self) -> None:
        self.setText(self.fontMetrics().elidedText(self.full_text, Qt.TextElideMode.ElideMiddle, self.width()))


class _WheelGuard(QObject):
    """Passes the mouse wheel on to what is behind a box while the box does not have the keyboard."""

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Wheel and not watched.hasFocus():
            event.ignore()  # the dock scrolls instead
            return True
        return False


def guard_wheel(box: QWidget) -> None:
    """Make `box` (a spin box, a combo box) react to the mouse wheel only while it has the keyboard:
    the dock scrolls with the wheel, and that must never change a value."""
    box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # the wheel alone does not give it the keyboard
    box.installEventFilter(_WheelGuard(box))


def field_label(text: str) -> QLabel:
    """The label of a field, for the left column of a panel's rows."""
    label = QLabel(text)
    label.setFixedWidth(LABEL_WIDTH)
    return label


def field_rows(fields) -> QGridLayout:
    """The rows of a panel: each of `fields` is (label, widget), the label a text or a label made
    with `field_label`, in the left column, the widget filling the rest; or (None, widget) for a
    widget over both columns. A field that is typed in is the buddy of its label."""
    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(12)
    grid.setVerticalSpacing(SPACING)
    grid.setColumnStretch(1, 1)
    for row, (word, part) in enumerate(fields):
        if word is None:
            grid.addWidget(part, row, 0, 1, 2)
            continue
        label = field_label(word) if isinstance(word, str) else word
        grid.addWidget(label, row, 0)
        grid.addWidget(part, row, 1)
        if isinstance(part, (QLineEdit, QSpinBox)):
            label.setBuddy(part)
            part.setMinimumHeight(CONTROL_HEIGHT)
    return grid


class CheckThread(QThread):
    """Last week's check of one video (`video.check_video`), off the GUI thread. `done(path, check)`
    is emitted when it is over, however it ended: the video's path and its `VideoCheck`, or None
    when the check could not be made. An error that is not a bad file's (`CHECK_ERRORS`) is a
    mistake in the program: it is not caught here, so Python reports it with its trace."""

    done = Signal(object, object)

    def __init__(self, path: Path, parent: QObject | None = None):
        super().__init__(parent)
        self.path = path

    def run(self) -> None:
        check = None
        try:
            check = video.check_video(self.path)
        except CHECK_ERRORS:  # the file went away, or cannot be decoded: the panel says so
            pass
        finally:  # also on the way out with any other error: the video was not checked then either
            self.done.emit(self.path, check)


class VideoPanel(QWidget):
    """The controls of panel 1. Parts: `name_edit`, `name_message`, `open_video_button`,
    `open_session_button`, `file_label`, `size_label`, `frame_size_label`, `frames_label`,
    `file_fps_word` and `file_fps_label` (the frame rate the file states), `warnings_label`,
    `start_box`, `end_box`, `step_box`, `clip_message`, `folder_label` (the run folder's name),
    `save_message`; `video_part`, which holds everything from `file_label` on and is hidden until
    a video is open; the File menu's `open_session_action` and `save_action`. `check` is the
    `VideoCheck` of the open video, None until its check is over and when it could not be made;
    `check_failed` says that it could not be made (`warnings_label` then says so, and the panel
    needs attention); `check_threads` are the checks that were started and may still run."""

    def __init__(self, window):
        super().__init__()
        self._window, self._controller, self._panel = window, window.controller, window.panels[0]
        self._start_hint = self._panel.hint.text()
        self._clip_error: tuple[str, str] | None = None
        self.check, self.check_failed = None, False
        self.check_threads: list[CheckThread] = []

        self.name_edit = QLineEdit()
        self.name_edit.setToolTip("Your name. The run folder is named after you.")
        self.name_message = Message()
        self.open_video_button, self.open_session_button = QPushButton("Open video"), QPushButton("Open session")
        self.open_video_button.setToolTip("Choose the video to track")
        self.open_session_button.setToolTip("Choose the session.json of a run folder, to go on with it")
        self.file_label, self.folder_label = ElidedLabel(), ElidedLabel()
        self.size_label, self.frame_size_label, self.frames_label = QLabel(), QLabel(), QLabel()
        self.file_fps_word, self.file_fps_label = field_label("The file says"), QLabel()
        for part in (self.file_fps_word, self.file_fps_label):
            part.setToolTip("The frame rate written in the file. Times come from fps_true (panel 2), never from this.")
        self.warnings_label, self.clip_message, self.save_message = Message(), Message(), Message()
        self.start_box, self.end_box, self.step_box = QSpinBox(), QSpinBox(), QSpinBox()
        tips = ("First frame of the clip", "Last frame of the clip", "Every how many frames one is tracked")
        for box, tip in zip((self.start_box, self.end_box, self.step_box), tips):
            box.setRange(0, LARGEST_FRAME)  # a wide limit only: what cannot be used is said, never changed
            box.setKeyboardTracking(False)  # a typed number counts when it is complete (Enter, or leaving)
            box.setAlignment(Qt.AlignmentFlag.AlignRight)
            box.setToolTip(tip)
            box.setEnabled(False)
            guard_wheel(box)
            box.valueChanged.connect(self._clip_typed)

        buttons = QHBoxLayout()
        buttons.setSpacing(SPACING)
        for button in (self.open_video_button, self.open_session_button):
            button.setAutoDefault(False)
            buttons.addWidget(button)
        buttons.addStretch(1)
        self.video_part = QWidget()  # what there is to show of a video: hidden until one is open
        rows = QVBoxLayout(self)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(SPACING)
        rows.addLayout(field_rows((("Name", self.name_edit), (None, self.name_message))))
        rows.addLayout(buttons)
        rows.addWidget(self.video_part)
        self.video_part.setLayout(field_rows((
            ("File", self.file_label), ("Size", self.size_label), ("Frame size", self.frame_size_label),
            ("Frames", self.frames_label), (self.file_fps_word, self.file_fps_label), (None, self.warnings_label),
            ("Start frame", self.start_box), ("End frame", self.end_box), ("Step", self.step_box),
            (None, self.clip_message), ("Run folder", self.folder_label), (None, self.save_message))))

        self.open_session_action, self.save_action = QAction("Open session", window), QAction("Save session", window)
        self.save_action.setShortcut(QKeySequence(QKeySequence.StandardKey.Save))
        (menu,) = [menu for menu in (entry.menu() for entry in window.menuBar().actions())
                   if menu is not None and window.open_action in menu.actions()]
        entries = menu.actions()
        after_open_video = entries[entries.index(window.open_action) + 1]
        menu.insertActions(after_open_video, [self.open_session_action, self.save_action])

        controller = self._controller
        self.name_edit.editingFinished.connect(self._name_typed)
        self.open_video_button.clicked.connect(window.open_action.trigger)
        self.open_session_button.clicked.connect(self.open_session_action.trigger)
        self.open_session_action.triggered.connect(self.choose_session)
        self.save_action.triggered.connect(self.save)
        controller.video_opened.connect(self._video_opened)
        controller.session_changed.connect(self._show)
        controller.about_to_save.connect(self._name_typed)
        controller.saved.connect(self._show)
        controller.trouble.connect(lambda text: dialogs.message(window, "problem", f"{NOT_WRITTEN}\n{text}"))
        window.closing.connect(self._closing)
        self._show()

    def choose_session(self) -> None:
        """Ask which session file to open; the chosen file goes to the window's `open_path`."""
        dialogs.open_file(self._window, "Open session", SESSION_FILTERS, self._window.open_path)

    def save(self) -> None:
        """Save the session now (the Save key), and say in the status bar where it is, or what is
        missing."""
        controller = self._controller
        written = controller.save_now()
        if controller.save_problem is not None:
            said = controller.save_problem
        elif written is not None:
            said = f"The session is saved in the run folder {controller.run_folder.name}."
        else:  # no video, or no name
            said = controller.refusal("export") or ""
        self._window.statusBar().showMessage(said)

    def _closing(self) -> None:
        """The window closes: save, and wait for the checks that still read a video file."""
        self._controller.save_now()
        for thread in self.check_threads:
            thread.wait()

    def _name_typed(self) -> None:
        name = self.name_edit.text().strip()
        if name != self.name_edit.text():
            self.name_edit.setText(name)
        self._controller.set_student(name)
        self._show()

    def _clip_typed(self) -> None:
        session = self._controller.session
        if session is None:
            return
        typed = (self.start_box.value(), self.end_box.value(), self.step_box.value())
        self._clip_error = clip_problem(*typed, session.video.n_frames)
        clip = session.clip
        if self._clip_error is None and typed != (clip.start, clip.end, clip.step):
            clip.start, clip.end, clip.step = typed
            self._controller.touch()
        else:
            self._show()

    def _video_opened(self) -> None:
        controller = self._controller
        if controller.student:  # an opened session's name, or the one typed before the video was opened
            self.name_edit.setText(controller.student)
        else:  # a name that is typed and not entered yet belongs to the new session
            self._name_typed()
        self._clip_error, self.check, self.check_failed = None, None, False
        self.check_threads = [thread for thread in self.check_threads if thread.isRunning()]
        thread = CheckThread(controller.video_path, self)
        thread.done.connect(self._checked)
        self.check_threads.append(thread)
        thread.start()
        self._show()

    def _checked(self, path, check) -> None:
        if path == self._controller.video_path:  # else: of a video that is no longer the open one
            self.check, self.check_failed = check, check is None
            self._show()

    def _show(self) -> None:
        """Bring every part in line with the session, the check and the last save."""
        controller, session = self._controller, self._controller.session
        named, boxes = bool(controller.student), (self.start_box, self.end_box, self.step_box)
        for box in boxes:
            box.setEnabled(session is not None)
        self.video_part.setVisible(session is not None)
        self.name_message.show_text("problem", NAME_MISSING if session is not None and not named else "")
        folder = controller.run_folder
        self.folder_label.set_full_text(NO_VALUE if folder is None else folder.name,
                                        "" if folder is None else str(folder))
        self.save_message.show_text("problem", controller.save_problem or "")
        if session is None:
            self._panel.set_state("todo")
            self._panel.set_hint(HINT_VIDEO if named else self._start_hint)
            return
        info, clip = session.video, session.clip
        self.file_label.set_full_text(controller.video_path.name)
        self.size_label.setText(size_text(info.size))
        self.frame_size_label.setText(f"{info.width} × {info.height} px")
        self.frames_label.setText(str(info.n_frames))
        self.file_fps_label.setText(f"{info.fps_container:.2f} fps")
        warnings = [NOT_CHECKED] if self.check_failed else [] if self.check is None else self.check.warnings
        self.warnings_label.show_text("warning", "\n\n".join(warnings))
        at_fault, sentence = self._clip_error or (None, "")
        self.clip_message.show_text("problem", sentence)
        for name, box in zip(("start", "end", "step"), boxes):
            theme.set_property(box, "check", "error" if name == at_fault else "")
        if self._clip_error is None:  # else what was typed stays to be seen
            for box, value in zip(boxes, (clip.start, clip.end, clip.step)):
                box.blockSignals(True)
                box.setValue(value)
                box.blockSignals(False)
        if warnings:
            hint = HINT_WARNINGS[len(warnings) > 1].format(n=len(warnings))
        elif not named:
            hint = HINT_NAME
        else:
            hint = HINT_DONE.format(file=controller.video_path.name, width=info.width, height=info.height,
                                    frames=info.n_frames, start=clip.start, end=clip.end, step=clip.step)
        attention = bool(warnings) or self._clip_error is not None or not self.save_message.isHidden()
        self._panel.set_state("attention" if attention else "done" if named else "todo")
        self._panel.set_hint(hint)


def build(window) -> VideoPanel:
    """The controls of panel 1 for `window` (a `MainWindow`), connected to its controller, its File
    menu and its `closing` signal. No quantities are computed here."""
    return VideoPanel(window)
