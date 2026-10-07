"""Panel 1, "Student and video" (SPEC 2 steps 1 to 3, 8.1, 10.1): the student's name, Open video
and Open session, what the file says about itself, the warnings of last week's check of the video,
the clip (start, end, step), and where the session is saved.

`build(window)` makes the panel's controls and connects them to the window's `SessionController`:
a value typed here goes into the session (`controller.touch()` then tells everyone), and whatever
changes the session is shown here. It also gives the window what belongs to saving: what the File
menu's Open session and Save session do, the save when the window closes, and the one dialog that
tells of a save that went wrong. The small parts that the panels share (`Message`, `ElidedLabel`,
`guard_wheel`, `field_rows`) are in gui/panel_parts.py, and importable from here as before.

The name is the controller's: one that code gave it (`set_student`, an opened session) shows in the
field, and the field's text becomes the name only if it was changed since the two last agreed, so a
field nobody touched never replaces a newer name. While a tracking job or an export runs
(`Jobs.writing`) the name, the clip and the two Open buttons are off, with the reason as their tooltip.

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
from PySide6.QtCore import QEvent, QObject, Qt, QThread, Signal  # noqa: F401 (see the next comment)
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,  # noqa: F401
                               QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from outline_tracker import schema, video
from outline_tracker.gui import dialogs, theme
from outline_tracker.gui.navigation import LARGEST_FRAME
# The parts every panel shares have their own file; their names, and the names they needed, are passed on:
# whatever this module offered before the two were split is still importable from it.
from outline_tracker.gui.panel_parts import (CONTROL_HEIGHT, LABEL_WIDTH, SPACING, ElidedLabel,  # noqa: F401
                                             Message, _WheelGuard, field_label, field_rows, guard_wheel)
from outline_tracker.gui.worker_jobs import RUNNING, jobs_of  # noqa: F401

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
    `start_box`, `end_box`, `step_box`, `clip_message`, `folder_label` (the run folder's name:
    where the files are; its tooltip is the whole path), `save_message`; `video_part`, which holds
    everything from `file_label` on and is hidden until a video is open; the File menu's
    `open_session_action` and `save_action` (made by gui/menus.py, connected here). `check` is the
    `VideoCheck` of the open video, None until its check is over and when it could not be made;
    `check_failed` says that it could not be made (`warnings_label` then says so, and the panel
    needs attention); `check_threads` are the checks that were started and may still run."""

    def __init__(self, window):
        super().__init__()
        self._window, self._controller, self._panel = window, window.controller, window.panels[0]
        self._start_hint = self._panel.hint.text()
        self._clip_error: tuple[str, str] | None = None
        self._name_shown = ""  # the name on which the field and the controller agreed last
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
        boxes = (self.start_box, self.end_box, self.step_box)
        for box, tip in zip(boxes, tips):
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

        # the File menu's two items that this panel has the functions for (gui/menus.py made them)
        self.open_session_action, self.save_action = window.menus.open_session_action, window.menus.save_action

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
        # the parts that are off while a run or an export is going, each with its own tooltip to put back after it
        self._lockable = {part: part.toolTip() for part in (self.name_edit, self.open_video_button,
                                                            self.open_session_button, *boxes)}
        self._jobs = jobs_of(window)  # after `closing` was connected: the session is saved before the worker stops
        self._jobs.writing_changed.connect(self._show)
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
        """The window closes: save what is due, and wait for the checks that still read a video
        file. A session that is on the disk as it is, is not written again: the time of
        session.json tells panel 9 whether the exported files are older than the session."""
        controller, folder = self._controller, self._controller.run_folder
        controller.about_to_save.emit()  # what is typed and not entered yet counts, as for any save
        on_disk = folder is not None and (folder / schema.SESSION_JSON).is_file()
        if controller.save_timer.isActive() or controller.save_problem is not None or not on_disk:
            controller.save_now()
        for thread in self.check_threads:
            thread.wait()

    def _name_typed(self) -> None:
        """The field's text counts if it was changed since it last agreed with the controller's name."""
        name = self.name_edit.text().strip()
        if name != self.name_edit.text():
            self.name_edit.setText(name)
        if name != self._name_shown and not self._jobs.writing():
            self._name_shown = name
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
        if controller.student:  # an opened session's name, or the one given before the video was opened
            self._name_shown = controller.student
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

    def _show(self, *_) -> None:
        """Bring every part in line with the session, the check, the last save and the lock."""
        controller, session, busy = self._controller, self._controller.session, self._jobs.writing()
        if controller.student != self._name_shown:  # given by code, or an opened session's: the newer one
            self._name_shown = controller.student
            self.name_edit.setText(controller.student)
        named, boxes = bool(controller.student), (self.start_box, self.end_box, self.step_box)
        for part, tip in self._lockable.items():
            part.setEnabled(not busy and (session is not None or part not in boxes))
            part.setToolTip(busy or tip)
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
