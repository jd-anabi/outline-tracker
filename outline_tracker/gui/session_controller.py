"""The owner of the open video and of its session (SPEC 8.1, 8.10, 10.1): one per window.

The window and its panels all work on the same `Session` and read frames from the same open file;
this object holds both and says when they change. A panel reads `controller.session`, changes it,
and calls `touch()`; whoever shows something of the session listens to `session_changed`.

The session is kept on disk as `session.json` in the run folder, which is SPEC 8.1's default for
the video and the student's name (`run_folder.default_run_folder`), or the folder a session was
opened from. It is written 0.75 s after the last change, and at once by `save_now()` (the Save
key, closing the window, before a run, on export). Nothing is written while the name is empty, and
a `session.json` that this window neither opened nor wrote is never written over. Every write
happens in the thread this object lives in, the GUI thread.

Units and coordinates are the session's (SPEC 3): px in Tracker's image coordinates (origin at the
top-left corner of the frame, u to the right, v downward, pixel centers at +0.5), frames as video
frame numbers counted from 0, fps_true in frames per second. No measurement is defined here.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, QTimer, Signal

from outline_tracker import schema
from outline_tracker.fileio import new_name, sha256_first_64mib
from outline_tracker.frame_source import FrameSource
from outline_tracker.run_folder import default_run_folder
from outline_tracker.session import Clip, Session, SessionVersionError, VideoNotFoundError, VideoRef, WrongVideoError

SAVE_DELAY_MS = 750  # session.json is written this long after the last change
ACTIONS = ("track", "export")

# What to do about a file that is no video the tracker can read, after "<name> could not be read."
WHAT_TO_DO = ("Check that the file is on this computer, not only in a cloud folder, and that it is a "
              "_tracker.mp4 file. For a .MOV file, run “outline-tracker convert” on it first.")
# What is missing before tracking or an export can start, and before anything is saved.
NAME_FIRST = "Type your name in panel 1 first."
VIDEO_FIRST = "Open your video in panel 1 first."
# Why a session file was not opened.
NO_SESSION = "{name} is not a session file of Outline Tracker. Choose the session.json of a run folder."
NOT_READ = "{name} could not be read. Check that the file is on this computer, not only in a cloud folder."
NOT_FOUND = "The video {name} was not found. Choose where it is now."
# Why the session is not in session.json now.
HELD = "The folder {folder} holds a session already. Open it with Open session, or type another name."
NOT_WRITTEN = ("{name} could not be written to the folder {folder}. Check that the folder is not read-only and "
               "that the disk is not full. Nothing is saved until this works.")
LOCKED = ("{name} is open in another program. Close it there. Until then the session is saved as {beside} in "
          "the run folder.")


class SessionController(QObject):
    """The open video and its session, or nothing yet.

    `session`: the `session.Session` being worked on, None before a video is open. `video_path`:
    the open video's `pathlib.Path`. `info`: what the file says about itself (`video.probe`: frame
    size in px, frame count, the file's own frame rate). `source`: the `FrameSource` that reads
    single frames of it by number. `student`: the student's name, which may be given before a
    video is open (`set_student`); the session has the same. `run_folder`: where the session's
    files go, None while there is no video or the name is empty. `saved_path`: the file the last save wrote (None before the first
    one, and after one that wrote nothing). `save_problem`: the sentence that says why the session
    is not in `session.json` after the last save, None when it is, or when nothing was to be
    saved. `save_timer`: the single-shot timer of the delayed save.

    `video_opened` is emitted when a video or a session was opened, `session_changed` when the
    session was changed (`touch`), `about_to_save` when `save_now` is called, before it looks at the
    session (a panel then puts in what is typed and not entered yet), `saved` after every save
    that was tried, and `trouble(text)` when a save has something to tell the user (`save_problem`):
    once for the same text in a row.
    """

    video_opened = Signal()
    session_changed = Signal()
    about_to_save = Signal()
    saved = Signal()
    trouble = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.session: Session | None = None
        self.video_path: Path | None = None
        self.info = None
        self.source: FrameSource | None = None
        self.student = ""
        self.run_folder: Path | None = None
        self.saved_path: Path | None = None
        self.save_problem: str | None = None
        self._own: set[Path] = set()     # the session.json files this window opened or wrote
        self._told: str | None = None    # what `trouble` said last, until a save goes well
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(SAVE_DELAY_MS)
        self.save_timer.timeout.connect(self._save)

    def open_video(self, path) -> None:
        """Open the video at `path` and start a new session for it, then emit `video_opened`.

        The session holds the video's facts (where it is, its size in bytes, the hash of its first
        64 MiB, the frame size in px, the frame count, the file's own frame rate in frames per
        second), a clip from frame 0 to the last frame with the default step, and the student's
        name if there is one (`student`), which also gives it its run folder; everything else is as
        a new session has it. A video that was open is released, after a save of its session that
        was still due. Raises `ValueError`, with a message for the user that names the file
        without its folder, when the file is missing or is no video that can be read; what was
        open then stays as it was.
        """
        path = Path(path)
        source = self._read(path)
        info = source.info
        try:
            if info.n_frames < 1:
                raise OSError(f"{path} reports {info.n_frames} frames.")
            video = VideoRef(abspath=os.path.abspath(path), size=path.stat().st_size,
                             sha256_first_64mib=sha256_first_64mib(path), width=info.width, height=info.height,
                             n_frames=info.n_frames, fps_container=info.fps_container)
        except OSError as error:
            source.close()
            raise ValueError(f"{path.name} could not be read. {WHAT_TO_DO}") from error
        session = Session(student=self.student, video=video, clip=Clip(start=0, end=info.n_frames - 1))
        self._take(session, path, source, None)
        if self.run_folder is not None:
            self.save_timer.start()  # a named session: its run folder gets its session.json

    def open_session(self, path, video=None) -> None:
        """Open the session file at `path` (a session.json) with its video, then emit `video_opened`.

        The folder of the file is the run folder. The video is looked for where the session says
        (`Session.locate_video`: the path relative to the run folder, then the absolute one), or
        taken from `video`, the user's answer to where it is now; either way it must have the
        session's size in bytes and hash. Its new place is put into the session, to be saved.

        Raises `session.VideoNotFoundError` (a `FileNotFoundError`) when the video is at none of
        the session's places: ask the user and call again with `video`. Raises `ValueError`, with
        a message for the user that names files without their folders, when the file is missing,
        is no session, has another schema version, or when the video is another file than the
        session's or cannot be read. What was open stays as it was in each of these cases.
        """
        path = Path(path)
        try:
            session = Session.load(path)
        except SessionVersionError as error:
            raise ValueError(str(error).replace(str(path), path.name)) from error
        except ValueError as error:
            raise ValueError(NO_SESSION.format(name=path.name)) from error
        except OSError as error:
            raise ValueError(NOT_READ.format(name=path.name)) from error
        run_folder = Path(os.path.abspath(path)).parent
        try:
            if video is None:
                video_path = session.locate_video(run_folder)
            else:
                video_path = Path(video)
                session.video.check(video_path)
        except VideoNotFoundError as error:
            stored = session.video.relpath or session.video.abspath
            name = stored.replace("\\", "/").rsplit("/", 1)[-1] or "of this session"
            raise VideoNotFoundError(NOT_FOUND.format(name=name)) from error
        except WrongVideoError as error:
            wrong = error.args[0].split(" is not the video of this session")[0]
            raise ValueError(str(error).replace(wrong, Path(wrong).name)) from error
        source = self._read(video_path)
        before = (session.video.relpath, session.video.abspath)
        session.video.point_to(video_path, run_folder)
        self._take(session, video_path, source, run_folder)
        if (session.video.relpath, session.video.abspath) != before:
            self.save_timer.start()  # the video is somewhere else now: write where

    def _read(self, path: Path) -> FrameSource:
        """The open `FrameSource` of the video at `path`, or the `ValueError` for the user."""
        try:
            return FrameSource(path)  # probes the file, and keeps it open
        except (OSError, cv2.error) as error:  # not there, or OpenCV cannot open it or decode its first frame
            raise ValueError(f"{path.name} could not be read. {WHAT_TO_DO}") from error

    def _take(self, session: Session, video_path: Path, source: FrameSource, run_folder: Path | None) -> None:
        """Replace what is open by a session and its video. `run_folder` is the folder the session
        was opened from; None for a new session, which gets the default for its name."""
        if self.save_timer.isActive():
            self._save()  # what was changed in the session that is left
        self.close()
        self.session, self.video_path, self.info, self.source = session, video_path, source.info, source
        self.student = session.student
        self.run_folder = self._default_folder() if run_folder is None else run_folder
        self.saved_path = self.save_problem = self._told = None
        if run_folder is not None:
            self._own.add(run_folder / schema.SESSION_JSON)
        self.video_opened.emit()

    def _default_folder(self) -> Path | None:
        """SPEC 8.1's run folder for the open video and the student's name, as an absolute path:
        `<video folder>/<video stem>_outline_<name>`. None while the name is empty."""
        if not self.student.strip():
            return None
        return Path(os.path.abspath(default_run_folder(self.video_path, self.student)))

    def set_student(self, name: str) -> None:
        """Take the student's name as typed (blanks around it are dropped). With a video open it
        goes into the session, which then says that it changed (`touch`); without one it is kept
        for the session of the video that is opened next."""
        name = name.strip()
        if self.session is None:
            self.student = name
        elif name != self.session.student:
            self.session.student = name
            self.touch()

    def touch(self) -> None:
        """Say that `session` was changed: emit `session_changed`. Call it after changing a value of
        the session, so that every part that shows the session shows the new value. The run folder
        follows the student's name (a new name is a new folder beside the video and the folder of
        the old name is left as it is; no name, no folder), and the session is saved 0.75 s after
        the last call."""
        if self.session.student != self.student:
            self.student = self.session.student
            self.run_folder = self._default_folder()
            self.save_problem = None  # it was about the folder that is left
        self.session_changed.emit()
        if self.run_folder is None:
            self.save_timer.stop()
        else:
            self.save_timer.start()

    def save_now(self) -> Path | None:
        """Write the session to `session.json` in the run folder now, in the calling thread (the GUI
        thread), and emit `saved`: for the Save key, the closing window, before a run and on export.
        `about_to_save` is emitted first, so that what is typed and not entered yet is saved too.
        Returns the file written: `session.json`, or `session.new.json` beside it while another
        program holds `session.json` open; None when nothing was written.

        Nothing is written without a video or while the student's name is empty (`refusal` says
        which), into a folder whose `session.json` this window neither opened nor wrote, or when
        the file cannot be written; `save_problem` then says why, and nothing is raised. The video's
        path relative to the run folder is put into the session first (SPEC 8.1).
        """
        self.about_to_save.emit()
        return self._save()

    def _save(self) -> Path | None:
        """`save_now` without asking for what is typed and not entered: the delayed save, which
        must not take half a name for the run folder."""
        self.save_timer.stop()
        if self.session is None or self.run_folder is None:
            return None
        target = self.run_folder / schema.SESSION_JSON
        names = {"name": target.name, "folder": self.run_folder.name, "beside": new_name(target).name}
        written = problem = None
        if target not in self._own and target.exists():
            problem = HELD.format(**names)
        else:
            self.session.video.point_to(self.video_path, self.run_folder)
            try:
                written = self.session.save(target)
                self._own.add(target)
            except ValueError as error:  # a number in the session that a file must never hold
                problem = f"{target.name} could not be written. {error}"
            except OSError:
                problem = NOT_WRITTEN.format(**names)
        if written is not None and written != target:
            problem = LOCKED.format(**names)
        self.saved_path, self.save_problem = written, problem
        if problem is not None and problem != self._told:
            self.trouble.emit(problem)
        self._told = problem
        self.saved.emit()
        return written

    def refusal(self, action: str) -> str | None:
        """Why `action` (`track` or `export`) cannot start, as one plain sentence that says what to
        do first, or None when nothing is missing: first of all the student's name, which names
        the run folder, then the video. Any other action is a `ValueError`."""
        if action not in ACTIONS:
            raise ValueError(f"An action is one of {', '.join(ACTIONS)}, not {action!r}.")
        if not self.student.strip():
            return NAME_FIRST
        return VIDEO_FIRST if self.session is None else None

    def close(self) -> None:
        """Release the video file (on Windows a file cannot be moved or deleted while it is open),
        and drop a save that was still due: whoever closes saves first (`save_now`). `source` is
        None afterwards; the session stays. Calling it again does nothing."""
        self.save_timer.stop()
        if self.source is not None:
            self.source.close()
            self.source = None
