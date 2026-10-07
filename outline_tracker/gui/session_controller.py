"""The owner of the open video and of its session (SPEC 8.10, 10.1): one per window.

The window and its panels all work on the same `Session` and read frames from the same open file;
this object holds both and says when they change. A panel reads `controller.session`, changes it,
and calls `touch()`; whoever shows something of the session listens to `session_changed`. Nothing
is written to disk here: the run folder and saving come with the task that opens sessions.

Units and coordinates are the session's (SPEC 3): px in Tracker's image coordinates (origin at the
top-left corner of the frame, u to the right, v downward, pixel centers at +0.5), frames as video
frame numbers counted from 0. No measurement is defined here.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Signal

from outline_tracker.fileio import sha256_first_64mib
from outline_tracker.frame_source import FrameSource
from outline_tracker.session import Clip, Session, VideoRef

# What to do about a file that is no video the tracker can read, after "<name> could not be read."
WHAT_TO_DO = ("Check that the file is on this computer, not only in a cloud folder, and that it is a "
              "_tracker.mp4 file. For a .MOV file, run “outline-tracker convert” on it first.")


class SessionController(QObject):
    """The open video and its session, or nothing yet.

    `session`: the `session.Session` being worked on, None before a video is open. `video_path`:
    the open video's `pathlib.Path`. `info`: what the file says about itself (`video.probe`: frame
    size in px, frame count, the file's own frame rate). `source`: the `FrameSource` that reads
    single frames of it by number. `run_folder`: where the session's files go, None until there is
    one. `video_opened` is emitted when a video was opened, `session_changed` when the session was
    changed (`touch`).
    """

    video_opened = Signal()
    session_changed = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.session: Session | None = None
        self.video_path: Path | None = None
        self.info = None
        self.source: FrameSource | None = None
        self.run_folder: Path | None = None

    def open_video(self, path) -> None:
        """Open the video at `path` and start a new session for it, then emit `video_opened`.

        The session holds the video's facts (where it is, its size in bytes, the hash of its first
        64 MiB, the frame size in px, the frame count, the file's own frame rate in frames per
        second) and a clip from frame 0 to the last frame with the default step; everything else
        is as a new session has it. A video that was open is released. Raises `ValueError`, with a
        message for the user that names the file without its folder, when the file is missing or
        is no video that can be read; what was open then stays as it was.
        """
        path = Path(path)
        try:
            source = FrameSource(path)  # probes the file, and keeps it open
        except (OSError, cv2.error) as error:  # not there, or OpenCV cannot open it or decode its first frame
            raise ValueError(f"{path.name} could not be read. {WHAT_TO_DO}") from error
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
        self.close()
        self.session = Session(video=video, clip=Clip(start=0, end=info.n_frames - 1))
        self.video_path, self.info, self.source = path, info, source
        self.video_opened.emit()

    def touch(self) -> None:
        """Say that `session` was changed: emit `session_changed`. Call it after changing a value of
        the session, so that every part that shows the session shows the new value."""
        self.session_changed.emit()

    def close(self) -> None:
        """Release the video file (on Windows a file cannot be moved or deleted while it is open).
        `source` is None afterwards; the session stays. Calling it again does nothing."""
        if self.source is not None:
            self.source.close()
            self.source = None
