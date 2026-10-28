"""session.json <-> dataclasses (schema version 2), and finding the video again (SPEC 8.1).

One dataclass per block of session.json, with the field names of SPEC 8.10. A `Session` holds everything
needed to reopen a run, run it again and export again. The format is that of docs/design/formats.md; a file
of version 1, the format of 0.1.0, is read through outline_tracker/core/session_v1.py.

This module holds `Session` (load, save, `world_frame`, `locate_video`, and the reading and writing of the
whole file), the video block `VideoRef` with the errors of finding the video again, `SessionVersionError`, and
`OlderFileNotKept`: a save keeps a session file of an older version beside the new one first (rule 5 of
docs/ROADMAP.md section 2, decision 36).
It also holds two names about fps_true that the `probe` and `from-tracker` commands share: `UNKNOWN_FPS`, what
they say when fps_true is neither typed nor in a manifest, and `SLOW_MOTION_FPS`, below which they warn.
The other blocks (`Clip`, `TimeSettings`, `CalibrationSettings`, `Axes`, `Boundary`, `Processing`, `ProbeBox`,
`Prompt`, `Track`, `RunRecord`, `Correction`) and the base class that reads and writes them are in
outline_tracker/core/session_parts.py; they are imported here, so `from outline_tracker.core.session import Track`
works.

Units and coordinates (SPEC 3): every `*_px` value is in image pixels in Tracker's convention (origin at the
top-left corner of the frame, u to the right, v down, the center of the pixel in column c and row r at
(c + 0.5, r + 0.5)); a point is a list [u, v]. `*_mm` values are mm, `*_s` seconds, `*_deg` degrees. Frames
are the video's own frame numbers (frame 0 first); fps_true is in frames per second, t_s = frame / fps_true.

Rules of the file:
- Values are kept as JSON holds them (lists, not tuples; numbers as written), so load and save lose nothing.
  numpy numbers and arrays are converted when saving.
- Keys this version does not know are kept in each record's `extra` and written back.
- The keys added by decision X18 are optional, written only when they have a value: `calibration.tracker_fit`,
  a prompt's `decoder`, a run's `model_id` and `weights_sha256`.
- `video.relpath` is relative to the run folder (the folder that holds session.json), with forward slashes
  on every platform, so the folder can be moved together with its video.

No Qt, no torch.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any, ClassVar

from outline_tracker import __version__
from outline_tracker.core.fileio import atomic_write, keep_beside, new_name, relative_path, sha256_first_64mib
from outline_tracker.core.geometry import WorldFrame, stick_scale
from outline_tracker.core.schema import SESSION_SCHEMA_VERSION

# The blocks written in core/session_parts.py, imported here so that they can be imported from this module too
# (`Prompt` is the one that this file does not use itself).
from outline_tracker.core.session_parts import (Axes, Boundary, CalibrationSettings, Clip, Correction, ProbeBox, Processing,
                                           Prompt, RunRecord, TimeSettings, Track, _Record)
# MixedVersionError is imported for those that import the errors of a session file from this module
from outline_tracker.core.session_v1 import MixedVersionError, SessionKeyError, upgrade  # noqa: F401

SESSION_FORMAT = "outline-tracker-session"  # the "format" key of session.json
READ_VERSIONS = (1, SESSION_SCHEMA_VERSION)  # the schema versions this build reads; it writes the last
# What a save that kept the file of an older version says, in one line, once the new file is in place (decision 39)
KEPT_OLDER = ("{name} is saved in session format version {new} now. The file as it was, of version {old}, is kept "
              "beside it as {kept}: outline-tracker 0.1.0 opens it after it is renamed to {name}.")

# What the two commands that read fps_true from `--fps` or from a manifest say about it (decision X9). They are
# here because this is the lowest module that both `cli/probe.py` and `workflows/from_tracker.py` import and
# that has room: geometry.py, which holds the manifest lookup, is at 400 lines, and schema.py is pinned below 420.
# the wording of last week's script (tracker_io.make_plan), which from-tracker also gives
UNKNOWN_FPS = "Unknown fps_true: give it with --fps (e.g. --fps 239.6), or fill in data/manifest.csv."
SLOW_MOTION_FPS = 100.0  # an fps_true below this gets a warning (SPEC 3.3)


class SessionVersionError(ValueError):
    """session.json has a schema version that this tool does not read, or none. `found` is the file's
    "schema_version" (None when missing), `supported` the versions this tool reads (`READ_VERSIONS`)."""

    def __init__(self, found, where: str = "The session file", written_by=None):
        self.found, self.supported = found, READ_VERSIONS
        by = f" ({written_by})" if isinstance(written_by, str) and written_by else ""
        if not isinstance(found, int) or isinstance(found, bool):
            has, advice = f"no usable session format version ({found!r})", "It may be damaged."
        elif found > max(READ_VERSIONS):
            has = f"session format version {found}"
            advice = f"It was written by a newer outline-tracker{by}: update yours."
        else:
            has = f"session format version {found}"
            advice = f"It was written by an older outline-tracker{by}: open it with that one, or start a new session."
        read = " and ".join(str(version) for version in READ_VERSIONS)
        super().__init__(f"{where} has {has}, but this outline-tracker ({__version__}) reads only versions {read}. "
                         f"{advice}")


class OlderFileNotKept(OSError):
    """A save found a session file of an older schema version and could not keep a copy of it (decision 36), so
    it wrote nothing. The message names the files without their folders and says why."""

    def __init__(self, name: str, version: int, error: OSError):
        kept = Path(error.filename).name if isinstance(error.filename, str) else f"{Path(name).stem}.v{version}{Path(name).suffix}"
        super().__init__(f"{name} is of session format version {version} and is kept as {kept} before it is "
                         f"replaced, but {kept} could not be written ({error.strerror or error}). {name} was not "
                         "replaced. Check that the folder is not read-only and that the disk is not full.")


class VideoNotFoundError(FileNotFoundError):
    """The session's video is at none of the places the session knows: ask the user where it is."""


class WrongVideoError(ValueError):
    """A file is not the video the session was made with (another size, or other content)."""


# --------------------------------------------------------------------------- the video and the whole session

@dataclass
class VideoRef(_Record):
    """The video of a session, stored so that it can be found again (SPEC 8.1).

    relpath: relative to the run folder, forward slashes; None when there is none (another Windows drive).
    abspath: where the video was when the session was last pointed at it. size: bytes. sha256_first_64mib:
    hex digest of the first 64 MiB (key "sha256_first_64MiB" in the file). width, height: px, as decoded.
    n_frames: frames. fps_container: the file's own frame rate in frames per second, never used for physics.
    """

    relpath: str | None = None
    abspath: str = ""
    size: int = 0
    sha256_first_64mib: str = field(default="", metadata={"key": "sha256_first_64MiB"})
    width: int = 0
    height: int = 0
    n_frames: int = 0
    fps_container: float = 0.0

    @classmethod
    def from_file(cls, video, run_folder, *, width, height, n_frames, fps_container) -> VideoRef:
        """The reference to an existing video file: its size (bytes) and hash are read here, its paths are
        stored for `run_folder` (see `point_to`). width and height in px, n_frames in frames and
        fps_container in frames per second come from the caller's probe of the file."""
        ref = cls(size=Path(video).stat().st_size, sha256_first_64mib=sha256_first_64mib(video), width=int(width),
                  height=int(height), n_frames=int(n_frames), fps_container=float(fps_container))
        ref.point_to(video, run_folder)
        return ref

    def point_to(self, video, run_folder) -> None:
        """Store where the video is now: `abspath`, and `relpath` from the run folder (None when no relative
        path exists). Both arguments are paths; relative ones are taken from the current folder. Does not
        look at the file: call `check` first for a file the user picked."""
        video = os.path.abspath(video)
        self.relpath = relative_path(video, os.path.abspath(run_folder))
        self.abspath = video

    def check(self, path) -> None:
        """Make sure the file at `path` is this session's video: the same size in bytes and the same
        SHA-256 of the first 64 MiB. Raises VideoNotFoundError when there is no file there, and
        WrongVideoError when it is another file."""
        path = Path(path)
        if not path.is_file():
            raise VideoNotFoundError(f"There is no video file at {path}.")
        size = path.stat().st_size
        advice = "Tracking results belong to one exact file: choose the video this session was made with."
        if size != self.size:
            raise WrongVideoError(f"{path} is not the video of this session: it has {size:,} bytes, and the session "
                                  f"was made with a video of {self.size:,} bytes. {advice}")
        if sha256_first_64mib(path) != self.sha256_first_64mib:
            raise WrongVideoError(f"{path} is not the video of this session: it has the right size but different "
                                  f"content (was it converted again?). {advice}")


@dataclass
class Session(_Record):
    """A whole session.json (docs/design/formats.md). tool_version: the outline-tracker version that made the
    session. student: key "label". complete: False while a run is partial (SPEC 6.4). boundary: None when no
    boundary was fitted (decision 8). The file's "format" and "schema_version" are constants written by
    `to_json`. read_version: the schema version of the file the session was read from (this build's for a new
    session); it is no key of the file."""

    read_version: ClassVar[int] = SESSION_SCHEMA_VERSION
    tool_version: str = __version__
    student: str = field(default="", metadata={"key": "label"})
    notes: str = ""
    complete: bool = True
    video: VideoRef = field(default_factory=VideoRef)
    clip: Clip = field(default_factory=Clip)
    time: TimeSettings = field(default_factory=TimeSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    axes: Axes = field(default_factory=Axes)
    boundary: Boundary | None = None
    processing: Processing = field(default_factory=Processing)
    probes: list[ProbeBox] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    runs: list[RunRecord] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """The whole file as a JSON object, keys in the order of SPEC 8.10."""
        body = {key: value for key, value in super().to_json().items() if key not in ("format", "schema_version")}
        return {"format": SESSION_FORMAT, "schema_version": SESSION_SCHEMA_VERSION, **body}

    @classmethod
    def from_json(cls, data, where: str = "The session file") -> Session:
        """A session from the parsed JSON of a session file of version 1 or 2 (a file of version 1 is read
        through `session_v1.upgrade`); `where` names the file in messages. Raises SessionVersionError for
        another schema version, SessionKeyError (MixedVersionError for a file of version 1 that holds a key of
        version 2) for a key whose value is refused, and ValueError when the data are not a session, a block
        is of the wrong kind, or fps_true is not a positive number."""
        if not isinstance(data, dict) or data.get("format") != SESSION_FORMAT:
            raise ValueError(f'{where} is not an outline-tracker session file (it has no "format": '
                             f'"{SESSION_FORMAT}").')
        found = data.get("schema_version")
        if isinstance(found, bool) or found not in READ_VERSIONS:
            raise SessionVersionError(found, where, data.get("tool_version"))
        if found == 1:
            data = upgrade(data, where)
        session = super().from_json({k: v for k, v in data.items() if k not in ("format", "schema_version")})
        session.read_version = found
        if session.boundary is not None and session.boundary.shape != "circle":
            raise SessionKeyError("boundary.shape", f'{where} has "boundary.shape" {session.boundary.shape!r}, but '
                                  'this outline-tracker knows only the shape "circle". It may have been written by '
                                  "a newer outline-tracker: update yours.")
        session.validate()
        return session

    @classmethod
    def load(cls, path) -> Session:
        """Read a session.json of version 1 or 2. Raises FileNotFoundError, SessionVersionError (the file has
        another schema version; the message names it and the versions read), SessionKeyError (a key whose value
        is refused; the message names the file and the key), or ValueError (not a session file)."""
        path = Path(path)
        try:
            data = json.loads(path.read_bytes())  # bytes: UTF-8 with or without a byte order mark
        except ValueError as err:
            raise ValueError(f"{path} is not a session file: it could not be read as JSON ({err}).") from None
        return cls.from_json(data, where=str(path))

    def validate(self) -> None:
        """Refuse what must never reach a session file: an fps_true that is not a positive, finite number of
        frames per second (None, not known yet, is allowed). Raises ValueError."""
        fps = self.time.fps_true
        if fps is None:
            return
        if not isinstance(fps, numbers.Real) or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps_true must be a positive number of frames per second, not {fps!r}. Take it from "
                             "the manifest or the stopwatch clip, or type it in.")

    def save(self, path, on_kept: Callable[[str], None] | None = None) -> Path:
        """Write the session to `path` (a session.json; its folder is created if needed): UTF-8, LF line
        ends, one key per line, atomically (`fileio.atomic_write`). Returns the path written: `path`, or
        `<stem>.new<suffix>` next to it when `path` stayed locked by another program. Raises ValueError,
        before anything is written, when fps_true is not positive or a number is NaN or infinite.

        A session file of an older schema version at `path` is kept first, byte for byte, as session.v1.json
        beside it (`fileio.keep_beside` says what happens when that name is taken; decision 36). Once `path`
        holds the new session, `on_kept` is called with a line for the user (`KEPT_OLDER`) that names the kept
        file. Raises OlderFileNotKept (an OSError) when that copy cannot be written, and OSError when `path` is
        there but cannot be read: `path` is then left as it was. A `path` that may not be read (PermissionError;
        on Windows, another program holds it open) is left as it was too, and the session goes to
        `<stem>.new<suffix>` beside it, as for a locked file."""
        self.validate()
        try:
            text = json.dumps(self.to_json(), indent=2, ensure_ascii=False, allow_nan=False)
        except ValueError as err:
            raise ValueError("The session cannot be saved: a number in it is not finite (NaN or infinity), or it "
                             f"refers to itself ({err}).") from None
        data = (text + "\n").encode("utf-8")
        path = target = Path(path)
        kept = None
        try:
            older = _older_file(path)
        except PermissionError:  # what it holds cannot be told: it is neither kept nor replaced
            older, target = None, new_name(path)
        if older is not None:
            raw, version, mtime_ns = older
            try:
                kept = keep_beside(path, f"v{version}", raw, mtime_ns)
            except OSError as error:
                raise OlderFileNotKept(path.name, version, error) from error
        written = atomic_write(target, lambda tmp: tmp.write_bytes(data))
        if kept is not None and written == path and on_kept is not None:
            on_kept(KEPT_OLDER.format(name=path.name, new=SESSION_SCHEMA_VERSION, old=version, kept=kept.name))
        return written

    def world_frame(self) -> WorldFrame:
        """The image <-> world transform of this session (SPEC 3.2): the scale in mm per px from the
        calibration stick when there is one, else from `calibration.tracker_fit`; the origin (image px) and
        the angle of +x (stored in degrees, counterclockwise on screen) from `axes`. Raises ValueError when
        there is no scale yet, or the stick, the fit or the axes are unusable (two identical ends, a length
        that is not positive, missing values)."""
        cal = self.calibration
        try:
            if cal.stick is not None:
                k = stick_scale(cal.stick["p1_px"], cal.stick["p2_px"], cal.stick["length_mm"],
                                cal.click_sigma_px).k_mm_per_px
            elif cal.tracker_fit is not None:
                k = cal.tracker_fit["mm_per_px"]
            else:
                raise ValueError("This session has no scale yet: place the calibration stick first.")
        except (KeyError, TypeError):
            raise ValueError('The calibration is incomplete: a stick needs "p1_px", "p2_px" (points in px) and '
                             '"length_mm"; a scale fitted to a Tracker export needs "mm_per_px".') from None
        try:
            u0, v0 = self.axes.origin_px
            return WorldFrame(k_mm_per_px=k, alpha_rad=math.radians(self.axes.angle_deg), u0=u0, v0=v0)
        except TypeError:
            raise ValueError("The axes are incomplete: the origin must be a point [u, v] in px and the angle a "
                             "number of degrees.") from None

    def locate_video(self, run_folder) -> Path:
        """Find this session's video (SPEC 8.1): first at `video.relpath` from `run_folder` (the folder that
        holds session.json), then at `video.abspath`. A file counts only if it has the stored size and hash.
        Returns its absolute path. Raises WrongVideoError when a file is there but is another video, and
        VideoNotFoundError when there is no file at either place: then ask the user, `video.check` the
        answer and store it with `video.point_to`."""
        video = self.video
        places = [os.path.join(run_folder, video.relpath)] if video.relpath else []
        if os.path.isabs(video.abspath):  # a path from another kind of computer is not one here
            places.append(video.abspath)
        places = list(dict.fromkeys(Path(os.path.abspath(place)) for place in places))
        refusal = None
        for place in places:
            try:
                video.check(place)
                return place
            except VideoNotFoundError:
                continue
            except WrongVideoError as err:
                refusal = refusal or err
        if refusal is not None:
            raise refusal
        stored = video.relpath or video.abspath
        if not stored:
            raise VideoNotFoundError("This session does not name a video yet.")
        name = re.split(r"[\\/]", stored)[-1]
        tried = "; ".join(str(place) for place in places) or video.abspath
        raise VideoNotFoundError(f"The video {name} was not found. It is not at: {tried}. Choose where it is now.")


def _older_file(path: Path) -> tuple[bytes, int, int] | None:
    """The bytes, the schema version and the time in ns of the session file at `path`, when it is of an older
    schema version than this build writes; None when there is no file there, or it is no session file of an
    older version. Raises OSError when the file is there and cannot be read (PermissionError when it may not
    be): what cannot be kept is not replaced."""
    try:
        mtime_ns = path.stat().st_mtime_ns
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        data = json.loads(raw)  # bytes: UTF-8 with or without a byte order mark, as `load` reads them
    except ValueError:  # no JSON, or no UTF-8 text: no session file of any version
        return None
    version = data.get("schema_version") if isinstance(data, dict) and data.get("format") == SESSION_FORMAT else None
    if not isinstance(version, int) or isinstance(version, bool) or version >= SESSION_SCHEMA_VERSION:
        return None
    return raw, version, mtime_ns
