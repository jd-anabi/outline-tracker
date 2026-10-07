"""session.json <-> dataclasses (SPEC 8.10, schema version 1), and finding the video again (SPEC 8.1).

One dataclass per block of session.json, with the field names of SPEC 8.10. A `Session` holds everything
needed to reopen a run, run it again and export again.

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

import copy
import json
import math
import numbers
import os
import re
import types
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import numpy as np

from outline_tracker import __version__
from outline_tracker.fileio import atomic_write, relative_path, sha256_first_64mib
from outline_tracker.geometry import WorldFrame, stick_scale
from outline_tracker.schema import SESSION_SCHEMA_VERSION

SESSION_FORMAT = "outline-tracker-session"  # the "format" key of session.json


class SessionVersionError(ValueError):
    """session.json has another schema version than this tool reads, or none. `found` is the file's
    "schema_version" (None when missing), `supported` the version this tool reads."""

    def __init__(self, found, where: str = "The session file", written_by=None):
        self.found, self.supported = found, SESSION_SCHEMA_VERSION
        by = f" ({written_by})" if isinstance(written_by, str) and written_by else ""
        if not isinstance(found, int) or isinstance(found, bool):
            has, advice = f"no usable session format version ({found!r})", "It may be damaged."
        elif found > self.supported:
            has = f"session format version {found}"
            advice = f"It was written by a newer outline-tracker{by}: update yours."
        else:
            has = f"session format version {found}"
            advice = f"It was written by an older outline-tracker{by}: open it with that one, or start a new session."
        super().__init__(f"{where} has {has}, but this outline-tracker ({__version__}) reads only version "
                         f"{self.supported}. {advice}")


class VideoNotFoundError(FileNotFoundError):
    """The session's video is at none of the places the session knows: ask the user where it is."""


class WrongVideoError(ValueError):
    """A file is not the video the session was made with (another size, or other content)."""


# --------------------------------------------------------------------------- records <-> JSON

def _optional():
    """A key added by X18: None by default, and then left out of the file."""
    return field(default=None, metadata={"optional": True})


def _plain(value):
    """A value as JSON holds it, always a copy: records become objects, tuples and arrays become lists,
    numpy numbers become Python numbers."""
    if isinstance(value, _Record):
        return value.to_json()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@lru_cache(maxsize=None)
def _layout(cls) -> tuple[tuple[str, str, bool, Any], ...]:
    """(field name, JSON key, optional, type) for every field of a record class, in file order."""
    hints = get_type_hints(cls)
    return tuple((f.name, f.metadata.get("key", f.name), bool(f.metadata.get("optional")), hints[f.name])
                 for f in fields(cls) if f.name != "extra")


def _decode(hint, value, where: str):
    """The value of one key: a record, a list of records, or the JSON value as it is (copied)."""
    nullable = get_origin(hint) in (Union, types.UnionType)
    if nullable:  # every union here is "X | None"
        (hint,) = (arg for arg in get_args(hint) if arg is not type(None))
    listed = get_origin(hint) is list
    item = get_args(hint)[0] if listed else hint
    if not (isinstance(item, type) and get_origin(item) is None and issubclass(item, _Record)):
        return copy.deepcopy(value)
    if value is None and nullable:
        return None
    if not listed:
        return item.from_json(value, where)
    if not isinstance(value, list):
        raise ValueError(f'In the session file, "{where}" must be a list [...], not {value!r}.')
    return [item.from_json(entry, f"{where}[{i}]") for i, entry in enumerate(value)]


@dataclass
class _Record:
    """One JSON object of session.json. `extra` holds the keys this version does not know."""

    extra: dict[str, Any] = field(default_factory=dict, kw_only=True)

    def to_json(self) -> dict[str, Any]:
        """This record as a JSON object of plain dicts, lists, numbers and text that shares nothing with
        the record: the known keys in the order of SPEC 8.10, then the unknown ones."""
        out = {}
        for name, key, optional, _ in _layout(type(self)):
            value = getattr(self, name)
            if not (optional and value is None):
                out[key] = _plain(value)
        for key, value in self.extra.items():
            out.setdefault(key, _plain(value))
        return out

    @classmethod
    def from_json(cls, data, where: str = ""):
        """A record from a parsed JSON object: missing keys take the defaults, unknown keys go to `extra`.
        `where` names the object in messages (its key path, e.g. "tracks[0]"). Raises ValueError when a
        block is not of the kind the file format needs."""
        if not isinstance(data, dict):
            raise ValueError(f'In the session file, "{where or cls.__name__}" must be an object {{...}}, not {data!r}.')
        by_key = {key: (name, hint) for name, key, _, hint in _layout(cls)}
        known, extra = {}, {}
        for key, value in data.items():
            if key in by_key:
                name, hint = by_key[key]
                known[name] = _decode(hint, value, f"{where}.{key}" if where else key)
            else:
                extra[key] = copy.deepcopy(value)
        return cls(**known, extra=extra)


# --------------------------------------------------------------------------- the blocks of SPEC 8.10

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
class Clip(_Record):
    """The tracked part of the video: frames start, start + step, ... up to `end` (inclusive). Video frame
    numbers; step in frames (SPEC 3.4)."""

    start: int = 0
    end: int = 0
    step: int = 2


@dataclass
class TimeSettings(_Record):
    """fps_true and where it came from (SPEC 4.1, X18). fps_true: true frames per second; None until it is
    known. source: "manifest", "stopwatch", "typed" or "tracker-export". manifest_path: the manifest used, or
    None. stopwatch: None or {"frame_a", "time_a_s", "frame_b", "time_b_s"}, two video frames and the
    stopwatch readings in s."""

    fps_true: float | None = None
    source: str = "typed"
    manifest_path: str | None = None
    stopwatch: dict[str, Any] | None = None


@dataclass
class CalibrationSettings(_Record):
    """The scale (SPEC 4.2, 4.3, X18). stick: None or {"p1_px": [u, v], "p2_px": [u, v], "length_mm"}, the two
    clicked ends in image px and the true length in mm. click_sigma_px: assumed precision of one click, px.
    check: None or {"p1_px", "p2_px", "true_mm"}, the tape check. tracker_fit (optional key): None or
    {"mm_per_px", "rms_mm", "n_points"}, the scale fitted to a Tracker export by from-tracker."""

    stick: dict[str, Any] | None = None
    click_sigma_px: float = 0.5
    check: dict[str, Any] | None = None
    tracker_fit: dict[str, Any] | None = _optional()


@dataclass
class Axes(_Record):
    """The user's axes (SPEC 3.2, 4.5). origin_px: [u0, v0] in image px. angle_deg: the direction of +x in
    degrees, counterclockwise on screen from the image's rightward direction."""

    origin_px: list[float] = field(default_factory=lambda: [0.0, 0.0])
    angle_deg: float = 0.0


@dataclass
class Circle(_Record):
    """The dish wall (SPEC 4.4), in image px: the clicked points [[u, v], ...], the fitted center [u, v], the
    radius and the RMS residual. dish_mm: the dish's known inner diameter in mm, or None."""

    points_px: list[list[float]] = field(default_factory=list)
    center_px: list[float] = field(default_factory=lambda: [0.0, 0.0])
    radius_px: float = 0.0
    rms_px: float = 0.0
    dish_mm: float | None = None


@dataclass
class Processing(_Record):
    """Settings of tracking and export, with the defaults of SPEC 8.10. radial_step_deg in degrees;
    shape_ok_min in px or grid cells along the major axis (SPEC 7.8); core_open_frac a fraction of the body
    length (SPEC 7.3); jump_mm_s in mm per s (SPEC 9); fine_window_factor times the object's length (6.3)."""

    model: str = "edgetam"
    device: str = "auto"
    dish_crop: bool = True
    outline_points: int = 128
    radial_step_deg: int = 5
    shape_ok_min: int = 20
    core_open_frac: float = 0.1
    jump_mm_s: float = 100.0
    fine_window_factor: float = 3.0
    shape_files_for_coarse: bool = False


@dataclass
class ProbeBox(_Record):
    """A brightness probe (SPEC 4.6): a name and a rectangle [u0, v0, u1, v1] in image px."""

    name: str = "LED1"
    rect_px: list[float] = field(default_factory=lambda: [0, 0, 0, 0])


@dataclass
class Prompt(_Record):
    """The clicks for one object on one frame (SPEC 5). frame: video frame number. frame_hash: hash of the
    decoded frame the user clicked on, or None (SPEC 3.5). decoder (optional key): the decoder tag that hash
    was made with (X8). points_px: [[u, v], ...] in image px. labels: one per point, 1 positive, 0 negative."""

    frame: int = 0
    frame_hash: str | None = None
    decoder: str | None = _optional()
    points_px: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)


@dataclass
class Track(_Record):
    """One tracked object (SPEC 5, 6). id: "A", "B", ...; pieces "A2", "A3". color: RGB hex such as "#FFFF00".
    mode: "coarse" or "fine". fine_window_px: side of the fine crop in px, None for automatic. start_frame,
    ended_at: video frame numbers (ended_at None while the track runs to the clip's end). head_px: the head
    click [u, v] in image px, or None."""

    id: str = ""
    color: str = ""
    mode: str = "coarse"
    fine_window_px: int | None = None
    start_frame: int = 0
    head_px: list[float] | None = None
    ended_at: int | None = None
    prompts: list[Prompt] = field(default_factory=list)


@dataclass
class RunRecord(_Record):
    """One tracking run (SPEC 6.1). tracks: the ids. start_frame: video frame number. mode: "coarse" or
    "fine". started, finished: ISO 8601 times (finished None while it runs). frames_done: frames tracked.
    seconds_per_frame: s. device: "cpu", "mps" or "cuda". model_id and weights_sha256 (optional keys, X18):
    the model and the SHA-256 of its weights file."""

    tracks: list[str] = field(default_factory=list)
    start_frame: int = 0
    mode: str = "coarse"
    started: str = ""
    finished: str | None = None
    frames_done: int = 0
    seconds_per_frame: float = 0.0
    device: str = ""
    model_id: str | None = _optional()
    weights_sha256: str | None = _optional()


@dataclass
class Correction(_Record):
    """One correction (SPEC 6.6). time: ISO 8601. tracks: the ids. action: "retrack", "end" or "new_piece".
    frame: video frame number. prompts: the clicks given with it."""

    time: str = ""
    tracks: list[str] = field(default_factory=list)
    action: str = ""
    frame: int = 0
    prompts: list[Prompt] = field(default_factory=list)


@dataclass
class Session(_Record):
    """A whole session.json (SPEC 8.10). tool_version: the outline-tracker version that made the session.
    complete: False while a run is partial (SPEC 6.4). circle: None when no dish wall was fitted. The file's
    "format" and "schema_version" are constants written by `to_json`."""

    tool_version: str = __version__
    student: str = ""
    notes: str = ""
    complete: bool = True
    video: VideoRef = field(default_factory=VideoRef)
    clip: Clip = field(default_factory=Clip)
    time: TimeSettings = field(default_factory=TimeSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    axes: Axes = field(default_factory=Axes)
    circle: Circle | None = None
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
        """A session from the parsed JSON of a session file; `where` names the file in messages. Raises
        SessionVersionError for another schema version, and ValueError when the data are not a session, a
        block is of the wrong kind, or fps_true is not a positive number."""
        if not isinstance(data, dict) or data.get("format") != SESSION_FORMAT:
            raise ValueError(f'{where} is not an outline-tracker session file (it has no "format": '
                             f'"{SESSION_FORMAT}").')
        found = data.get("schema_version")
        if isinstance(found, bool) or found != SESSION_SCHEMA_VERSION:
            raise SessionVersionError(found, where, data.get("tool_version"))
        session = super().from_json({k: v for k, v in data.items() if k not in ("format", "schema_version")})
        session.validate()
        return session

    @classmethod
    def load(cls, path) -> Session:
        """Read a session.json. Raises FileNotFoundError, SessionVersionError (the file has another schema
        version; the message names both), or ValueError (not a session file)."""
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

    def save(self, path) -> Path:
        """Write the session to `path` (a session.json; its folder is created if needed): UTF-8, LF line
        ends, one key per line, atomically (`fileio.atomic_write`). Returns the path written: `path`, or
        `<stem>.new<suffix>` next to it when `path` stayed locked by another program. Raises ValueError,
        before anything is written, when fps_true is not positive or a number is NaN or infinite."""
        self.validate()
        try:
            text = json.dumps(self.to_json(), indent=2, ensure_ascii=False, allow_nan=False)
        except ValueError as err:
            raise ValueError("The session cannot be saved: a number in it is not finite (NaN or infinity), or it "
                             f"refers to itself ({err}).") from None
        data = (text + "\n").encode("utf-8")
        return atomic_write(path, lambda tmp: tmp.write_bytes(data))

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
