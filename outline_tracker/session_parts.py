"""The records of session.json below the whole session (SPEC 8.10, schema version 1), and how a record is read
from and written to JSON.

One dataclass per block of session.json, with the field names of SPEC 8.10: `Clip`, `TimeSettings`,
`CalibrationSettings`, `Axes`, `Circle`, `Processing`, `ProbeBox`, `Prompt`, `Track`, `RunRecord` and
`Correction`, and `_Record`, the base class that all of them share with `Session` and that does the reading
and writing. The video block (`VideoRef`, with the errors of finding the video again), the whole `Session`
(load, save, `world_frame`, `locate_video`) and `SessionVersionError` are in outline_tracker/session.py,
which imports from this module and gives every name of it too, so that `from outline_tracker.session import
Track` works.

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

No Qt, no torch.
"""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Any, Union, get_args, get_origin, get_type_hints

import numpy as np


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
