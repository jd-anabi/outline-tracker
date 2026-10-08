"""The single source of truth for every output file of SPEC 8: columns, flags, file names, formats.

Everything that writes or describes an output file reads it from here: the CSV column tables (SPEC 8.2,
8.4, 8.5, 8.7), the Tracker-format columns of 8.3, the quality flags of 9, the file list of 8.1, the key
lists of results.npz (8.12) and outlines.npz (8.6) and the versions. The texts of README.txt (8.11) and
docs/OUTPUTS.md are built from these tables in schema_docs.py (`readme_text` is importable from here).

Units and coordinates: the unit of a column is in its name where it has one (_s seconds, _mm millimetres,
_mm2 square millimetres, _rad radians, _px image pixels; ratios have none, probe means use the 0-255
scale). Image coordinates (u, v) follow Tracker: origin at the top-left corner of the frame, u to the
right, v down, the center of the pixel in column c and row r at (c + 0.5, r + 0.5). World coordinates
(x, y) are in mm in the user's axes, y pointing up (SPEC 3). This module does no geometry; it only names
and formats numbers.

Text rules of the new CSV files (X11): UTF-8, LF line ends on every platform, no quoting unless a
cell holds a comma, a quote or a line break, a missing number is an empty cell, integers are written
as integers. The Tracker-format files keep last week's writer and are not written here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# --------------------------------------------------------------------------- versions and text rules

RESULTS_VERSION = 1  # results.npz format version (SPEC 8.12), stored under RESULTS_VERSION_KEY
RESULTS_VERSION_KEY = "version"
SESSION_SCHEMA_VERSION = 1  # session.json "schema_version" (SPEC 8.10)

CSV_ENCODING = "utf-8"
CSV_NEWLINE = "\n"  # on every platform: write the encoded text as bytes, or open with newline="\n"
FLAG_SEPARATOR = ";"

OUTLINE_POINTS = 128  # points per exported outline (SPEC 7.4)
STORED_OUTLINE_POINTS = 256  # points per outline in results.npz
RADIAL_STEP_DEG = 5  # SPEC 7.6
RADIAL_ANGLES = 360 // RADIAL_STEP_DEG  # 72 columns r_000 ... r_355

SESSION_JSON = "session.json"
POSITIONS_CSV = "positions.csv"
SHAPES_CSV = "shapes.csv"
RADIAL_CSV = "radial.csv"
OUTLINES_NPZ = "outlines.npz"
PROBES_CSV = "probes.csv"
OVERLAY_MP4 = "overlay.mp4"
RESULTS_NPZ = "results.npz"
RUN_LOG = "run.log"
README_TXT = "README.txt"
TRACKER_FILE = "<model>/<id>.csv"  # Tracker format, in a folder named after the model (SPEC 8.3)


# --------------------------------------------------------------------------- CSV columns

@dataclass(frozen=True)
class Column:
    """One CSV column. `dtype` is "str", "int" or "float"; `unit` is "" for none; `fmt` is the
    format spec of the written number ("d", ".6f"; "" for text); `meaning` is ASCII prose."""

    name: str
    dtype: str
    unit: str
    fmt: str
    meaning: str


def _str(name: str, meaning: str) -> Column:
    return Column(name, "str", "", "", meaning)


def _int(name: str, meaning: str) -> Column:
    return Column(name, "int", "", "d", meaning)


def _float(name: str, unit: str, decimals: int, meaning: str) -> Column:
    return Column(name, "float", unit, f".{decimals}f", meaning)


_TRACK_ID = _str("track_id", "track name: A, B, ...; later pieces of the same animal are A2, A3, ...")
_FRAME = _int("frame", "video frame number, counted from 0 as Tracker counts")
_T_S = _float("t_s", "s", 7, "time, frame / fps_true (frame 0 is 0 s)")

POSITIONS: list[Column] = [
    _TRACK_ID, _FRAME, _T_S,
    _float("x_mm", "mm", 6, "area centroid of the full mask in your axes; empty if the track is not visible"),
    _float("y_mm", "mm", 6, "the same point, y pointing up"),
    _float("u_px", "px", 3, "the same point in image coordinates (Tracker's pixelx)"),
    _float("v_px", "px", 3, "the same point in image coordinates (Tracker's pixely)"),
    _float("area_mm2", "mm^2", 6, "mask area"),
    _int("visible", "1 if the mask is non-empty, else 0"),
    _str("mode", "coarse (the model saw the whole frame or dish) or fine (a crop that follows the object)"),
    _str("flags", "quality codes, separated by ';' (see the quality flags); empty if none"),
]

SHAPES: list[Column] = [
    _TRACK_ID, _FRAME, _T_S,
    _float("area_mm2", "mm^2", 6, "mask area"),
    _float("perimeter_mm", "mm", 6, "length of the outline polygon"),
    _float("major_mm", "mm", 6, "major axis L1 = 4 sqrt(lambda1) of the full mask"),
    _float("minor_mm", "mm", 6, "minor axis L2 = 4 sqrt(lambda2) of the full mask"),
    _float("eccentricity", "", 6, "sqrt(1 - lambda2 / lambda1) of the full mask"),
    _float("theta_rad", "rad", 6, "heading: direction from the body center to the head, counterclockwise from +x "
           "in your axes, unwrapped so that it is continuous (it may leave -pi..pi)"),
    _float("core_x_mm", "mm", 6, "x of the core's centroid: the body without thin appendages, the 'body center'"),
    _float("core_y_mm", "mm", 6, "y of the core's centroid"),
    _float("core_frac", "", 6, "area of the core / area of the full mask (the core is the body without thin parts such "
           "as antennae, found by a morphological opening); if the core fallback was used (the opening removed more "
           "than half the area, so the full mask serves as the core) it is still the fraction the opening kept: "
           "below 0.5, not 1"),
    _float("solidity", "", 6, "outline polygon area / convex hull area"),
    _float("circularity", "", 6, "4 pi polygon area / perimeter^2 (1 for a circle)"),
    _float("feret_max_mm", "mm", 6, "largest distance between two points of the outline"),
    _int("n_components", "number of connected pieces of the mask (0 if the track is lost)"),
    _float("largest_fraction", "", 6, "area of the largest piece / area of the mask"),
    _float("px_along_major", "px", 3, "camera pixels along the major axis of the largest piece of the mask (L1 / k)"),
    _float("cells_along_major", "cells", 3, "model grid cells along the major axis, (L1 / k) / cell, "
           "with cell = max(width, height) / 256 px of the image the model saw"),
    _int("shape_ok", "1 if min(px_along_major, cells_along_major) >= 20 (setting shape_ok_min), else 0"),
    _float("wall_dist_centroid_mm", "mm", 6, "dish radius minus the centroid's distance from the dish center; "
           "empty without a dish circle; negative = outside the circle"),
    _float("wall_dist_min_mm", "mm", 6, "dish radius minus the outline's largest distance from the dish center "
           "(how close the outline comes to the wall); empty without a dish circle"),
    _str("flags", "quality codes, separated by ';' (see the quality flags); the same text as in positions.csv"),
]

_RADIAL_MEANING = (
    "radius at the angle ddd degrees of the column name, counterclockwise from the head direction: the distance "
    "from the body center to the farthest crossing of that ray with the outline; r_000 is the head radius; "
    "empty when the track is not visible"
)
RADIAL: list[Column] = [_TRACK_ID, _FRAME, _T_S] + [
    _float(f"r_{deg:03d}", "mm", 6, _RADIAL_MEANING) for deg in range(0, 360, RADIAL_STEP_DEG)
]

PROBES: list[Column] = [
    _FRAME, _T_S,
    _str("probe", "name of the probe rectangle (default LED1)"),
    _float("r", "0-255", 3, "mean red inside the rectangle, on the 0-255 scale"),
    _float("g", "0-255", 3, "mean green inside the rectangle, on the 0-255 scale"),
    _float("b", "0-255", 3, "mean blue inside the rectangle, on the 0-255 scale"),
    _float("gray", "0-255", 3, "0.299 r + 0.587 g + 0.114 b of the three means"),
]

# <model>/<id>.csv is written by the ported write_tracker_file (SPEC 8.3); its columns are listed here for the docs.
TRACKER_COLUMNS: list[Column] = [
    _float("t", "s", 7, "time, frame / fps_true"),
    _int("frame", "video frame number"),
    _float("x", "mm", 6, "x in your axes; empty where the track is lost"),
    _float("y", "mm", 6, "y in your axes, y pointing up; empty where the track is lost"),
    _float("pixelx", "px", 3, "image coordinate u; empty where the track is lost"),
    _float("pixely", "px", 3, "image coordinate v; empty where the track is lost"),
]


# --------------------------------------------------------------------------- quality flags (SPEC 9)

@dataclass(frozen=True)
class FlagInfo:
    """A quality flag. `scope` is "position" if it concerns where the object is, "shape" if it
    only concerns the outline and heading."""

    code: str
    scope: str
    condition: str
    meaning: str


FLAG_INFO: dict[str, FlagInfo] = {
    f.code: f
    for f in [
        FlagInfo("LOST", "position", "mask empty", "no position; re-click if needed"),
        FlagInfo("JUMP", "position",
                 "centroid speed between consecutive visible frames above jump_mm_s (default 100 mm/s)",
                 "probably jumped to another object"),
        FlagInfo("SIZE", "position", "area outside 0.5 to 2 times the track's median area",
                 "merged with a neighbor, or partly lost"),
        FlagInfo("CONTACT", "position",
                 "this track's outline comes within max(3 px, 2 grid cells) of another track's outline "
                 "on the same frame", "identities may swap here: check"),
        FlagInfo("EDGE", "position",
                 "the mask touches the border of the model's input (frame, dish crop or fine crop)",
                 "outline may be cut off"),
        FlagInfo("MULTI", "shape", "more than one component, the second at least 10% of the largest",
                 "ragged or split mask"),
        FlagInfo("LOWRES", "shape", "shape_ok = 0 (fewer than 20 camera pixels or fewer than 20 grid cells along the major axis)",
                 "shape columns unreliable; use fine mode or closer footage"),
        FlagInfo("ORIENT", "shape",
                 "core nearly round (lambda2 / lambda1 > 0.8), heading axis jumped by more than 60 degrees, "
                 "or the core fallback was used", "heading unreliable"),
        FlagInfo("HEADGUESS", "shape", "head side taken from motion (no head click); on every frame of the track",
                 "check the head in the overlay"),
    ]
}
FLAGS: list[str] = list(FLAG_INFO)  # SPEC 9 order, also the order inside the flags column
POSITION_FLAGS: list[str] = [c for c in FLAGS if FLAG_INFO[c].scope == "position"]
SHAPE_FLAGS: list[str] = [c for c in FLAGS if FLAG_INFO[c].scope == "shape"]


def join_flags(codes: Iterable[str]) -> str:
    """The text of a `flags` cell: the given codes without repeats, in SPEC 9 order, joined by ';'.

    No units or coordinates. An unknown code raises ValueError.
    """
    wanted = set(codes)
    unknown = sorted(wanted - FLAG_INFO.keys())
    if unknown:
        raise ValueError(f"unknown flag code(s): {', '.join(unknown)}")
    return FLAG_SEPARATOR.join(c for c in FLAGS if c in wanted)


# --------------------------------------------------------------------------- files (SPEC 8.1, 8.13)

@dataclass(frozen=True)
class FileSpec:
    """A file of the run folder. `in_git` follows SPEC 8.13; `columns` is set for the CSV files."""

    name: str
    meaning: str
    in_git: bool
    optional: bool = False
    columns: list[Column] | None = None


FILES: list[FileSpec] = [
    FileSpec(SESSION_JSON, "everything needed to reopen the run, run it again and export again: settings, "
             "calibration, clicks, run history", True),
    FileSpec(POSITIONS_CSV, "one row per track and tracked frame: where the object is", True, columns=POSITIONS),
    FileSpec(TRACKER_FILE, "one file per track in the layout of last week's Tracker files, in a folder named "
             "after the model, e.g. edgetam/A.csv; the folder holds nothing else", True, columns=TRACKER_COLUMNS),
    FileSpec(SHAPES_CSV, "one row per track and tracked frame: size, axes, heading, solidity, resolution, wall "
             "distance", True, columns=SHAPES),
    FileSpec(RADIAL_CSV, "the outline's radius at 72 angles around the body center, for fine tracks", True,
             columns=RADIAL),
    FileSpec(OUTLINES_NPZ, "the outlines themselves (128 points per frame), in your axes and in the body frame, "
             "for fine tracks", True),
    FileSpec(PROBES_CSV, "brightness of the probe rectangles (the LED) on every frame of the clip; written only "
             "if probes exist", True, optional=True, columns=PROBES),
    FileSpec(OVERLAY_MP4, "the clip with each outline, centroid and id drawn on it; plays in PowerPoint and "
             "Google Slides", False),
    FileSpec(RESULTS_NPZ, "internal pixel-space store: every other file is rebuilt from it and session.json; "
             "not meant for analysis", False),
    FileSpec(RUN_LOG, "what was run: software, video, time and calibration, runs, corrections, quality summary, "
             "files written", True),
    FileSpec(README_TXT, "this file", True),
]


# --------------------------------------------------------------------------- npz keys (SPEC 8.6, 8.12)

@dataclass(frozen=True)
class ArrayKey:
    """One array of an npz file. `shape` holds ints and names ("n" = tracked frames of the track)."""

    name: str
    dtype: str  # numpy dtype string
    shape: tuple[int | str, ...]
    unit: str
    meaning: str


_N = ("n",)

# Per track and tracked frame, stored under the key "<track id>__<name>" (see npz_key). Pixel-space
# values use image coordinates (u, v) in px. All rows of one track share the same n, in frame order.
RESULTS_KEYS: list[ArrayKey] = [
    ArrayKey("frames", "int32", _N, "", "video frame numbers of the records, ascending, on the clip's frame grid"),
    ArrayKey("visible", "bool", _N, "", "the mask is non-empty"),
    ArrayKey("area_px", "int32", _N, "px", "number of pixels of the mask; 0 if not visible"),
    ArrayKey("u", "float64", _N, "px", "area centroid of the mask, image u; NaN if not visible"),
    ArrayKey("v", "float64", _N, "px", "area centroid of the mask, image v; NaN if not visible"),
    ArrayKey("cov_full", "float64", ("n", 3), "px^2", "covariance of the pixel centers of the full mask: "
             "mu_uu, mu_uv, mu_vv"),
    ArrayKey("cov_core", "float64", ("n", 3), "px^2", "the same for the core mask"),
    ArrayKey("core_u", "float64", _N, "px", "core centroid, image u"),
    ArrayKey("core_v", "float64", _N, "px", "core centroid, image v"),
    ArrayKey("core_frac", "float64", _N, "", "area of the core / area of the mask; with the core fallback it is "
             "the fraction the opening kept, below 0.5, not 1"),
    ArrayKey("core_fallback", "bool", _N, "", "the opening removed more than half the area, so the full mask "
             "was used as the core"),
    ArrayKey("core_r_px", "int32", _N, "px", "radius of the opening disk that made the core; 0 if not visible"),
    ArrayKey("outline_px", "float32", ("n", STORED_OUTLINE_POINTS, 2), "px", "outline resampled to 256 points "
             "in image coordinates (u, v); NaN if not visible"),
    ArrayKey("n_components", "int32", _N, "", "number of connected pieces of the mask"),
    ArrayKey("largest_fraction", "float64", _N, "", "area of the largest piece / area of the mask"),
    ArrayKey("second_fraction", "float64", _N, "", "area of the second-largest piece / area of the largest; "
             "0 with one piece"),
    ArrayKey("cell_px", "float64", _N, "px", "size of one model grid cell in image pixels: max(width, height) / 256 "
             "of the image the model saw"),
    ArrayKey("edge", "bool", _N, "", "the mask touches the border of the model's input"),
    ArrayKey("mode", "U6", _N, "", "coarse or fine"),
    ArrayKey("score", "float32", _N, "", "the model's object-presence logit; NaN if the backend has none"),
    ArrayKey("mask_offset", "int32", ("n", 2), "px", "(column, row) of each mask crop's top-left pixel in the "
             "full frame"),
    ArrayKey("mask_shape", "int32", ("n", 2), "px", "(rows, columns) of each mask crop"),
    ArrayKey("mask_bits", "uint8", ("total_bytes",), "", "all mask crops one after another, each flattened row by "
             "row and packed with numpy.packbits; crop i takes ceil(rows * columns / 8) bytes"),
]

OUTLINES_KEYS: list[ArrayKey] = [
    ArrayKey("frames", "int32", _N, "", "video frame numbers"),
    ArrayKey("xy_mm", "float32", ("n", "N", 2), "mm", "outline points in your axes, counterclockwise, starting at "
             "the head point; NaN for frames where the track is not visible"),
    ArrayKey("xieta_mm", "float32", ("n", "N", 2), "mm", "the same points in the body frame: xi toward the head, "
             "eta 90 degrees counterclockwise from xi"),
]
OUTLINES_META_KEY = "meta"  # a JSON text, see OUTLINES_META_FIELDS
OUTLINES_META_FIELDS: dict[str, str] = {
    "n_points": "N, the number of points of every outline",
    "tracks": "the track ids that have outlines in the file (empty when there is no fine track)",
    "conventions": "a short statement of the coordinate conventions",
    "tool_version": "the Outline Tracker version that wrote the file",
}

_NPZ_SEPARATOR = "__"


def npz_key(track_id: str, name: str) -> str:
    """The key of one per-track array in results.npz or outlines.npz: `<track id>__<name>` (text only)."""
    return f"{track_id}{_NPZ_SEPARATOR}{name}"


def split_npz_key(key: str) -> tuple[str, str] | None:
    """Inverse of `npz_key`: (track id, name), or None for keys without a track ("meta", "version").

    Text only, no units. The split is at the last separator, so a track id may itself contain two
    underscores.
    """
    if _NPZ_SEPARATOR not in key:
        return None
    track_id, name = key.rsplit(_NPZ_SEPARATOR, 1)
    return track_id, name


# --------------------------------------------------------------------------- writing rows

def header_line(columns: Sequence[Column]) -> str:
    """The header line of a CSV file (no line end): the column names, comma separated. No units involved."""
    return ",".join(c.name for c in columns)


def format_row(columns: Sequence[Column], row: Mapping[str, object] | Sequence[object]) -> str:
    """One CSV line (no line end) for `row`, in the units and formats of `columns`.

    `row` is a mapping from column name to value (every column must be present; other keys are
    ignored) or a sequence in column order. Floats are written with the column's decimals; NaN, None
    and infinities become an empty cell. Integer columns must hold a whole number (a lost row keeps
    0): NaN or a fraction raises ValueError. Text is written as is, quoted only if it holds a comma,
    a quote or a line break. Values are in the units named by the columns (s, mm, mm^2, rad, px);
    no coordinate transformation happens here.
    """
    if isinstance(row, Mapping):
        for col in columns:
            if col.name not in row:
                raise KeyError(f"row has no value for column {col.name!r}")
        values = [row[col.name] for col in columns]
    else:
        values = list(row)
        if len(values) != len(columns):
            raise ValueError(f"row has {len(values)} values for {len(columns)} columns")
    return ",".join(_cell(col, value) for col, value in zip(columns, values))


def csv_text(columns: Sequence[Column], rows: Iterable[Mapping[str, object] | Sequence[object]]) -> str:
    """A whole CSV file as text: the header line, then one line per row, each ended by "\\n".

    Values are in the units of the columns (see `format_row`). Encode it with CSV_ENCODING and write
    the bytes (or open with newline="\\n"), so that the file has LF line ends on Windows too. With no
    rows the file is the header line alone.
    """
    lines = [header_line(columns)] + [format_row(columns, row) for row in rows]
    return CSV_NEWLINE.join(lines) + CSV_NEWLINE


def _cell(col: Column, value: object) -> str:
    if col.dtype == "str":
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if not isinstance(value, str):
            raise ValueError(f"column {col.name}: expected text, got {value!r}")
        if any(ch in value for ch in ',"\r\n'):
            return '"' + value.replace('"', '""') + '"'
        return value
    if col.dtype == "int":
        return format(_whole_number(col, value), col.fmt)
    if value is None:
        return ""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"column {col.name}: expected a number, got {value!r}")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as err:
        raise ValueError(f"column {col.name}: expected a number, got {value!r}") from err
    return format(number, col.fmt) if math.isfinite(number) else ""


def _whole_number(col: Column, value: object) -> int:
    if isinstance(value, (bool, int)):
        return int(value)
    problem = f"column {col.name}: an integer column needs a whole number (a lost row keeps 0), got {value!r}"
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(problem)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as err:
        raise ValueError(problem) from err
    if not math.isfinite(number) or number != int(number):
        raise ValueError(problem)
    return int(number)


def __getattr__(name: str) -> object:
    """`readme_text` is built in schema_docs, which imports this module: handed out from here on first use."""
    if name != "readme_text":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from outline_tracker.schema_docs import readme_text

    return readme_text
