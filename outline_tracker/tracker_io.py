"""Tracker's files and calibration, ported from the course's shrimp.segment (unchanged behavior).

The functions and classes below are last week's, moved over unchanged under their own names and
checked against the reference copy by tests/test_port_fidelity.py and
tests/test_port_equivalence.py. A "fix" here would change what students' files from last week mean.

What the module does:
- `read_tracker_export` reads a file exported from Tracker (File > Export > Data) with one point
  mass or several ("#multi:"): one table per point mass with the columns t (s), frame, x and y (mm,
  the user's axes, y up in Tracker), pixelx and pixely (px).
- `fit_calibration` fits Tracker's map between those two coordinate systems; `Calibration` holds it.
- `make_plan` decides, from an export, which frames to track and where each object starts.
- `write_tracker_file` writes one track in the same layout as an export.

Coordinates: pixelx and pixely are Tracker's image coordinates in px: the origin is the top-left
corner of the frame, pixelx grows to the right, pixely downward, and the pixel in column c and row r
has its center at (c + 0.5, r + 0.5). x and y are in mm. `frame` is the video's own frame number
and fps_true is in frames per second (see SPEC 3). geometry.py imports this module, never the reverse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- reading Tracker exports

def _cells(line: str) -> tuple[list[str], str]:
    delim = "\t" if "\t" in line else (";" if ";" in line else ",")
    return [c.strip() for c in line.split(delim)], delim


def read_tracker_export(path) -> dict[str, pd.DataFrame]:
    """Read a file exported from Tracker with one or several point masses.

    Returns {name: table} with the columns t, frame, x, y, pixelx, pixely (float, NaN where empty),
    one row per line of the file. Several point masses exported together start with "#multi:",
    then a line of names, then the column names repeated for each point mass. With one point mass the
    name is the file name without extension (that is how the tracks are matched across tools).
    """
    path = Path(path)
    lines = [ln.rstrip("\r\n") for ln in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()]
    lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    header_i = next((i for i, ln in enumerate(lines[:6]) if {"x", "y", "pixelx", "pixely"} <= set(_cells(ln)[0])), None)
    if header_i is None:
        raise ValueError(f"{path.name}: no line with the columns x, y, pixelx, pixely. In Tracker's Export Data "
                         "dialog, choose the columns frame, x, y, pixelx and pixely.")
    header, _ = _cells(lines[header_i])
    while header and header[-1] == "":
        header.pop()
    names = [c for c in _cells(lines[header_i - 1])[0] if c] if header_i > 0 else []
    if len(names) <= 1:
        names = [path.stem]
    has_t = header[0] == "t"  # Tracker writes the time column once, first
    first = 1 if has_t else 0
    per = len(header) - first
    if per % len(names):
        raise ValueError(f"{path.name}: {per} columns cannot be split between the point masses {names}.")
    per //= len(names)
    rows = [_cells(ln)[0] for ln in lines[header_i + 1:]]
    out = {}
    for k, name in enumerate(names):
        cols = header[first + k * per: first + (k + 1) * per]
        missing = [c for c in ("frame", "x", "y", "pixelx", "pixely") if c not in cols]
        if missing:
            raise ValueError(f"{path.name}: point mass {name!r} has no column {', '.join(missing)}.")
        data = {"t": [(r[0] if has_t and r else "") for r in rows]}
        for c in ("frame", "x", "y", "pixelx", "pixely"):
            j = first + k * per + cols.index(c)
            data[c] = [r[j] if j < len(r) else "" for r in rows]
        df = pd.DataFrame(data).replace("", np.nan).apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["frame", "pixelx", "pixely"]).sort_values("frame").reset_index(drop=True)
        df["frame"] = np.round(df["frame"]).astype(int)
        out[name] = df
    return out


# --------------------------------------------------------------------------- pixels <-> mm

@dataclass
class Calibration:
    """Tracker's map from image pixels (pixelx, pixely) to mm (x, y): a scale, a rotation, a shift and,
    because Tracker's y points up while image rows go down, usually a flip:
        flip:     x = a px + c py + tx,   y = c px - a py + ty
        no flip:  x = a px - c py + tx,   y = c px + a py + ty
    The scale is sqrt(a^2 + c^2) mm per pixel."""

    a: float
    c: float
    tx: float
    ty: float
    flip: bool
    rms_mm: float

    @property
    def mm_per_px(self) -> float:
        return float(np.hypot(self.a, self.c))

    def matrix(self) -> np.ndarray:
        if self.flip:
            return np.array([[self.a, self.c], [self.c, -self.a]])
        return np.array([[self.a, -self.c], [self.c, self.a]])

    def to_mm(self, px, py):
        m = self.matrix()
        px, py = np.asarray(px, float), np.asarray(py, float)
        return m[0, 0] * px + m[0, 1] * py + self.tx, m[1, 0] * px + m[1, 1] * py + self.ty

    def to_px(self, x, y):
        m = np.linalg.inv(self.matrix())
        x, y = np.asarray(x, float) - self.tx, np.asarray(y, float) - self.ty
        return m[0, 0] * x + m[0, 1] * y, m[1, 0] * x + m[1, 1] * y


def fit_calibration(px, py, x, y) -> Calibration:
    """Least-squares fit of Tracker's pixel -> mm map from points where both are known.

    Tracker's map is a similarity (scale, rotation, shift) with y flipped (its y points up, image rows
    go down), so two different points determine it and the residual of many points is zero up to
    rounding. The unflipped form is kept only if it fits clearly better: with two points, or points on
    one line, both forms fit exactly, and the flip must not be left to rounding.
    """
    px, py, x, y = (np.asarray(v, float) for v in (px, py, x, y))
    ok = np.isfinite(px) & np.isfinite(py) & np.isfinite(x) & np.isfinite(y)
    px, py, x, y = px[ok], py[ok], x[ok], y[ok]
    if len(px) < 2 or np.hypot(np.ptp(px), np.ptp(py)) < 1.0:
        raise ValueError("Need at least two points at least 1 px apart with both mm and pixel coordinates "
                         "to get the calibration. Export a track that moves, or several point masses.")
    fits = {}
    n = len(px)
    one, zero = np.ones(n), np.zeros(n)
    for flip in (True, False):
        if flip:  # x = a px + c py + tx ;  y = c px - a py + ty
            A = np.block([[px[:, None], py[:, None], one[:, None], zero[:, None]],
                          [-py[:, None], px[:, None], zero[:, None], one[:, None]]])
        else:     # x = a px - c py + tx ;  y = c px + a py + ty
            A = np.block([[px[:, None], -py[:, None], one[:, None], zero[:, None]],
                          [py[:, None], px[:, None], zero[:, None], one[:, None]]])
        b = np.concatenate([x, y])
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        rms = float(np.sqrt(np.mean((A @ sol - b) ** 2)))
        fits[flip] = Calibration(*map(float, sol), flip=flip, rms_mm=rms)
    if fits[True].rms_mm > 1e-6 and fits[False].rms_mm < 0.5 * fits[True].rms_mm:
        return fits[False]  # a mirrored image: only when the data say so
    return fits[True]


# --------------------------------------------------------------------------- writing a Tracker file

def write_tracker_file(path, name: str, frames, t, x, y, px, py) -> None:
    """Write one track like a Tracker export: a line with the name, the column names, then one row per
    frame. Frames where the shrimp was lost keep their row with x and y empty, as Tracker does."""

    def f(v, digits):
        return "" if not np.isfinite(v) else f"{v:.{digits}f}"

    lines = [f",{name},,,,,", "t,frame,x,y,pixelx,pixely"]
    for fr, tt, xx, yy, pp, qq in zip(frames, t, x, y, px, py):
        lines.append(f"{tt:.7f},{int(fr)},{f(xx, 6)},{f(yy, 6)},{f(pp, 3)},{f(qq, 3)}")
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- settings

def _fps_from_export(tracks: dict[str, pd.DataFrame]) -> float | None:
    for df in sorted(tracks.values(), key=len, reverse=True):
        d = df.dropna(subset=["t"])
        if len(d) >= 2:
            dframe, dt = np.diff(d["frame"].to_numpy(float)), np.diff(d["t"].to_numpy(float))
            ok = dt > 0
            if ok.any():
                return float(np.median(dframe[ok] / dt[ok]))
    return None


def _fps_from_manifest(video: Path, manifest: Path) -> float | None:
    if not manifest.exists():
        return None
    try:
        m = pd.read_csv(manifest, dtype=str)
    except Exception:
        return None
    if "video_file" not in m.columns or "fps_true" not in m.columns:
        return None
    stem = re.sub(r"_tracker$", "", video.stem)
    rows = m[m["video_file"].fillna("").map(lambda s: Path(s).stem == stem)]
    if len(rows) != 1:
        return None
    try:
        return float(rows["fps_true"].iloc[0])
    except (TypeError, ValueError):
        return None


@dataclass
class Plan:
    names: list[str]
    points_px: list[tuple[float, float]]
    start: int
    step: int
    n: int
    fps: float
    calibration: Calibration

    @property
    def frames(self) -> list[int]:
        return [self.start + i * self.step for i in range(self.n)]


def make_plan(export, seconds=None, step=None, fps=None, video=None, manifest="data/manifest.csv") -> Plan:
    """Decide which frames to track and where each shrimp starts, from the Tracker export."""
    tracks = read_tracker_export(export)
    names = list(tracks)
    allrows = pd.concat(tracks.values(), ignore_index=True)
    cal = fit_calibration(allrows["pixelx"], allrows["pixely"], allrows["x"], allrows["y"])
    firsts = {n: int(df["frame"].iloc[0]) for n, df in tracks.items() if len(df)}
    if len(firsts) < len(names):
        raise ValueError(f"No marked frame for {sorted(set(names) - set(firsts))} in {Path(export).name}.")
    start = min(firsts.values())
    late = [n for n, f in firsts.items() if f != start]
    if late:
        raise ValueError(f"Every shrimp must be marked on the same first frame ({start}); "
                         f"{', '.join(late)} start later. Mark them all on frame {start} in Tracker.")
    points = [(float(tracks[n]["pixelx"].iloc[0]), float(tracks[n]["pixely"].iloc[0])) for n in names]
    longest = max(tracks.values(), key=len)
    if step is None:
        step = int(round(np.median(np.diff(longest["frame"])))) if len(longest) >= 2 else 2
    step = max(int(step), 1)
    if fps is None and video is not None:
        fps = _fps_from_manifest(Path(video), Path(manifest))  # the stopwatch calibration comes first
    if fps is None:
        fps = _fps_from_export(tracks)  # Tracker's t: right if Clip Settings had frame rate = fps_true
    if fps is None:
        raise ValueError("Unknown fps_true: give it with --fps (e.g. --fps 239.6), or fill in data/manifest.csv.")
    if seconds is None and len(longest) >= 2:
        n = (int(longest["frame"].iloc[-1]) - start) // step + 1
    else:
        n = int(round((10.0 if seconds is None else seconds) * fps / step))
    return Plan(names, points, start, step, max(n, 1), float(fps), cal)
