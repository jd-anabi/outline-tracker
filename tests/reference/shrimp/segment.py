"""Track shrimp with SAM 2 or EdgeTAM. PROVIDED by the course and tested; you do not need to change it.

SAM 2 and EdgeTAM are AI models from Meta. Given the frames of a video and one click on an object,
they return the object's outline (its "mask": which pixels belong to it) in every frame. They know
nothing about mm or seconds and do not output positions. This script does what Tracker does
internally:
    1. reads your video one frame at a time (every `step`-th frame);
    2. gives each frame to the model, which returns each shrimp's outline;
    3. takes the center of each outline (the mean position of its pixels) as the shrimp's position;
    4. converts pixels to mm with YOUR Tracker calibration, and frame numbers to seconds with fps_true;
    5. saves one file per shrimp in the same format as a Tracker export, plus a slowed-down video with
       the outlines drawn on (for your slides).
The outline includes the beating antennae, so its center is not the point Tracker's autotracker
follows. REPORT.md asks you to explain the difference.

What it needs:
    * the video you opened in Tracker (the ..._tracker.mp4 copy), so frame numbers and pixels match;
    * a file exported from Tracker (File > Export > Data) with the columns frame, x, y, pixelx, pixely:
        - one shrimp: its Tracker track. The model starts at the track's first frame and covers the
          same frames, so the three tools can be compared frame by frame;
        - many shrimp: one point mass per shrimp, each marked once (shift-click) on the SAME frame,
          all exported together in one file, saved in extra/ (e.g. extra/start.csv): it is not a track.
      Tracker knows both pixels and mm for every point in the file; that gives the conversion.

Usage, from the repository folder (the first run downloads PyTorch and the model, a few hundred MB).
First check that it works on your laptop and how long it will take (about a minute):
    uv run --with torch --with transformers==5.18.0 --with timm python -m shrimp.segment --selftest --model edgetam
Then:
    uv run --with torch --with transformers==5.18.0 --with timm python -m shrimp.segment VIDEO EXPORT --model edgetam
    uv run --with torch --with transformers==5.18.0 --with timm python -m shrimp.segment VIDEO EXPORT --model sam2

Options:
    --seconds S   how long to track (default: the length of the Tracker track, or 10 s)
    --step K      use every K-th frame (default: the Tracker track's step, or 2)
    --fps F       fps_true (default: data/manifest.csv, else the t column of the Tracker export)
    --out FOLDER  where to save (default: a folder named after the model next to EXPORT, or next to
                  its extra/ folder when EXPORT is in extra/, e.g. .../ana/extra/start.csv -> .../ana/edgetam)
    --device D    cpu, mps (Apple GPU) or cuda (NVIDIA GPU); default: the fastest one that works

It prints how long it will take. On a laptop processor, EdgeTAM needs a few minutes for one shrimp over
10 s at every 2nd frame; SAM 2 about 4 times longer; each extra shrimp adds to the time.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TRANSFORMERS_VERSION = "5.18.0"
RUN_COMMAND = f"uv run --with torch --with transformers=={TRANSFORMERS_VERSION} --with timm python -m shrimp.segment"
MODELS = {
    "sam2": "facebook/sam2.1-hiera-tiny",
    "sam2-small": "facebook/sam2.1-hiera-small",
    "edgetam": "facebook/EdgeTAM",
}
MAX_SPEED_MM_S = 100.0  # a step faster than this is a jump to something else (as in kinematics.find_jumps)
KEEP_FRAMES = 20        # per-frame model results kept in memory (the model looks back at most 16 frames)
COLORS = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 255, 0),
          (128, 0, 255), (0, 0, 255), (255, 0, 0), (128, 255, 128)]


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


def mask_center(mask: np.ndarray) -> tuple[float, float, int]:
    """Center of a boolean mask in Tracker's image coordinates, and its area in pixels.

    The mean column and row of the mask's pixels, plus 0.5: in Tracker, pixel (column c, row r) is the
    square from c to c + 1 and r to r + 1, so its center is at (c + 0.5, r + 0.5). NaN if empty.
    """
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return float("nan"), float("nan"), 0
    return float(cols.mean() + 0.5), float(rows.mean() + 0.5), int(len(rows))


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


# --------------------------------------------------------------------------- video and model

def iter_rgb_frames(path, frames):
    """Yield (frame number, RGB image) for the requested frame numbers (increasing), reading the video
    from the start so the numbering is exactly the decoding order Tracker uses."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}. Use the ..._tracker.mp4 copy made by shrimp.convert.")
    wanted = list(frames)
    i, k = 0, 0
    try:
        while k < len(wanted):
            if i < wanted[k]:
                if not cap.grab():
                    break
            else:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                k += 1
            i += 1
    finally:
        cap.release()


class TransformersSegmenter:
    """SAM 2 / EdgeTAM through Hugging Face transformers, one frame at a time ("streaming")."""

    def __init__(self, model_key: str, device: str = "auto", edgetam_checkpoint=None, model=None, processor=None):
        import torch

        self.torch = torch
        self.model_key = model_key
        if model is None:
            model, processor = load_model(model_key, edgetam_checkpoint)
        self.model, self.processor = model.eval(), processor
        self.device = self._pick_device(device)
        self.model.to(self.device)
        self.session = None

    def _pick_device(self, device):
        torch = self.torch
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _masks(self, out, inputs):
        sizes = [[int(v) for v in size] for size in inputs.original_sizes]
        masks = self.processor.post_process_masks([out.pred_masks.float().cpu()], original_sizes=sizes, binarize=True)[0]
        return [m[0].numpy().astype(bool) for m in masks]

    def _forward(self, rgb, points=None):
        torch = self.torch
        inputs = self.processor(images=rgb, device=self.device, return_tensors="pt")
        if points is not None:
            self.session = self.processor.init_video_session(inference_device=self.device, dtype=torch.float32)
            self.index = 0
            self.processor.add_inputs_to_inference_session(
                inference_session=self.session, frame_idx=0, obj_ids=list(range(1, len(points) + 1)),
                input_points=[[[[float(x), float(y)]] for x, y in points]],
                input_labels=[[[1] for _ in points]], original_size=inputs.original_sizes[0])
        else:
            self.index += 1
        with torch.inference_mode():
            # the frame number is given explicitly: the session would otherwise count its stored frames
            out = self.model(inference_session=self.session, frame_idx=self.index,
                             frame=inputs.pixel_values[0].to(self.device))
        self._prune()
        return self._masks(out, inputs)

    def start(self, rgb, points):
        """First frame: the click (one point per shrimp, in image pixels) starts the tracking."""
        try:
            return self._forward(rgb, points)
        except Exception as err:  # an Apple GPU (mps) can lack an operation: fall back to the processor
            if self.device == "mps":
                print(f"  The Apple GPU failed ({type(err).__name__}); using the processor instead.")
                self.device = "cpu"
                self.model.to("cpu")
                return self._forward(rgb, points)
            raise

    def step(self, rgb):
        return self._forward(rgb)

    def _prune(self):
        """Forget what the model no longer uses: the image of every earlier frame (12.6 MB each at the
        model's 1024 x 1024 input) and the per-frame results older than KEEP_FRAMES frames. Without this
        the memory grows by about 15 MB per frame. Entries are emptied, not removed, so the session's
        frame count stays right."""
        s = self.session
        current = self.index
        old = current - KEEP_FRAMES
        if s.processed_frames:
            for k in s.processed_frames:
                if k < current:
                    s.processed_frames[k] = None
        for obj in s.output_dict_per_obj:
            store = s.output_dict_per_obj[obj]["non_cond_frame_outputs"]
            for k in [k for k in store if k < old]:
                del store[k]
        for obj in s.frames_tracked_per_obj:
            tracked = s.frames_tracked_per_obj[obj]
            for k in [k for k in tracked if k < old]:
                del tracked[k]


def load_model(model_key: str, edgetam_checkpoint=None):
    """The model and its image processor. SAM 2.1 comes from Meta's Hugging Face page; EdgeTAM is
    Meta's original file (facebook/EdgeTAM), converted once and kept in ~/.cache/shrimp-models."""
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    if model_key.startswith("sam2"):
        name = MODELS[model_key]
        return Sam2VideoModel.from_pretrained(name), Sam2VideoProcessor.from_pretrained(name)
    if model_key == "edgetam":
        from shrimp._edgetam import load_edgetam

        return load_edgetam(edgetam_checkpoint)
    raise ValueError(f"Unknown model {model_key!r}: choose sam2, sam2-small or edgetam.")


# --------------------------------------------------------------------------- the run

def _overlay(rgb, masks, names, centers, frame, t):
    h, w = rgb.shape[:2]
    scale = 960 / w
    small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (960, int(round(h * scale / 2)) * 2))
    for k, (m, name, (cx, cy)) in enumerate(zip(masks, names, centers)):
        color = COLORS[k % len(COLORS)]
        if m is not None and m.any():
            ys, xs = np.nonzero(m)
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            crop = m[y0:y1, x0:x1].astype(np.uint8)
            contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                pts = ((cnt[:, 0, :] + [x0, y0]) * scale).round().astype(np.int32)
                cv2.polylines(small, [pts], True, color, 1, cv2.LINE_AA)
        if np.isfinite(cx):
            p = (int(round((cx - 0.5) * scale)), int(round((cy - 0.5) * scale)))
            cv2.circle(small, p, 2, color, -1, cv2.LINE_AA)
            cv2.putText(small, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(small, f"t = {t:.3f} s   frame {frame}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def _flags(name, frames, t, x, y, area):
    msgs = []
    lost = ~np.isfinite(x)
    if lost.any():
        msgs.append(f"{name}: lost in {int(lost.sum())} of {len(x)} frames (first at t = {t[np.argmax(lost)]:.3f} s)")
    ok = np.nonzero(~lost)[0]
    if len(ok) >= 2:
        sp = np.hypot(np.diff(x[ok]), np.diff(y[ok])) / np.diff(t[ok])
        for j in np.nonzero(sp > MAX_SPEED_MM_S)[0][:5]:
            msgs.append(f"{name}: jumps {sp[j] * np.diff(t[ok])[j]:.2f} mm at t = {t[ok][j + 1]:.3f} s "
                        f"(frame {frames[ok][j + 1]}): check the video there")
        a = area[ok].astype(float)
        med = np.median(a)
        big = np.nonzero((a > 2 * med) | (a < 0.5 * med))[0]
        if len(big):
            msgs.append(f"{name}: outline size changes by more than 2x in {len(big)} frames (first at "
                        f"t = {t[ok][big[0]]:.3f} s): two shrimp touching, or a lost outline")
    return msgs


def track_video(video, export, model="edgetam", seconds=None, step=None, fps=None, out=None, device="auto",
                segmenter=None, overlay=True, manifest="data/manifest.csv", edgetam_checkpoint=None,
                save_every=200, log=print) -> dict:
    """Run the whole thing; returns {"files": [...], "overlay": path or None, "flags": [...], "plan": Plan}."""
    video, export = Path(video), Path(export)
    if not video.exists():
        raise FileNotFoundError(f"No such video: {video}")
    plan = make_plan(export, seconds, step, fps, video, manifest)
    if out is None:  # next to the export; a start file kept in extra/ writes next to extra/
        home = export.parent.parent if export.parent.name == "extra" else export.parent
        out = home / model
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    frames = plan.frames
    if n_video and frames[-1] >= n_video:
        frames = [f for f in frames if f < n_video]
        log(f"  The video has {n_video} frames: tracking only up to frame {frames[-1]}.")
    log(f"{model}: {len(plan.names)} shrimp ({', '.join(plan.names)}), frames {frames[0]}-{frames[-1]} every "
        f"{plan.step} ({len(frames)} frames, {len(frames) * plan.step / plan.fps:.1f} s at fps_true = {plan.fps:g}); "
        f"scale {1000 * plan.calibration.mm_per_px:.2f} um per pixel")
    if plan.calibration.rms_mm > 1e-3:
        log(f"  WARNING: the pixel and mm columns of {export.name} do not fit one calibration "
            f"(rms {1000 * plan.calibration.rms_mm:.1f} um). Were they exported from the same .trk?")
    if segmenter is None:
        log("  loading the model (the first time this downloads it) ...")
        segmenter = TransformersSegmenter(model, device, edgetam_checkpoint)
        log(f"  running on: {segmenter.device}")
    nobj = len(plan.names)
    px = np.full((nobj, len(frames)), np.nan)
    py = np.full((nobj, len(frames)), np.nan)
    area = np.zeros((nobj, len(frames)), int)
    t = np.array(frames, float) / plan.fps
    writer = None
    if overlay:
        import imageio_ffmpeg

        overlay_path = out / "overlay.mp4"
    done = 0

    def save():
        files = []
        for k, name in enumerate(plan.names):
            x, y = plan.calibration.to_mm(px[k], py[k])
            fname = out / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}.csv"
            write_tracker_file(fname, name, frames, t, x, y, px[k], py[k])
            files.append(fname)
        return files

    t0 = time.time()
    try:
        for i, (frame, rgb) in enumerate(iter_rgb_frames(video, frames)):
            masks = segmenter.start(rgb, plan.points_px) if i == 0 else segmenter.step(rgb)
            centers = []
            for k, m in enumerate(masks[:nobj]):
                cx, cy, a = mask_center(m)
                px[k, i], py[k, i], area[k, i] = cx, cy, a
                centers.append((cx, cy))
            if overlay:
                img = _overlay(rgb, masks, plan.names, centers, frame, t[i])
                if writer is None:
                    writer = imageio_ffmpeg.write_frames(str(overlay_path), (img.shape[1], img.shape[0]), fps=30,
                                                         codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
                                                         output_params=["-crf", "23"], quality=None)
                    writer.send(None)
                writer.send(np.ascontiguousarray(img))
            done = i + 1
            if i in (0, 4) or done % 50 == 0 or done == len(frames):
                rate = (time.time() - t0) / done
                log(f"  frame {done}/{len(frames)}: {rate:.2f} s per frame, about "
                    f"{rate * (len(frames) - done) / 60:.0f} min left")
            if save_every and done % save_every == 0:
                save()
    except KeyboardInterrupt:
        log(f"  stopped at frame {done}; saving what was tracked so far")
    finally:
        if writer is not None:
            writer.close()
    files = save()
    flags = []
    for k, name in enumerate(plan.names):
        x, y = plan.calibration.to_mm(px[k], py[k])
        flags += _flags(name, np.array(frames), t, np.asarray(x), np.asarray(y), area[k])
    elapsed = time.time() - t0
    info = [f"model: {model} ({MODELS.get(model, model)})", f"video: {video.name}", f"export: {export.name}",
            f"frames: {frames[0]} to {frames[-1]} every {plan.step} ({done} tracked)", f"fps_true: {plan.fps:g}",
            f"scale: {1000 * plan.calibration.mm_per_px:.3f} um per pixel (calibration rms "
            f"{1000 * plan.calibration.rms_mm:.4f} um)", f"time: {elapsed / 60:.1f} min "
            f"({elapsed / max(done, 1):.2f} s per frame)", "position: center of the outline (mask), in Tracker's axes"]
    (out / "run.log").write_text("\n".join(info + ["", "checks:"] + (flags or ["none"])) + "\n")
    for msg in flags:
        log("  CHECK: " + msg)
    log(f"  saved {', '.join(str(f) for f in files)}" + (f" and {overlay_path}" if overlay else ""))
    return {"files": files, "overlay": overlay_path if overlay else None, "flags": flags, "plan": plan,
            "seconds_per_frame": elapsed / max(done, 1)}


EXTRA_PER_SHRIMP = {"edgetam": 0.5, "sam2": 0.6, "sam2-small": 0.6}  # extra time per extra shrimp (measured)


def selftest(model="edgetam", device="auto", segmenter=None, folder=None, log=print) -> dict:
    """Check the installation and time the model on THIS computer: a made-up 1080p video of one
    shrimp-sized dark ellipse swimming 5 mm/s, tracked for 20 frames at step 2."""
    import tempfile

    folder = Path(folder or tempfile.mkdtemp(prefix="shrimp-selftest-"))
    folder.mkdir(parents=True, exist_ok=True)
    video, export = folder / "selftest_tracker.mp4", folder / "selftest.csv"
    w, h, mm_per_px, n = 1920, 1080, 0.0324, 40
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 205.0 - 15.0 * ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (0.49 * h) ** 2
    truth = [(700.0 + 0.6 * f, 500.0 + 0.2 * f) for f in range(n)]  # array coordinates: pixel centers at integers
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 240, (w, h))
    angle = float(np.degrees(np.arctan2(0.2, 0.6)))
    for x, y in truth:
        img = base.copy()
        cv2.ellipse(img, (int(round(x * 16)), int(round(y * 16))), (8 * 16, 3 * 16), angle, 0, 360, 70.0, -1,
                    cv2.LINE_AA, 4)
        img = np.clip(img + rng.normal(0, 3.0, img.shape), 0, 255).astype(np.uint8)
        out.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    out.release()
    frames = list(range(0, n, 2))
    tpx = np.array([truth[f][0] + 0.5 for f in frames])  # Tracker's pixel convention (+0.5)
    tpy = np.array([truth[f][1] + 0.5 for f in frames])
    write_tracker_file(export, "selftest", frames, np.array(frames) / 240.0, (tpx - w / 2) * mm_per_px,
                       -(tpy - h / 2) * mm_per_px, tpx, tpy)
    res = track_video(video, export, model=model, fps=240.0, out=folder / model, device=device,
                      segmenter=segmenter, overlay=False, manifest=folder / "none.csv", log=log)
    got = pd.read_csv(res["files"][0], skiprows=1)
    err = np.hypot(got["pixelx"] - tpx, got["pixely"] - tpy)
    spf = res["seconds_per_frame"]
    extra = EXTRA_PER_SHRIMP.get(model, 0.6)
    one, ten = spf * 1200 / 60, spf * (1 + 9 * extra) * 1200 / 60
    ok = bool(np.isfinite(err).all() and np.nanmax(err) < 3.0)
    log(f"\n{'OK' if ok else 'PROBLEM'}: {model} followed the test shrimp within {np.nanmax(err):.1f} pixels "
        f"(should be under 3). {spf:.2f} s per frame here.")
    log(f"Estimate for 10 s at step 2 (1,200 frames): one shrimp about {one:.0f} min; "
        f"10 shrimp together about {ten:.0f} min.")
    return {"ok": ok, "max_error_px": float(np.nanmax(err)), "seconds_per_frame": spf, "minutes_one": one,
            "minutes_ten": ten}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m shrimp.segment", description=__doc__.split("\n")[0])
    p.add_argument("video", nargs="?", help="the ..._tracker.mp4 video you opened in Tracker")
    p.add_argument("export", nargs="?", help="file exported from Tracker with the columns frame, x, y, pixelx, pixely")
    p.add_argument("--selftest", action="store_true", help="check the installation and time the model on a made-up clip")
    p.add_argument("--model", default="edgetam", choices=sorted(MODELS))
    p.add_argument("--seconds", type=float)
    p.add_argument("--step", type=int)
    p.add_argument("--fps", type=float)
    p.add_argument("--out")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--no-video", action="store_true", help="do not write the overlay video")
    p.add_argument("--edgetam-checkpoint", help=argparse.SUPPRESS)  # a local edgetam.pt (testing)
    a = p.parse_args(argv)
    if not a.selftest and not (a.video and a.export):
        p.error("give the video and the Tracker export (or --selftest)")
    try:
        import torch  # noqa: F401
        import transformers

        if transformers.__version__ != TRANSFORMERS_VERSION:
            print(f"Note: tested with transformers {TRANSFORMERS_VERSION}, you have {transformers.__version__}.")
    except ImportError:
        print("PyTorch and transformers are not installed in this environment. Run it like this "
              "(the first time takes a few minutes):\n"
              f"  {RUN_COMMAND} {' '.join(sys.argv[1:] if argv is None else argv)}")
        return 2
    if a.selftest:
        return 0 if selftest(a.model, a.device)["ok"] else 1
    try:
        track_video(a.video, a.export, a.model, a.seconds, a.step, a.fps, a.out, a.device,
                    overlay=not a.no_video, edgetam_checkpoint=a.edgetam_checkpoint)
    except (ValueError, FileNotFoundError, IOError) as err:
        print(f"ERROR: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
