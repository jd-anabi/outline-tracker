"""Tests for the provided SAM 2 / EdgeTAM script (shrimp.segment). These must always pass.

They do not run the AI models (that needs PyTorch and a download); a simple stand-in finds a dark disk
instead, so everything around the model is tested: reading Tracker's exports, the pixel -> mm
calibration, frame numbers, time, and the files written.
"""

import cv2
import numpy as np
import pandas as pd
import pytest
from conftest import java_sci

from shrimp import segment

MM_PER_PX = 0.05
W, H = 320, 240


def tracker_map(px, py, angle_deg=0.0, origin=(160.0, 120.0)):
    """Tracker's pixel -> mm map: origin at `origin` (pixels), x axis rotated by angle_deg, y up."""
    a = np.radians(angle_deg)
    dx, dy = np.asarray(px) - origin[0], -(np.asarray(py) - origin[1])
    return (MM_PER_PX * (np.cos(a) * dx + np.sin(a) * dy), MM_PER_PX * (-np.sin(a) * dx + np.cos(a) * dy))


def export_text(rows, name="mass A"):
    """One point mass as Tracker exports it (name line, header with trailing comma, full precision)."""
    lines = [f",{name},,,,,", "t,frame,x,y,pixelx,pixely,"]
    for t, f, px, py in rows:
        x, y = tracker_map(px, py)
        lines.append(",".join(java_sci(float(v)) for v in (t, f, x, y, px, py)) + ",")
    return "\n".join(lines) + "\n"


def test_read_one_point_mass_named_after_the_file(tmp_path):
    p = tmp_path / "A.csv"
    p.write_text(export_text([(0.0, 120, 100.5, 80.5), (1 / 120, 122, 101.5, 80.5)]))
    tracks = segment.read_tracker_export(p)
    assert list(tracks) == ["A"]
    a = tracks["A"]
    assert list(a["frame"]) == [120, 122]
    assert list(a["pixelx"]) == pytest.approx([100.5, 101.5])
    assert a["x"].iloc[0] == pytest.approx(tracker_map(100.5, 80.5)[0])


def test_read_several_point_masses_exported_together(tmp_path):
    # Tracker's layout: "#multi:", a line of names, the columns repeated for each point mass
    p = tmp_path / "start.csv"
    rows = ["#multi:", ",A,,,,,B,,,,,C,,,,,", "t," + "frame,x,y,pixelx,pixely," * 3]
    cells = [java_sci(0.0)]
    for px, py in [(50.5, 60.5), (200.5, 100.5), (120.5, 200.5)]:
        x, y = tracker_map(px, py)
        cells += [java_sci(float(v)) for v in (300, x, y, px, py)]
    rows.append(",".join(cells) + ",")
    rows.append(java_sci(1 / 120) + "," + ",".join([""] * 15) + ",")  # a later time with no marks
    p.write_text("\n".join(rows) + "\n")
    tracks = segment.read_tracker_export(p)
    assert list(tracks) == ["A", "B", "C"]
    assert [len(t) for t in tracks.values()] == [1, 1, 1]
    assert tracks["B"]["pixelx"].iloc[0] == pytest.approx(200.5)
    assert tracks["C"]["frame"].iloc[0] == 300


def test_read_needs_the_pixel_columns(tmp_path):
    p = tmp_path / "A.csv"
    p.write_text(",A,,,\nt,frame,x,y,\n0,0,1,1,\n")
    with pytest.raises(ValueError, match="pixelx"):
        segment.read_tracker_export(p)


@pytest.mark.parametrize("angle", [0.0, 12.0])
def test_calibration_recovers_trackers_map(angle):
    rng = np.random.default_rng(0)
    px, py = rng.uniform(0, W, 30), rng.uniform(0, H, 30)
    x, y = tracker_map(px, py, angle)
    cal = segment.fit_calibration(px, py, x, y)
    assert cal.mm_per_px == pytest.approx(MM_PER_PX)
    assert cal.flip
    assert cal.rms_mm < 1e-9
    qx, qy = cal.to_mm(17.25, 203.5)
    ex, ey = tracker_map(17.25, 203.5, angle)
    assert (qx, qy) == pytest.approx((ex, ey))
    assert cal.to_px(qx, qy) == pytest.approx((17.25, 203.5))


def test_calibration_from_a_straight_track():
    px = np.linspace(10, 200, 40)  # a shrimp swimming in a straight line: all points on one line
    py = 0.5 * px + 30
    cal = segment.fit_calibration(px, py, *tracker_map(px, py))
    assert cal.mm_per_px == pytest.approx(MM_PER_PX)
    assert cal.to_mm(5.0, 230.0) == pytest.approx(tracker_map(5.0, 230.0))


@pytest.mark.parametrize("a, b", [((10.5, 20.5), (200.5, 130.5)), ((80.5, 60.5), (240.5, 170.5))])
def test_two_points_give_trackers_flipped_map(a, b):
    # A mirrored map also fits two points (or points on one line) exactly, so rounding used to decide,
    # differently on different computers. Tracker's y points up while image rows go down: its map is
    # always the flipped one.
    px, py = np.array([a[0], b[0]]), np.array([a[1], b[1]])
    cal = segment.fit_calibration(px, py, *tracker_map(px, py))
    assert cal.flip
    assert cal.to_mm(a[0] + 2, a[1]) == pytest.approx(tracker_map(a[0] + 2, a[1]))


def test_calibration_needs_two_points():
    with pytest.raises(ValueError):
        segment.fit_calibration([10.0], [10.0], [0.0], [0.0])


def test_mask_center_uses_trackers_pixel_convention():
    m = np.zeros((40, 40), bool)
    m[20:22, 10:13] = True  # rows 20-21, columns 10-12
    cx, cy, area = segment.mask_center(m)
    assert (cx, cy, area) == (pytest.approx(11.5), pytest.approx(21.0), 6)
    assert np.isnan(segment.mask_center(np.zeros((5, 5), bool))[0])


def test_written_file_reads_like_a_tracker_export(tmp_path):
    p = tmp_path / "A.csv"
    segment.write_tracker_file(p, "A", [0, 2, 4], [0.0, 1 / 120, 2 / 120], [1.0, np.nan, 1.2],
                               [2.0, np.nan, 2.2], [10.5, np.nan, 12.5], [20.5, np.nan, 22.5])
    lines = p.read_text().splitlines()
    assert "A" in lines[0]
    df = pd.read_csv(p, skiprows=1)
    assert list(df.columns) == ["t", "frame", "x", "y", "pixelx", "pixely"]
    assert list(df["frame"]) == [0, 2, 4]
    assert np.isnan(df["x"].iloc[1]) and df["x"].iloc[2] == pytest.approx(1.2)


def test_plan_follows_the_tracker_track(tmp_path):
    p = tmp_path / "A.csv"
    rows = [((f - 100) / 240.0, f, 100.5 + 0.1 * f, 80.5) for f in range(120, 361, 2)]  # Tracker's t from clip start
    p.write_text(export_text(rows))
    plan = segment.make_plan(p)
    assert (plan.start, plan.step, plan.n) == (120, 2, 121)
    assert plan.frames[-1] == 360
    assert plan.fps == pytest.approx(240.0, rel=1e-5)
    assert plan.points_px[0] == pytest.approx((112.5, 80.5))


def test_plan_for_many_shrimp_needs_fps_true(tmp_path):
    p = tmp_path / "start.csv"
    rows = ["#multi:", ",A,,,,,B,,,,,", "t," + "frame,x,y,pixelx,pixely," * 2]
    cells = [java_sci(0.0)]
    for px, py in [(50.5, 60.5), (200.5, 100.5)]:
        cells += [java_sci(float(v)) for v in (500, *tracker_map(px, py), px, py)]
    rows.append(",".join(cells) + ",")
    p.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="fps"):
        segment.make_plan(p, manifest=tmp_path / "none.csv")
    plan = segment.make_plan(p, fps=239.5)
    assert (plan.start, plan.step, plan.n) == (500, 2, round(10 * 239.5 / 2))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("video_file,fps_true\ngroupB_main.MOV,238.0\n")
    plan = segment.make_plan(p, video=tmp_path / "groupB_main_tracker.mp4", manifest=manifest)
    assert plan.fps == pytest.approx(238.0)


def test_every_shrimp_must_start_on_the_same_frame(tmp_path):
    p = tmp_path / "start.csv"
    rows = ["#multi:", ",A,,,,,B,,,,,", "t," + "frame,x,y,pixelx,pixely," * 2]
    a = [java_sci(float(v)) for v in (500, *tracker_map(50.5, 60.5), 50.5, 60.5)]
    b = [java_sci(float(v)) for v in (502, *tracker_map(80.5, 60.5), 80.5, 60.5)]
    rows.append(",".join([java_sci(0.0)] + a + [""] * 5) + ",")
    rows.append(",".join([java_sci(2 / 240)] + [""] * 5 + b) + ",")
    p.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="same first frame"):
        segment.make_plan(p, fps=240.0)


class DiskFinder:
    """Stand-in for SAM 2/EdgeTAM: for each object, the dark pixels around its last position."""

    def start(self, rgb, points):
        self.ps = [tuple(p) for p in points]
        return self.step(rgb)

    def step(self, rgb):
        n, lab, _, centers = cv2.connectedComponentsWithStats((rgb[:, :, 0] < 128).astype(np.uint8))
        masks = []
        for i, (cx, cy) in enumerate(self.ps):
            ids = [k for k in range(1, n) if np.hypot(centers[k][0] - int(cx), centers[k][1] - int(cy)) < 20]
            m = np.isin(lab, ids)
            if m.any():
                r, c = np.nonzero(m)
                self.ps[i] = (c.mean() + 0.5, r.mean() + 0.5)
            masks.append(m)
        return masks


def disk_video(path, centers, n=200):
    """A 240 fps video of dark disks; centers(n) gives their (x, y) in frame n (pixel centers at integers)."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 240, (W, H))
    for f in range(n):
        img = np.full((H, W, 3), 220, np.uint8)
        for x, y in centers(f):
            cv2.circle(img, (int(round(x * 16)), int(round(y * 16))), 6 * 16, (40, 40, 40), -1, cv2.LINE_AA, 4)
        out.write(img)
    out.release()


def test_whole_run_with_a_stand_in_model(tmp_path):
    # a dark disk moving 0.5 px per frame to the right; frame n shows it at x = 60 + 0.5 n (pixel centers)
    video = tmp_path / "clip_tracker.mp4"
    disk_video(video, lambda n: [(60 + 0.5 * n, 100)])
    export = tmp_path / "A.csv"
    frames = range(40, 161, 4)  # tracked in Tracker every 4th frame from frame 40
    export.write_text(export_text([((f - 40) / 240.0, f, 60.5 + 0.5 * f, 100.5) for f in frames]))
    res = segment.track_video(video, export, model="stand-in", segmenter=DiskFinder(), log=lambda *a: None)
    got = pd.read_csv(res["files"][0], skiprows=1)
    assert list(got["frame"]) == list(frames), "same frames as the Tracker track"
    assert np.allclose(got["t"], got["frame"] / 240.0)
    ex, ey = tracker_map(60.5 + 0.5 * got["frame"], 100.5)
    assert np.allclose(got["x"], ex, atol=0.1 * MM_PER_PX), "x in mm, from Tracker's calibration"
    assert np.allclose(got["y"], ey, atol=0.1 * MM_PER_PX)
    assert res["overlay"].exists() and res["overlay"].stat().st_size > 0
    assert (tmp_path / "stand-in" / "run.log").exists()  # not .txt: load_tracks reads .csv and .txt
    assert res["flags"] == []


def test_many_shrimp_from_a_start_file_in_extra(tmp_path):
    # two disks; both marked once on frame 40 and exported together into ana/extra/start.csv
    video = tmp_path / "clip_tracker.mp4"
    disk_video(video, lambda n: [(60 + 0.5 * n, 60), (250 - 0.25 * n, 170)])
    start = tmp_path / "ana" / "extra" / "start.csv"
    start.parent.mkdir(parents=True)
    rows = ["#multi:", ",A,,,,,B,,,,,", "t," + "frame,x,y,pixelx,pixely," * 2]
    cells = [java_sci(0.0)]
    for px, py in [(60.5 + 20, 60.5), (250.5 - 10, 170.5)]:
        cells += [java_sci(float(v)) for v in (40, *tracker_map(px, py), px, py)]
    rows.append(",".join(cells) + ",")
    start.write_text("\n".join(rows) + "\n")
    res = segment.track_video(video, start, model="stand-in", seconds=100 / 240, step=4, fps=240.0,
                              segmenter=DiskFinder(), overlay=False, log=lambda *a: None)
    assert [f.relative_to(tmp_path).as_posix() for f in res["files"]] == ["ana/stand-in/A.csv", "ana/stand-in/B.csv"]
    for f, (x0, vx, y0) in zip(res["files"], [(60.5, 0.5, 60.5), (250.5, -0.25, 170.5)]):
        got = pd.read_csv(f, skiprows=1)
        assert list(got["frame"]) == list(range(40, 140, 4))
        ex, ey = tracker_map(x0 + vx * got["frame"], y0)
        # thresholding a slowly moving disk puts its center within ~0.2 px of the truth
        assert np.allclose(got["x"], ex, atol=0.25 * MM_PER_PX) and np.allclose(got["y"], ey, atol=0.25 * MM_PER_PX)


def test_selftest_runs_and_reports_the_time(tmp_path):
    res = segment.selftest(model="stand-in", segmenter=DiskFinder(), folder=tmp_path, log=lambda *a: None)
    assert res["ok"] and res["max_error_px"] < 1.0
    assert res["minutes_ten"] > res["minutes_one"] > 0
