"""The ported code does what last week's did, on random inputs (SPEC 0, 13.1 Centroid, 13.3).

tests/test_port_fidelity.py shows that the source text of every moved function equals the
template's. These tests run the two copies side by side on 200 random inputs each and compare what
they return, raise or write: `fit_calibration`, `write_tracker_file`, `make_plan` (and the reader
under it) and `mask_center`. The reference is `shrimp.segment` from tests/reference, which pytest
puts on the path; the package never imports it.

Both copies run in the same process on the same platform. Nothing is compared with a literal text,
because the Tracker-format writer uses the platform's line ends (CRLF on Windows).
"""

import dataclasses

import numpy as np
import pandas as pd
from conftest import java_sci
from shrimp import segment as reference

from outline_tracker import measure, tracker_io

N_RANDOM = 200


def _outcome(function, *args, **kwargs):
    """What a call gives, in a form that compares with `==` (NaN equal to NaN, types included).

    ("ok", repr of the value) or ("error", exception type, message). A dataclass is turned into a dict
    of its fields, so the new class and the reference class give the same text.
    """
    try:
        value = function(*args, **kwargs)
    except Exception as error:
        return "error", type(error).__name__, str(error)
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    return "ok", repr(value)


def _random_map(rng, mirrored=False):
    """A random pixel -> mm map as a function of arrays: Tracker's flipped similarity, or a mirrored one."""
    k = 10 ** rng.uniform(-2.5, 0.0)  # mm per pixel
    alpha = rng.uniform(-np.pi, np.pi)
    a, c = k * np.cos(alpha), k * np.sin(alpha)
    shift = rng.uniform(-300, 300, 2)
    m = np.array([[a, -c], [c, a]]) if mirrored else np.array([[a, c], [c, -a]])

    def to_mm(px, py):
        px, py = np.asarray(px, float), np.asarray(py, float)
        return m[0, 0] * px + m[0, 1] * py + shift[0], m[1, 0] * px + m[1, 1] * py + shift[1]

    return to_mm


# ------------------------------------------------------------------------------ fit_calibration


def _fit_inputs(seed):
    """Random arguments of fit_calibration: mostly good, some degenerate (one line, two points, one point,
    one place, missing values), exact or noisy, as lists or arrays."""
    rng = np.random.default_rng(seed)
    kind = seed % 9
    n = {2: 2, 3: 1}.get(kind, int(rng.integers(3, 41)))
    px, py = rng.uniform(0, 1920, n), rng.uniform(0, 1080, n)
    if kind == 1:  # a shrimp swimming in a straight line
        py = rng.uniform(0.2, 2.0) * px + 30
    if kind == 4:  # every point in one place
        px, py = np.full(n, 640.5), np.full(n, 360.5)
    if kind == 6:  # two points a given distance apart: the smallest spread that is accepted is 1 px
        n = 2
        px, py = np.array([640.5, 640.5 + rng.choice([0.5, 1.0, 1.2, 2.0])]), np.full(n, 360.5)
    x, y = _random_map(rng, mirrored=rng.random() < 0.3)(px, py)
    noise = rng.choice([0.0, 1e-9, 1e-3, 5e-2])
    x, y = x + noise * rng.normal(size=n), y + noise * rng.normal(size=n)
    if kind == 5:  # some points without a value
        for v in (px, py, x, y):
            v[rng.random(n) < 0.3] = np.nan
    if rng.random() < 0.5:
        return px.tolist(), py.tolist(), x.tolist(), y.tolist()
    return px, py, x, y


def test_fit_calibration_returns_the_same_fields_on_200_random_inputs():
    flips, errors = set(), 0
    for seed in range(N_RANDOM):
        px, py, x, y = _fit_inputs(seed)
        new, old = _outcome(tracker_io.fit_calibration, px, py, x, y), _outcome(reference.fit_calibration, px, py, x, y)
        assert new == old, f"seed {seed}"
        if new[0] == "error":
            errors += 1
        else:
            flips.add("'flip': True" in new[1])
    # The cases cover both forms of the map and the refusal, so the comparison is not vacuous.
    assert flips == {True, False}
    assert 10 <= errors <= N_RANDOM // 2


def test_fit_calibration_chooses_the_same_form_from_noise_free_to_very_noisy_mirrored_data():
    # Data from a mirrored map fit the unflipped form until the noise gets as large as the misfit of
    # the flipped form; the choice is made by a threshold on the ratio of the two residuals. Sweep the
    # noise through that ratio, so that the threshold itself is tested.
    rng = np.random.default_rng(11)
    px, py = rng.uniform(0, 1920, 12), rng.uniform(0, 1080, 12)
    x, y = _random_map(rng, mirrored=True)(px, py)
    z = rng.normal(size=(2, 12))
    flips = []
    for sigma_mm in np.geomspace(1e-3, 1e3, 400):
        new = _outcome(tracker_io.fit_calibration, px, py, x + sigma_mm * z[0], y + sigma_mm * z[1])
        old = _outcome(reference.fit_calibration, px, py, x + sigma_mm * z[0], y + sigma_mm * z[1])
        assert new == old, f"noise {sigma_mm} mm"
        flips.append("'flip': True" in new[1])
    assert flips[0] is False and flips[-1] is True  # the sweep goes through the change of form


def test_calibration_methods_agree_with_the_reference_on_random_points():
    rng = np.random.default_rng(7)
    for seed in range(N_RANDOM // 4):
        px, py = rng.uniform(0, 1920, 12), rng.uniform(0, 1080, 12)
        x, y = _random_map(rng, mirrored=seed % 4 == 3)(px, py)
        new, old = tracker_io.fit_calibration(px, py, x, y), reference.fit_calibration(px, py, x, y)
        probe_px, probe_py = rng.uniform(-100, 2000, 5), rng.uniform(-100, 1200, 5)
        probe_x, probe_y = rng.uniform(-500, 500, 5), rng.uniform(-500, 500, 5)
        assert repr(new.mm_per_px) == repr(old.mm_per_px), f"seed {seed}"
        assert new.matrix().tolist() == old.matrix().tolist(), f"seed {seed}"
        assert [v.tolist() for v in new.to_mm(probe_px, probe_py)] == [v.tolist() for v in old.to_mm(probe_px, probe_py)]
        assert [v.tolist() for v in new.to_px(probe_x, probe_y)] == [v.tolist() for v in old.to_px(probe_x, probe_y)]


# ------------------------------------------------------------------------------ write_tracker_file


def _file_columns(seed):
    """Random arguments of write_tracker_file (after the path and the name): frames, t, x, y, px, py."""
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(0, 31))
    frames = np.sort(rng.integers(0, 100_000, n))
    t = frames / rng.uniform(24, 480)

    def column(low_exp, high_exp):  # magnitudes from tiny to large, both signs, some lost or infinite
        v = rng.choice([-1.0, 1.0], n) * 10 ** rng.uniform(low_exp, high_exp, n)
        v[rng.random(n) < 0.2] = np.nan
        v[rng.random(n) < 0.03] = np.inf
        return v

    x, y, px, py = column(-9, 4), column(-9, 4), column(-5, 4), column(-5, 4)
    if seed % 3 == 0:
        return frames.tolist(), t.tolist(), x.tolist(), y.tolist(), px.tolist(), py.tolist()
    if seed % 3 == 1:
        return frames, t, x, y, px, py
    return frames.astype(float), pd.Series(t), pd.Series(x), y, px, py


def test_write_tracker_file_writes_the_same_bytes_on_200_random_inputs(tmp_path):
    folder = tmp_path / "tracks é"  # a space and a non-ASCII letter in the folder name
    folder.mkdir()
    names = ["A", "mass A", "B2", "shrimp 1", ""]
    for seed in range(N_RANDOM):
        name, columns = names[seed % len(names)], _file_columns(seed)
        new, old = folder / f"new_{seed}.csv", folder / f"old_{seed}.csv"
        assert tracker_io.write_tracker_file(new, name, *columns) is None
        reference.write_tracker_file(old, name, *columns)
        assert new.read_bytes() == old.read_bytes(), f"seed {seed}"
        # Not a comparison of two empty files: a name line, the column names, one line per frame.
        lines = new.read_bytes().splitlines()
        assert lines[:2] == [f",{name},,,,,".encode(), b"t,frame,x,y,pixelx,pixely"], f"seed {seed}"
        assert len(lines) == 2 + len(columns[0]), f"seed {seed}"


# ------------------------------------------------------------------------------ make_plan


def _manifest_text(rng, stem):
    """A random manifest.csv: this clip once (the usual case), twice, never, a bad fps, or the wrong columns."""
    kind = int(rng.integers(0, 6))
    fps = float(rng.uniform(100, 300))
    other = "other_clip.MOV,200.0"
    return {
        0: f"video_file,fps_true\n{stem}.MOV,{fps}\n{other}\n",
        1: f"video_file,fps_true\n{stem}.MOV,{fps}\n{stem}.mp4,{fps + 1}\n",
        2: f"video_file,fps_true\n{other}\n",
        3: f"video_file,fps_true\n{stem}.MOV,not a number\n",
        4: f"name,rate\n{stem}.MOV,{fps}\n",
        5: "",
    }[kind]


def _export_text(rng, to_mm, sep, newline):
    """A random Tracker export: one track named after its file, or several point masses ("#multi:")."""
    multi = rng.random() < 0.4
    names = [chr(ord("A") + i) for i in range(int(rng.integers(2, 5)))] if multi else ["mass A"]
    start = int(rng.integers(0, 3000))
    fps_tracker = float(rng.uniform(24, 480))
    step = int(rng.choice([1, 2, 2, 3, 5]))
    if multi:
        frames = [start, start + int(rng.integers(1, 4)) * step]
    else:
        frames = [start + i * step for i in range(int(rng.integers(1, 41)))]
    per = ["frame", "x", "y", "pixelx", "pixely"]
    lines = ["#multi:"] if multi else []
    lines.append(sep + "".join(n + sep * 5 for n in names))
    lines.append("t" + sep + sep.join(sep.join(per) for _ in names) + sep)
    late = multi and rng.random() < 0.15  # one point mass is marked on a later frame: an error
    for i, frame in enumerate(frames):
        cells = [java_sci((frame - start) / fps_tracker)]
        for k, _ in enumerate(names):
            if multi:  # all on the first frame; when `late`, the last one on the second frame instead
                marked = (i == 0) != (late and k == len(names) - 1)
            else:  # a track: some frames have no mark
                marked = rng.random() > 0.1
            if marked:
                px, py = rng.uniform(0, 1920), rng.uniform(0, 1080)
                x, y = to_mm(px, py)
                cells += [java_sci(float(v)) for v in (frame, x, y, px, py)]
            else:
                cells += [""] * len(per)
        lines.append(sep.join(cells) + sep)
    return names, newline.join(lines) + newline


def _write_export(seed, folder):
    """Write a random Tracker export into `folder` (comma, tab or semicolon; LF or CRLF; maybe a BOM)."""
    rng = np.random.default_rng(5000 + seed)
    sep, newline = [(",", "\n"), (",", "\r\n"), ("\t", "\n"), (";", "\n")][int(rng.integers(0, 4))]
    names, text = _export_text(rng, _random_map(rng), sep, newline)
    export = folder / (f"export_{seed}.csv" if len(names) == 1 else f"start_{seed}.csv")
    export.write_bytes((b"\xef\xbb\xbf" if rng.random() < 0.15 else b"") + text.encode("utf-8"))
    return str(export) if rng.random() < 0.5 else export


def _plan_kwargs(seed, folder):
    """Random options of make_plan, with a manifest in `folder` (given always: never the working folder's)."""
    rng = np.random.default_rng(7000 + seed)
    stem = f"clip{seed}"
    kwargs = {"manifest": folder / f"manifest_{seed}.csv"}
    if rng.random() < 0.7:
        kwargs["manifest"].write_text(_manifest_text(rng, stem))
    if rng.random() < 0.6:
        kwargs["video"] = folder / f"{stem}_tracker.mp4"
    if rng.random() < 0.3:
        kwargs["fps"] = float(rng.uniform(24, 480))
    if rng.random() < 0.3:
        kwargs["seconds"] = float(rng.choice([0.0, 0.5, 3.0, 12.5]))
    if rng.random() < 0.3:
        kwargs["step"] = int(rng.choice([0, 1, 2, 3, 7]))
    return kwargs


def test_make_plan_returns_the_same_plan_on_200_random_inputs(tmp_path):
    folder = tmp_path / "exports é"
    folder.mkdir()
    plans, messages = 0, set()
    for seed in range(N_RANDOM):
        export, kwargs = _write_export(seed, folder), _plan_kwargs(seed, folder)
        new, old = _outcome(tracker_io.make_plan, export, **kwargs), _outcome(reference.make_plan, export, **kwargs)
        assert new == old, f"seed {seed}: {kwargs}"
        if new[0] == "ok":
            plans += 1
        else:
            messages.add(new[2][:15])
    # Many plans and several different refusals, so the comparison is not a comparison of errors only.
    assert plans >= N_RANDOM // 2
    assert len(messages) >= 3, messages


def test_read_tracker_export_returns_the_same_tables_on_200_random_files(tmp_path):
    folder = tmp_path / "exports \u00e9"
    folder.mkdir()
    tables = 0
    for seed in range(N_RANDOM):
        export = _write_export(seed, folder)
        new, old = tracker_io.read_tracker_export(export), reference.read_tracker_export(export)
        assert list(new) == list(old), f"seed {seed}"
        for name in new:
            pd.testing.assert_frame_equal(new[name], old[name], check_exact=True)
            tables += 1
    assert tables >= N_RANDOM


# ------------------------------------------------------------------------------ mask_center


def _random_mask(seed):
    """A random mask: a size (sometimes empty), a density (sometimes none or all), a dtype and a memory layout."""
    rng = np.random.default_rng(9000 + seed)
    h, w = (0, int(rng.integers(1, 9))) if seed % 31 == 0 else (int(rng.integers(1, 61)), int(rng.integers(1, 61)))
    density = [0.0, 0.002, 0.05, 0.3, 0.7, 1.0][int(rng.integers(0, 6))]
    mask = rng.random((h, w)) < density
    mask = [mask, mask.astype(np.uint8), mask.astype(np.uint8) * 255, mask.astype(np.int32),
            mask.astype(np.float32)][seed % 5]
    return [mask, mask.T, mask[::2, ::3]][seed % 3]


def test_mask_center_returns_the_same_triple_on_200_random_masks():
    empty = found = 0
    for seed in range(N_RANDOM):
        mask = _random_mask(seed)
        new, old = measure.mask_center(mask), reference.mask_center(mask)
        assert repr(new) == repr(old), f"seed {seed}"  # NaN equal to NaN, int and float kept apart
        assert [type(v) for v in new] == [float, float, int]
        if new[2] == 0:
            empty += 1
        else:
            found += 1
    assert empty >= 10 and found >= 100
