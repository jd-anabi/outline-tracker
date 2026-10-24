"""The frozen numbers of the real model, where they are read, and how a run is judged by them.

The numbers are two text files of tests/data/ (the first layout of tests/support/frozen_helpers.py):
`POSITIONS`, where EdgeTAM found each object of two synthetic clips on the processor, and `WEIGHTS`, the
hash and size of the weights that gave them. A position is (u_px, v_px): px in Tracker's convention
(pixel centers at +0.5, u to the right, v downward) in the full frame; a frame is a video frame number.
The slow tests that compare a run with them (tests/slow/test_frozen_reference.py) leave two decisions to
this module: on which machine the limit of 0.01 px is asserted (`same_machine`), and how positions are
judged (`compare_with_frozen`). They ask both with one call for a clip of the table
(`compare_with_the_frozen_positions`). Where the limit was not asserted, one test of that file is
skipped, and this module words its reason (`why_the_limit_was_not_asserted`).

How a frozen file is written and read, and the golden Tracker-format files, are in
tests/support/frozen_helpers.py. This is a helper of the tests, not a part of the package. It imports no torch.
"""

import math
import platform
import sys
from pathlib import Path

import numpy as np
from frozen_helpers import DATA, machine_name, read_frozen

from outline_tracker.core import provenance

POSITIONS = DATA / "edgetam_cpu_positions.csv"  # where the real model found each object, on the processor
WEIGHTS = DATA / "edgetam_weights.txt"  # the hash and size of the weights that gave those positions
COLUMNS = "clip,frame,track_id,u_px,v_px"  # the column line of the positions table
LIMIT_PX = 0.01  # on the machine that froze them, a position is at most this far from its frozen one, px
TRUTH_PX = 3.0  # on every machine, a found center is under this far from the true one, px (SPEC 13.4)


# ---------------------------------------------------------------------------------------------
# The frozen numbers of the real model


def position_rows(clip: str, frames, names, positions) -> list[str]:
    """The rows of the positions table for one clip, in the order of `COLUMNS`: one row per frame and
    object, the frames in the given order and on each frame the objects in the order of `names`.
    `positions[i][k]` is (u_px, v_px) of the object `names[k]` on the frame `frames[i]`, or NaN, NaN
    where it was not found. A position is written with 4 decimals; a lost row has two empty cells."""
    rows = []
    for frame, found in zip(frames, positions, strict=True):
        for name, (u_px, v_px) in zip(names, found, strict=True):
            lost = math.isnan(u_px) or math.isnan(v_px)
            rows.append(f"{clip},{int(frame)},{name}," + ("," if lost else f"{u_px:.4f},{v_px:.4f}"))
    return rows


def read_positions(path=POSITIONS) -> tuple[dict[str, str], list[tuple[str, int, str, float, float]]]:
    """A positions table as (header, rows): the header as `read_frozen` gives it, and each row as
    (clip, frame, track_id, u_px, v_px) in the file's order, with NaN, NaN for a lost row. Raises
    ValueError when the line after the header is not `COLUMNS`."""
    header, lines = read_frozen(path)
    if not lines or lines[0] != COLUMNS:
        raise ValueError(f"{Path(path).name}: the line after the header is not {COLUMNS}")
    rows = []
    for line in lines[1:]:
        clip, frame, track_id, u_px, v_px = line.split(",")
        rows.append((clip, int(frame), track_id, float(u_px or "nan"), float(v_px or "nan")))
    return header, rows


def read_weights(path=WEIGHTS) -> tuple[dict[str, str], dict[str, str]]:
    """The frozen weights file as (header, facts): the header as `read_frozen` gives it, and the rows
    after it, each `key: value`, as a dict in the file's order: `sha256` and `bytes` of the converted
    weights first. The values are text as written."""
    header, rows = read_frozen(path)
    return header, dict(row.split(": ", 1) for row in rows)


def machine_here(weights_sha256: str | None = None) -> dict[str, str]:
    """This machine as the machine rule (`same_machine`) sees it: `platform` (the operating system's
    family as Python names it: darwin, linux, win32), `architecture` (arm64, x86_64, AMD64), `chip`
    (`machine_name()`), `torch` (the installed version, read without importing torch) and, when a hash
    is given, `weights sha256`: the SHA-256 of the weights file that the loaded model was read from."""
    here = {"platform": sys.platform, "architecture": platform.machine(), "chip": machine_name(),
            "torch": provenance.library_versions(("torch",))["torch"]}
    return here if weights_sha256 is None else {**here, "weights sha256": weights_sha256}


def machine_of(header: dict[str, str]) -> dict[str, str]:
    """The machine that made a frozen file, read from its header, with the keys of `machine_here`:
    `platform` and `architecture` are the last two parts of the `decoder` line
    (`opencv-5.0.0/darwin/arm64`), `chip` is the `machine` line, `torch` is its version in the
    `libraries` line, `weights sha256` is the line of that name. A fact that the header does not hold
    is ""."""
    decoder = header.get("decoder", "").split("/")
    family, architecture = decoder[1:] if len(decoder) == 3 else ("", "")
    libraries = dict(item.partition(" ")[::2] for item in header.get("libraries", "").split("; "))
    return {"platform": family, "architecture": architecture, "chip": header.get("machine", ""),
            "torch": libraries.get("torch", ""), "weights sha256": header.get("weights sha256", "")}


# The facts of `machine_here`, as a sentence names them.
_FACTS = {"platform": "operating system", "architecture": "architecture", "chip": "chip", "torch": "torch",
          "weights sha256": "hash of the weights"}


def same_machine(header: dict[str, str], here: dict[str, str]) -> bool:
    """The machine rule: whether `here` (`machine_here`) is the machine that made the frozen file with
    this header.

    docs/ROADMAP.md, W1 step 4: "The limit is 0.01 px on the machine that froze the numbers; on another
    machine it is measured, not assumed." So a test asserts `LIMIT_PX` only where this is true: every
    fact of `here` is the header's, and none is empty: the operating system's family, the architecture,
    the chip, the torch version and, when `here` holds one, the hash of the weights. Everything else is
    another machine: there the test asserts what holds on every machine and prints what it measured.
    """
    made = machine_of(header)
    return all(value and made.get(key) == value for key, value in here.items())


def why_the_limit_was_not_asserted(header: dict[str, str], here: dict[str, str]) -> str:
    """What a run says where the machine rule gives "another machine": "" when `same_machine(header,
    here)`, where `LIMIT_PX` was asserted; otherwise the reason, in two sentences. The first names
    each fact of `here` (`machine_here`) that is not the header's, this machine's value before the
    header's, in the order of `here`; a fact that could not be read here is "not known", one that the
    header does not hold is "not named". The second says what the position tests asserted instead
    (`compare_with_frozen`). A passing test prints nothing that pytest's summary shows, so the slow
    tests skip one test with this reason (tests/slow/test_frozen_reference.py)."""
    made = machine_of(header)
    other = [f"{_FACTS[key]}: {value or 'not known'} here, {made.get(key) or 'not named'} in the header"
             for key, value in here.items() if not value or made.get(key) != value]
    if not other:
        return ""
    return (f"The limit of {LIMIT_PX} px was not asserted: this is not the machine that froze the numbers "
            f"({'; '.join(other)}). The position tests asserted instead that no row is lost and that every position "
            f"is under {TRUTH_PX:g} px from its true center, and printed the distance from the frozen positions "
            "(shown with -rP).")


def compare_with_frozen(what: str, found, frozen, true, strict: bool) -> None:
    """Judge the positions of a run against the frozen ones, and print what was measured in a line
    that starts with `VALIDATION`.

    `found`, `frozen`, `true`: arrays of one shape (..., 2) that hold (u_px, v_px) for the same rows:
    what this run found, what the frozen table holds, and the true centers; NaN, NaN for a lost row.
    `what` names the clip and the level in the printed lines.

    Asserted always: the rows that are lost are those of the frozen table. When `strict` (this is the
    machine that froze the numbers, `same_machine`): every row is at most `LIMIT_PX` from its frozen
    position. Otherwise: no row is lost and every row is under `TRUTH_PX` from its true center; the
    largest distance from the frozen positions is printed in a second line, which says that the limit
    was not measured for this machine.
    """
    found, frozen, true = (np.asarray(positions, float).reshape(-1, 2) for positions in (found, frozen, true))
    lost, lost_frozen = np.isnan(found).any(axis=1), np.isnan(frozen).any(axis=1)
    both = ~(lost | lost_frozen)
    from_frozen = float(np.hypot(*(found - frozen)[both].T).max()) if both.any() else float("nan")
    from_truth = float(np.hypot(*(found - true)[~lost].T).max()) if not lost.all() else float("nan")
    print(f"VALIDATION frozen numbers, {what}: max distance from the frozen positions {from_frozen:.4f} px over "
          f"{int(both.sum())} rows; max distance from the true centers {from_truth:.3f} px; lost rows: frozen "
          f"{int(lost_frozen.sum())}, found {int(lost.sum())}")
    assert np.array_equal(lost, lost_frozen), (
        f"{what}: the lost rows are not those of the frozen table (row numbers found "
        f"{np.flatnonzero(lost).tolist()}, frozen {np.flatnonzero(lost_frozen).tolist()})")
    if strict:
        assert from_frozen <= LIMIT_PX, f"{what}: {from_frozen:.4f} px from the frozen positions (limit {LIMIT_PX} px)"
        return
    print(f"VALIDATION frozen numbers, {what}: this is not the machine that froze the numbers, so the limit of "
          f"{LIMIT_PX} px was not measured for it; measured here: {from_frozen:.4f} px from the frozen positions")
    assert not lost.any(), f"{what}: {int(lost.sum())} rows are lost"
    assert from_truth < TRUTH_PX, f"{what}: {from_truth:.3f} px from the true centers (limit {TRUTH_PX:g} px)"


def _frozen_positions(clip: str, frames, names) -> tuple[dict[str, str], np.ndarray]:
    """The frozen table (tests/data/edgetam_cpu_positions.csv) for one of its clips: its header, and
    positions[i][k] = (u_px, v_px) of the object names[k] on the frame frames[i], px in Tracker's
    convention in the full frame. A frame or an object that the table does not hold is a KeyError."""
    header, rows = read_positions()
    table = {(row_clip, frame, track_id): (u_px, v_px) for row_clip, frame, track_id, u_px, v_px in rows}
    return header, np.array([[table[clip, frame, name] for name in names] for frame in frames])


def compare_with_the_frozen_positions(what: str, clip: str, frames, names, found, true, weights_sha256) -> None:
    """Judge the positions that a run found on a clip of the frozen table against that table
    (`compare_with_frozen`, which also prints what it measured): the rows that are lost are the
    table's, and every position is within 0.01 px of its frozen one on the machine that froze them,
    with the weights whose hash is `weights_sha256` (`same_machine`); on another machine no row is
    lost and every position is under 3 px from its true center.

    found, true: [i][k] = (u_px, v_px) of names[k] on frames[i], what the run found (NaN, NaN where
    lost) and the true center, px in Tracker's convention in the full frame.
    """
    header, frozen = _frozen_positions(clip, frames, names)
    compare_with_frozen(what, found, frozen, true, same_machine(header, machine_here(weights_sha256)))
