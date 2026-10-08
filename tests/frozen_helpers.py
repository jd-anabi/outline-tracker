"""Frozen reference files under tests/data/: how one is written, and how one is read.

A frozen file holds what the code itself gave, in a run that an independent check confirmed in that
same run (docs/ROADMAP.md, section 2, rule 2). Later runs compare with the file, so a test never
rewrites it: `write_frozen` writes only when the environment variable `OUTLINE_TRACKER_FREEZE` is
`1`, and only when the package is as committed, because the header names the commit.

Two layouts:
- a text file of its own: lines `# key: value` (the header), then its rows (`frozen_text`,
  `read_frozen`);
- files that are compared byte for byte and so have no header inside: a `HEADER.txt` in a folder above
  them holds the header and one row per file, `path,size in bytes,SHA-256` (`write_listing`).

Every frozen file is ASCII with LF line ends; `.gitattributes` keeps it so in every checkout. A header
holds no path under a home or temp folder and no user or computer name: write `command` relative to
the repository. tests/test_frozen_files.py holds the rules as tests.

The frozen numbers of the real model are two such text files (the last part of this module):
`POSITIONS`, where EdgeTAM found each object of two synthetic clips on the processor, and `WEIGHTS`, the
hash and size of the weights that gave them. A position is (u_px, v_px): px in Tracker's convention
(pixel centers at +0.5, u to the right, v downward) in the full frame; a frame is a video frame number.
The slow tests that compare a run with them leave two decisions to this module: on which machine the
limit of 0.01 px is asserted (`same_machine`), and how positions are judged (`compare_with_frozen`).

The golden Tracker-format files are files of the second layout (the part before the last): `GOLDEN`,
what the tool wrote for three cases with the stand-in model. The tests that compare a run with them
leave two decisions of the same kind to this module: with which decoder their bytes are asserted
(`same_decoder`), and how a file is judged with another decoder (`compare_with_golden`). There a
position is (pixelx, pixely), px in Tracker's convention too, and x, y are mm in the axes of the
case's export, y up.

This is a helper of the tests, not a part of the package. It imports no torch. Apart from those two
parts there are no units or coordinates here: text and bytes only.
"""

import hashlib
import math
import os
import platform
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

from outline_tracker import provenance, video

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tests" / "data"
SWITCH = "OUTLINE_TRACKER_FREEZE"  # the environment variable; `1` asks for the frozen files to be written
SIDECAR = "HEADER.txt"  # the header and the list of the files that have no header inside
LIBRARIES = ("torch", "torchvision", "transformers", "timm", "safetensors", "numpy", "opencv-python-headless",
             "scikit-image")  # named in a header, as far as installed
LEFT_BY_A_SYSTEM = (".DS_Store", "Thumbs.db", "desktop.ini")  # not frozen files (.gitignore has the same names)
_TIMEOUT_S = 30.0

POSITIONS = DATA / "edgetam_cpu_positions.csv"  # where the real model found each object, on the processor
WEIGHTS = DATA / "edgetam_weights.txt"  # the hash and size of the weights that gave those positions
COLUMNS = "clip,frame,track_id,u_px,v_px"  # the column line of the positions table
LIMIT_PX = 0.01  # on the machine that froze them, a position is at most this far from its frozen one, px
TRUTH_PX = 3.0  # on every machine, a found center is under this far from the true one, px (SPEC 13.4)

GOLDEN = DATA / "tracker_format"  # the golden Tracker-format files, <case>/<name>.csv, and their HEADER.txt
STAND_IN_PX = 0.25  # with every decoder, the stand-in finds a disk at most this far from its true center, px
_FLOAT_MM = 1e-9  # what floating point may add to a calibrated value, mm: far below a file's last decimal


def header_lines(command: str, extra: dict[str, str] | None = None) -> list[str]:
    """How a frozen file was made, one `key: value` per line, in this order: `made` (today's date),
    `commit` (the short hash of HEAD), `tool` (the tool's version line), `system` (the operating system
    and the Python version), `machine` (`machine_name()`), `decoder` (what decodes video frames here),
    `libraries` (name and version of those of `LIBRARIES` that are installed), `command` (as given: the
    command that makes the file, written relative to the repository), then the lines of `extra`.
    Raises when git cannot say the commit."""
    versions = provenance.library_versions(LIBRARIES)
    facts = {
        "made": date.today().isoformat(),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "tool": provenance.tool_version(),
        "system": provenance.machine_text(),
        "machine": machine_name(),
        "decoder": video.decoder_tag(),
        "libraries": "; ".join(f"{name} {found}" for name, found in versions.items() if found != "not installed"),
        "command": command,
    }
    return [f"{key}: {value}" for key, value in [*facts.items(), *(extra or {}).items()]]


def machine_name() -> str:
    """The chip as the operating system names it, for example "Apple M1 Max"; where that cannot be
    read, the processor name that `platform` gives. One line of ASCII."""
    name = ""
    try:
        if sys.platform == "darwin":
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=_TIMEOUT_S, check=False).stdout
        elif sys.platform.startswith("linux"):
            lines = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines()
            name = next((line.partition(":")[2] for line in lines if line.startswith("model name")), "")
    except (OSError, subprocess.SubprocessError):
        pass
    name = " ".join(name.split()) or platform.processor() or platform.machine()
    return name.encode("ascii", "replace").decode("ascii")


def freeze_asked() -> bool:
    """Whether this run was asked to write the frozen files: only when `OUTLINE_TRACKER_FREEZE` is `1`."""
    return os.environ.get(SWITCH) == "1"


def write_frozen(path, data: bytes) -> bool:
    """Write `data` to `path` as a frozen file, with the folders above it, and return True; when
    `freeze_asked()` is false, write nothing and return False. `data` has LF line ends and ends with LF.
    Raises, and writes nothing, when the package differs from the last commit or `data` has other line
    ends."""
    if not freeze_asked():
        return False
    changed = _package_status()
    if changed:
        raise RuntimeError(
            "Nothing was frozen: outline_tracker/ differs from the last commit, so the commit in the header "
            f"would not be the code that made the numbers. Commit or undo these changes first:\n{changed}")
    if b"\r" in data or not data.endswith(b"\n"):
        raise ValueError(f"Nothing was frozen: {Path(path).name} must have LF line ends and end with LF.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def frozen_text(header: list[str], rows: list[str]) -> bytes:
    """The bytes of a frozen text file: each line of `header` after `# `, then the rows, every line
    ended by LF. ASCII."""
    return "".join(f"{line}\n" for line in [*(f"# {line}" for line in header), *rows]).encode("ascii")


def read_frozen(path) -> tuple[dict[str, str], list[str]]:
    """A frozen text file as (header, rows): the lines `# key: value` at its start as a dict, in the
    file's order, and the lines after them as they are, without their line ends. A file without such
    lines has an empty header."""
    lines = Path(path).read_bytes().decode("ascii").splitlines()
    header = {}
    for count, line in enumerate(lines):
        key, colon, value = line[2:].partition(": ")
        if not line.startswith("# ") or not colon:
            return header, lines[count:]
        header[key] = value
    return header, []


def frozen_files(folder) -> list[Path]:
    """Every file below `folder`, in the order of its path relative to the folder, without what an
    operating system leaves there (`LEFT_BY_A_SYSTEM`)."""
    folder = Path(folder)
    files = [path for path in folder.rglob("*") if path.is_file() and path.name not in LEFT_BY_A_SYSTEM]
    return sorted(files, key=lambda path: path.relative_to(folder).as_posix())


def write_listing(folder, command: str, extra: dict[str, str] | None = None) -> bool:
    """Write `<folder>/HEADER.txt` for the frozen files below `folder` that have no header inside: the
    header lines (`header_lines`, with a line that says what a row holds), then one row per file:
    its path relative to `folder` with `/`, its size in bytes and its SHA-256, with commas between.
    Only when `freeze_asked()`, and refused, like `write_frozen`. Returns whether it was written."""
    if not freeze_asked():
        return False
    folder = Path(folder)
    rows = [f"{path.relative_to(folder).as_posix()},{path.stat().st_size},{_sha256(path)}"
            for path in frozen_files(folder) if path.name != SIDECAR]
    described = {"each row": "path relative to this file, size in bytes, SHA-256", **(extra or {})}
    return write_frozen(folder / SIDECAR, frozen_text(header_lines(command, described), rows))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_status() -> str:
    """What `git status --porcelain -- outline_tracker` prints: one line per file of the package that
    differs from the last commit, nothing when there is none."""
    return _git("status", "--porcelain", "--", "outline_tracker")


def _git(*args: str) -> str:
    """What a git command prints for this repository, without the line end at its end. Raises when git
    is missing or fails: without git no header can name its commit. Git's own variables (set while a
    git hook runs) are not passed on, as in `provenance`."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    try:
        done = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=_TIMEOUT_S, check=False, env=env)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"git {args[0]} could not be run: {error}") from error
    if done.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {done.stderr.strip()}")
    return done.stdout.rstrip()


# ---------------------------------------------------------------------------------------------
# The golden Tracker-format files


def same_decoder(header: dict[str, str], here: str) -> bool:
    """The decoder rule: whether `here` (`video.decoder_tag()` on this computer) is the decoder that
    made the frozen files with this header.

    The golden Tracker-format files hold positions that were found in a clip which OpenCV encodes and
    decodes on the computer that runs the test, and another build of OpenCV gives slightly other pixels
    on some frames. Measured on 2026-10-08 with files frozen on macOS, arm64: the test computers with
    Linux and Windows wrote the same digits as each other, and in the three files that the log shows,
    2 to 5 of their 25 to 31 rows differ from the frozen ones, by about 0.05 px. Decision X8 of the
    first build says the same of frame hashes (docs/PLAN.md). The owner's rule for this outcome
    (2026-10-07): "Bytes asserted on the system that froze them, the 0.25 px truth criterion elsewhere."

    So a test asserts the golden bytes only where this is true: the header has a `decoder` line, and
    `here` is that line, character for character: OpenCV's version, the system and the architecture.
    Everything else is another decoder: there the test asserts what holds with every decoder
    (`compare_with_golden`).
    """
    return bool(here) and header.get("decoder") == here


def compare_with_golden(what: str, written, golden, true_px, to_mm) -> None:
    """Judge a Tracker-format file that a run wrote against its golden file, where the decoder is not
    the one that froze the golden bytes (`same_decoder`): a position may then differ in its last
    digits, and this asserts what holds with every decoder.

    `written`: the run's file, which has this system's line ends. `golden`: the golden file (LF line
    ends). `true_px(frame)`: (pixelx, pixely) of the object's true center on a video frame, px in
    Tracker's convention (pixel centers at +0.5), or None where the clip does not show the object.
    `to_mm(pixelx, pixely)`: the calibration that the case's export defines, as (x, y) in mm with y up.
    `what` names the file in the messages.

    Asserted:
    - the text that does not come from pixels is the golden file's: the line ends are this system's,
      the two lines above the rows, the number of rows, and `t` and `frame` of every row; a row is
      lost (its four position cells are empty) where the golden row is, and nowhere else;
    - every number of a found row has as many decimals as in the golden file;
    - `pixelx`, `pixely` are at most `STAND_IN_PX` from the true center on that frame;
    - `x`, `y` are what `to_mm` gives for the row's own `pixelx`, `pixely`, as far as the rounding of
      the cells allows: a written number stands for every value within half a step of its last
      decimal. So the mm columns need no limit of their own.
    """
    lines = Path(written).read_bytes().decode("ascii").split(os.linesep)
    assert lines[-1] == "" and not any("\r" in line or "\n" in line for line in lines), (
        f"{what}: the line ends are not this system's, or the last line has none")
    rows = [line.split(",") for line in lines[:-1]]
    golden_rows = [line.split(",") for line in Path(golden).read_bytes().decode("ascii").splitlines()]
    assert rows[:2] == golden_rows[:2], f"{what}: the two lines above the rows are {lines[:2]}"
    assert len(rows) == len(golden_rows), f"{what}: {len(rows) - 2} rows, the golden file has {len(golden_rows) - 2}"
    for cells, golden_cells in zip(rows[2:], golden_rows[2:], strict=True):
        where = f"{what}, frame {golden_cells[1]}"
        assert cells[:2] == golden_cells[:2], (
            f"{where}: t and frame are {cells[:2]}, the golden file has {golden_cells[:2]}")
        lost, golden_lost = (positions[2:] == [""] * 4 for positions in (cells, golden_cells))
        assert lost == golden_lost, (
            f"{where}: {'lost' if lost else 'found'} here, {'lost' if golden_lost else 'found'} in the golden file")
        if lost:
            continue
        decimals = [_decimals(cell) for cell in cells]
        assert None not in decimals and decimals == [_decimals(cell) for cell in golden_cells], (
            f"{where}: other decimals than the golden row: {cells}, the golden file has {golden_cells}")
        x_mm, y_mm, u_px, v_px = (float(cell) for cell in cells[2:])
        true = true_px(int(cells[1]))
        assert true is not None, f"{where}: found, but the clip does not show the object on this frame"
        distance = math.hypot(u_px - true[0], v_px - true[1])
        assert distance <= STAND_IN_PX, f"{where}: {distance:.3f} px from the true center (limit {STAND_IN_PX} px)"
        # The two pixel cells stand for a small square of positions. A calibration is a scale, a
        # rotation and a shift, so over that square its smallest and its largest x, and y, are at the
        # four corners.
        half_u, half_v = (_half_step(count) for count in decimals[4:])
        corners = [to_mm(u_px + i * half_u, v_px + j * half_v) for i in (-1, 1) for j in (-1, 1)]
        xs_mm, ys_mm = zip(*corners, strict=True)
        for name, value, count, possible in (("x", x_mm, decimals[2], xs_mm), ("y", y_mm, decimals[3], ys_mm)):
            slack = _half_step(count) + _FLOAT_MM
            assert min(possible) - slack <= value <= max(possible) + slack, (
                f"{where}: {name} = {value} mm is not what the calibration gives for pixelx, pixely = {u_px}, {v_px}")


def _decimals(cell: str) -> int | None:
    """How many digits a written number has after its point (0 for a whole number); None when the cell
    is not a plain decimal number: empty, with an exponent, `nan`."""
    number = re.fullmatch(r"-?\d+(?:\.(\d+))?", cell)
    return len(number.group(1) or "") if number else None


def _half_step(decimals: int) -> float:
    """Half a step of the last decimal of a number that is written with `decimals` decimals."""
    return 0.5 * 10.0 ** -decimals


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
