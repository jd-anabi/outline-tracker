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

This is a helper of the tests, not a part of the package. It imports no torch. Apart from that last
part there are no units or coordinates here: text and bytes only.
"""

import hashlib
import math
import os
import platform
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
