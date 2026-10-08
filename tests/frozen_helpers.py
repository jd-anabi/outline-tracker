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

This is a helper of the tests, not a part of the package. It imports no torch. No units or
coordinates here: text and bytes only.
"""

import hashlib
import os
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

from outline_tracker import provenance, video

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tests" / "data"
SWITCH = "OUTLINE_TRACKER_FREEZE"  # the environment variable; `1` asks for the frozen files to be written
SIDECAR = "HEADER.txt"  # the header and the list of the files that have no header inside
LIBRARIES = ("torch", "torchvision", "transformers", "timm", "safetensors", "numpy", "opencv-python-headless",
             "scikit-image")  # named in a header, as far as installed
LEFT_BY_A_SYSTEM = (".DS_Store", "Thumbs.db", "desktop.ini")  # not frozen files (.gitignore has the same names)
_TIMEOUT_S = 30.0


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
