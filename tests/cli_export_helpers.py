"""Shared by the tests of the `export` command (tests/test_cli_export.py).

The run folders are the ones of tests/export_helpers.py: tracked by `tracking.run_job` with `ExactFake` on
the synthetic dish clip, with a calibration stick of 300 px = 30 mm (K = 0.1 mm per px). Expected numbers
come from the scene's true masks and that stick, never from the command's own output. Sessions are edited
as a person would edit them, as JSON text.

Sizes are printed in SI units (1 kB = 1000 bytes), one decimal from kB up: a printed size stands for a real
size within half of its last digit.
"""

import json
import os
import re

from outline_tracker import cli, fileio

WROTE = re.compile(r"^wrote (?P<name>.+): (?P<number>\d+(?:\.\d)?) (?P<unit>B|kB|MB|GB)$")
UNIT_BYTES = {"B": 1, "kB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9}


def export(*args) -> int:
    """Run `outline-tracker export ARGS...` in this process and return its exit code."""
    return cli.main(["export", *(str(arg) for arg in args)])


def listing(folder) -> list[str]:
    """Every file and folder under `folder`, as sorted relative paths with forward slashes."""
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*"))


def edit_session(run_folder, change) -> None:
    """Open session.json of a run folder as JSON, let `change(data)` edit it, and write it back."""
    path = run_folder / "session.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def written(out: str) -> dict[str, tuple[float, str]]:
    """The `wrote NAME: SIZE` lines of the command's output: name -> (printed number, unit)."""
    found = {}
    for line in out.splitlines():
        match = WROTE.match(line)
        if match:
            assert match["name"] not in found, f"{match['name']} is listed twice"
            found[match["name"]] = (float(match["number"]), match["unit"])
    return found


def within_rounding(shown: tuple[float, str], size: int) -> bool:
    """Whether a printed size (number, unit) is `size` bytes rounded to its last digit."""
    number, unit = shown
    half_digit = 0.5 if unit == "B" else 0.05
    return abs(number - size / UNIT_BYTES[unit]) <= half_digit + 1e-9


def lock(monkeypatch, *names) -> None:
    """Make every rename onto a file of one of these names fail as Windows does while another program
    holds the file open, and do not wait between the tries."""
    real = os.replace

    def replace(src, dst, **kwargs):
        if os.path.basename(dst) in names:
            raise PermissionError(13, "The process cannot access the file", str(dst))
        return real(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(fileio, "RETRY_DELAYS_S", ())
