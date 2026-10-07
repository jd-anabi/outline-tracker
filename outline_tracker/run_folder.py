"""Where a run's files go (SPEC 8.1): the default run folder, and the check that keeps the tool out of a
folder of Tracker files.

One run folder per video and student, by default `<video folder>/<video stem>_outline_<student>/`, so
students who share a video folder never overwrite each other. Students keep their Tracker exports as .csv
and .txt files directly in a folder, and their notebooks read every such file as a track: this tool's
tables must never land among them.

No units or coordinates here: paths and names only. No Qt, no torch.
"""

from __future__ import annotations

import re
from pathlib import Path

from outline_tracker import schema
from outline_tracker.fileio import new_name

TABLE_SUFFIXES = (".csv", ".txt")  # what a student's folder of Tracker files holds
# not allowed in a file name on Windows (< > : " / \ | ? * and the control characters), plus all whitespace
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]')
_TEMPORARY = re.compile(r"-[0-9a-f]{8}-tmp$")  # the end of the stem of a file `fileio.atomic_write` is writing


def default_run_folder(video_path, student: str) -> Path:
    """The default run folder for a video and a student: `<video folder>/<video stem>_outline_<student>/`.

    video_path: the video file (text or Path; it need not exist). Its stem is used as it is, `_tracker`
    included, and the result is relative when the video path is. student: the student's name as typed; it
    is made safe for a folder name on Windows and macOS: whitespace around it is dropped, then every path
    separator, every character Windows forbids in a name (< > : " / \\ | ? * and control characters), every
    whitespace character and every dot at the end become `_`. Letters of any alphabet are kept.

    Nothing is created or looked up on disk. No units or coordinates. Raises ValueError for an empty name.
    """
    name = _UNSAFE.sub("_", str(student).strip())
    name = re.sub(r"\.+$", lambda dots: "_" * len(dots.group()), name)  # Windows drops dots at a name's end
    if not name:
        raise ValueError("A name is needed for the run folder (it becomes <video stem>_outline_<name>): "
                         "an empty name was given.")
    video = Path(video_path)
    return video.parent / f"{video.stem}_outline_{name}"


def folder_has_foreign_tables(path) -> bool:
    """Whether a folder directly holds .csv or .txt files that are not this tool's own.

    True means the folder looks like a student's folder of Tracker files and must not be written into.
    path: the folder (text or Path); a folder that does not exist, or a path that is not a folder, gives
    False. Only files directly in the folder count, not those in subfolders, and the suffix is compared
    without regard to case. This tool's own files are those of `schema.FILES` under exactly those names,
    and what `fileio.atomic_write` leaves next to them: `<stem>.new<suffix>` (the target was locked) and
    `<stem>-<8 hex digits>-tmp<suffix>` (a write was cut short). Hidden files (a name starting with a
    dot, such as the `._name` files macOS adds on some drives) are passed over. No units or coordinates.
    """
    folder = Path(path)
    if not folder.is_dir():
        return False
    own = {spec.name for spec in schema.FILES}
    own |= {new_name(name).name for name in own}
    for entry in folder.iterdir():
        name = entry.name
        if name.startswith(".") or entry.suffix.lower() not in TABLE_SUFFIXES or not entry.is_file():
            continue
        if name not in own and _TEMPORARY.sub("", entry.stem) + entry.suffix not in own:
            return True
    return False
