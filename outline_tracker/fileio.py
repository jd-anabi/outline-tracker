"""Atomic file writes with the Windows lock retry, the hash that identifies a video, and the relative
paths that let a run folder be moved (SPEC 8.1).

Every output file of a run folder is written through `atomic_write`: the data go into a temporary
file in the same folder, and only a finished file is renamed onto the target. A reader (a student's
notebook, a cloud-sync client, the GUI) therefore sees the old file or the new one, never half of
one. On Windows a rename fails while another program holds the target open (a CSV open in a
spreadsheet program, a file being synced): the rename is retried for about 5 s, and if the target
stays locked the data are kept next to it as `<stem>.new<suffix>`.

No units or coordinates here: paths and bytes only. No Qt, no torch.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import secrets
import time
from collections.abc import Callable
from pathlib import Path

RETRY_DELAYS_S = (0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.5)  # waits between tries at a locked target: 5 s in all
HASH_BYTES = 64 * 1024 * 1024  # a video is identified by the SHA-256 of its first 64 MiB (SPEC 8.1)
_READ_BYTES = 1024 * 1024


def new_name(path) -> Path:
    """Where the data go when `path` stays locked: `<stem>.new<suffix>` in the same folder
    (positions.csv -> positions.new.csv)."""
    path = Path(path)
    return path.with_name(f"{path.stem}.new{path.suffix}")


def atomic_write(path, write_fn: Callable[[Path], object]) -> Path:
    """Write a file so that `path` always holds a complete file: the old one or the new one.

    path: the file to write (its folder is created if needed). write_fn: called once with the path
    of a temporary file in the same folder, whose name ends with `path`'s suffix (so a writer that
    picks its format from the name, such as ffmpeg for ".mp4" or numpy for ".npz", behaves as it
    would on `path`). It must create that file, write everything, and close it before it returns.

    Then the temporary file is renamed onto `path` (`os.replace`). If that raises PermissionError
    (on Windows: another program holds `path` open), it is tried again for about 5 s. If `path`
    stays locked, the data are kept as `<stem>.new<suffix>` next to it and `path` is left as it was.

    Returns the path that now holds the data: `path`, or the `.new` path, which the caller should
    report to the user. If write_fn raises, or the rename fails for another reason, the error is
    passed on, `path` is untouched and the temporary file is removed. PermissionError is raised
    when the `.new` file is locked as well.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{secrets.token_hex(4)}.tmp{path.suffix}")
    try:
        write_fn(tmp)
        for delay in (*RETRY_DELAYS_S, None):
            try:
                os.replace(tmp, path)
                return path
            except PermissionError:
                if delay is None:
                    break
                time.sleep(delay)
        fallback = new_name(path)
        try:
            os.replace(tmp, fallback)
        except PermissionError:
            raise PermissionError(
                f"Could not write {path.name}: it is open in another program, and so is {fallback.name}. "
                f"Close them (folder: {path.parent}) and try again."
            ) from None
        return fallback
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def relative_path(target, folder, pathmod=os.path) -> str | None:
    """A file's path as session.json stores it (SPEC 8.1): relative to `folder` (the run folder), with
    forward slashes on every platform. Both arguments are absolute paths (text or Path). Returns None
    when there is no relative path: on Windows, a file on another drive or on a network share.

    pathmod is the path module to use (the running system's by default; `ntpath` in tests).
    """
    try:
        relative = pathmod.relpath(os.fspath(target), os.fspath(folder))
    except ValueError:
        return None
    return relative.replace(pathmod.sep, "/")


def sha256_first_64mib(path) -> str:
    """SHA-256 of the first 64 MiB (67,108,864 bytes) of a file, or of the whole file when it is
    shorter, as 64 lowercase hex digits. With the size in bytes it identifies a video (SPEC 8.1)
    without reading all of a long one."""
    digest = hashlib.sha256()
    left = HASH_BYTES
    with open(path, "rb") as f:
        while left > 0:
            block = f.read(min(_READ_BYTES, left))
            if not block:
                break
            digest.update(block)
            left -= len(block)
    return digest.hexdigest()
