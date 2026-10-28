"""Reading a session.json of version 1, the format of 0.1.0: its parsed object turned into one of version 2.

docs/design/formats.md, section "session.json, version 2", is the contract: its table says what happens to
each key, and its examples are the expected values of tests/core/test_session_v1.py. `upgrade` only reads:
it returns a new object that shares nothing with the one it was given, and it writes no file. Keeping the
file of version 1 when a session is saved over it is the work of `Session.save` (rule 5 of docs/ROADMAP.md).
"""

from __future__ import annotations

import copy
from typing import Any

# 0.1.0's values of the two keys whose default for new sessions W3 changes (decisions 9 and 13). A file of
# version 1 that lacks the key is read with this value.
V1_JUMP_MM_S = 100.0
V1_PROBE_NAME = "LED1"

# The keys that version 2 gives an object of version 1 (decision 39): the path of the object ("" for the
# top of the file), and its keys. A file of version 1 that holds one cannot say which value counts.
_CLASHES = {"": ("label", "boundary"), "circle": ("shape", "diameter_mm"),
            "processing": ("boundary_crop", "tracker_format")}


class SessionKeyError(ValueError):
    """A session file holds a key whose value this build refuses (formats.md, "Reading a file", item 6).
    `key` is the key's path in the file, e.g. "boundary.shape"; the message names the file and the key."""

    def __init__(self, key: str, message: str):
        self.key = key
        super().__init__(message)


class MixedVersionError(SessionKeyError):
    """A session file of version 1 holds a key of version 2 (decision 39)."""

    def __init__(self, key: str, where: str = "The session file"):
        super().__init__(key, f'{where} has session format version 1 but also holds "{key}", a key of version 2. '
                              "Version 0.1.0 never writes that key, so this outline-tracker cannot tell which "
                              "value counts. Take the key out of the file, or open the file with the "
                              "outline-tracker that wrote it.")


def upgrade(data: dict[str, Any], where: str = "The session file") -> dict[str, Any]:
    """The parsed session.json `data` of version 1 as a parsed session.json of version 2, as the table of
    docs/design/formats.md says; `where` names the file in messages. Raises MixedVersionError when the file
    holds a key of version 2, and ValueError when "circle" is neither an object nor null. Another block of
    the wrong kind passes unchanged, and the records refuse it."""
    data = copy.deepcopy(data)
    for parent, keys in _CLASHES.items():
        block = data.get(parent) if parent else data
        for key in keys:
            if isinstance(block, dict) and key in block:
                raise MixedVersionError(f"{parent}.{key}" if parent else key, where)
    out = {}
    for key, value in data.items():
        if key == "schema_version":
            value = 2
        elif key == "student":
            key = "label"
        elif key == "circle":
            key, value = "boundary", _boundary(value)
        elif key == "time" and isinstance(value, dict):
            value = _time(value)
        elif key == "processing" and isinstance(value, dict):
            value = _processing(value)
        elif key == "probes" and isinstance(value, list):
            value = [_probe(probe) for probe in value]
        out[key] = value
    out.setdefault("processing", _processing({}))
    return out


def _renamed(block: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    """`block` with its key `old` called `new`, in the same place."""
    return {new if key == old else key: value for key, value in block.items()}


def _boundary(circle):
    """The block "circle" as the block "boundary": null stays null; an object gets its shape first."""
    if circle is None:
        return None
    if not isinstance(circle, dict):  # the words of the records (session_parts._Record.from_json)
        raise ValueError(f'In the session file, "circle" must be an object {{...}}, not {circle!r}.')
    return {"shape": "circle", **_renamed(circle, "dish_mm", "diameter_mm")}


def _time(time: dict[str, Any]) -> dict[str, Any]:
    """The block "time" without the manifest's path; a rate read from a manifest counts as typed."""
    time = {key: value for key, value in time.items() if key != "manifest_path"}
    if time.get("source") == "manifest":
        time["source"] = "typed"
    return time


def _processing(processing: dict[str, Any]) -> dict[str, Any]:
    """The block "processing" of version 2: 0.1.0's jump limit when none is given, and the Tracker-format
    files on, because 0.1.0 always wrote them (decisions 7, 13 and 34)."""
    processing = _renamed(processing, "dish_crop", "boundary_crop")
    processing.setdefault("jump_mm_s", V1_JUMP_MM_S)
    processing["tracker_format"] = True
    return processing


def _probe(probe):
    """An item of "probes", named as 0.1.0 named its one probe when it has no name."""
    if isinstance(probe, dict) and "name" not in probe:
        return {"name": V1_PROBE_NAME, **probe}
    return probe
