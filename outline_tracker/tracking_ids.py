"""The names and colors of tracks (SPEC 3.4; decision X14).

A track id is capital letters, optionally followed by a piece number: A, B, ..., Z, AA, AB, ... for
new objects, and A2, A3, ... for the later pieces of the animal A (regex ^[A-Z]+[0-9]*$). Pieces of
one animal share the letters; this is the analysis template's convention.

An id is never given twice in a session: letters and piece numbers count as used by the session's
tracks and also by the tracks its runs and corrections name, so that the record of an object that
was removed never reads as the record of a new one.

No units, no coordinates. No Qt, no torch.
"""

from __future__ import annotations

import re
from itertools import count, product
from string import ascii_uppercase

from outline_tracker.session import Session

# Last week's overlay colors (X14), as RGB hex, in the order of the ids: A is yellow.
TRACK_COLORS = ("#FFFF00", "#FF00FF", "#00FF00", "#0080FF", "#FF8000", "#00FFFF", "#FF0080", "#FF0000", "#0000FF",
                "#80FF80")
_TRACK_ID = re.compile(r"([A-Z]+)([0-9]*)")  # the letters, then the piece number


def _remembered(session: Session) -> list[re.Match]:
    """Every id of this form that the session holds: its tracks', and those its runs and
    corrections name. Other names (a Tracker export's, kept by from-tracker) cannot clash."""
    ids = ([track.id for track in session.tracks] + [name for run in session.runs for name in run.tracks]
           + [name for correction in session.corrections for name in correction.tracks])
    return [match for match in map(_TRACK_ID.fullmatch, ids) if match]


def next_track_id(session: Session) -> str:
    """The id of the next new object: the first of A, B, ..., Z, AA, AB, ... whose letters the
    session does not use yet, for a track, a piece (B2 uses B) or a track it remembers."""
    used = {match.group(1) for match in _remembered(session)}
    for length in count(1):
        for letters in map("".join, product(ascii_uppercase, repeat=length)):
            if letters not in used:
                return letters


def next_piece_id(session: Session, parent_id: str) -> str:
    """The id of the next piece of the animal of track `parent_id`: its letters and the first
    free number from 2 on (A2, then A3; the same for the parent A or A2). Raises ValueError for a
    parent whose id is not capital letters with an optional number."""
    parent = _TRACK_ID.fullmatch(parent_id)
    if parent is None:
        raise ValueError(f"Track {parent_id} cannot be continued as a new track: pieces are named after a track "
                         "with a capital-letter name (A, A2, ...).")
    letters = parent.group(1)
    taken = {int(match.group(2) or 1) for match in _remembered(session) if match.group(1) == letters}
    return f"{letters}{next(number for number in count(2) if number not in taken)}"


def track_color(track_id: str) -> str:
    """The color of a new object, RGB hex such as "#FFFF00": `TRACK_COLORS` in the order of the
    ids (A, B, ..., Z, AA, ...), ten colors and then again. `track_id` is letters only."""
    number = 0
    for letter in track_id:
        number = 26 * number + ord(letter) - ord("A") + 1
    return TRACK_COLORS[(number - 1) % len(TRACK_COLORS)]
