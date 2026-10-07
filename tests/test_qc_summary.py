"""Tests for outline_tracker.qc.summary_lines: the QC summary of run.log (SPEC 8.9): per track
and flag the number of frames, the first frame and its time, in the wording of last week's CHECK
lines ("A: lost in 2 of 5 frames (first at t = 0.250 s)").

Times are frame / fps_true in s (240 frames per s here); frames are video frame numbers.
"""

import pytest
import qc_helpers as q
from qc_helpers import CENTER, FULL_HD

from outline_tracker import qc

FRAMES = [0, 60, 120, 180, 240]   # 0, 0.25, 0.5, 0.75 and 1 s


def derived_tracks(*names):
    """Derived tracks on FRAMES with these ids; only their frames and times matter here."""
    return q.tracks({name: [q.ellipse(frame) for frame in FRAMES] for name in names})[0]


def test_a_line_per_track_and_flag_gives_the_count_the_first_frame_and_its_time():
    flags = {"A": ["", "LOST", "JUMP;SIZE", "LOST", "SIZE"]}
    assert qc.summary_lines(flags, derived_tracks("A")) == [
        "A: LOST in 2 of 5 frames (first at frame 60, t = 0.250 s)",
        "A: JUMP in 1 of 5 frames (first at frame 120, t = 0.500 s)",
        "A: SIZE in 2 of 5 frames (first at frame 120, t = 0.500 s)",
    ]


def test_tracks_come_in_the_order_given_and_flags_in_the_order_of_spec_9():
    every = "LOST;JUMP;SIZE;CONTACT;EDGE;MULTI;LOWRES;ORIENT;HEADGUESS"
    flags = {"B": ["HEADGUESS", "HEADGUESS", "HEADGUESS", "HEADGUESS", every], "A2": [""] * 5,
             "A": ["LOWRES"] * 5}
    lines = qc.summary_lines(flags, derived_tracks("A", "A2", "B"))
    assert lines == [
        "B: LOST in 1 of 5 frames (first at frame 240, t = 1.000 s)",
        "B: JUMP in 1 of 5 frames (first at frame 240, t = 1.000 s)",
        "B: SIZE in 1 of 5 frames (first at frame 240, t = 1.000 s)",
        "B: CONTACT in 1 of 5 frames (first at frame 240, t = 1.000 s)",
        "B: EDGE in 1 of 5 frames (first at frame 240, t = 1.000 s)",
        "B: MULTI in 1 of 5 frames (first at frame 240, t = 1.000 s)",
        "B: LOWRES in 1 of 5 frames (first at frame 240, t = 1.000 s)",
        "B: ORIENT in 1 of 5 frames (first at frame 240, t = 1.000 s)",
        "B: HEADGUESS in 5 of 5 frames (first at frame 0, t = 0.000 s)",
        "A2: no flags in 5 frames",
        "A: LOWRES in 5 of 5 frames (first at frame 0, t = 0.000 s)",
    ]
    assert all(line.isascii() and "\n" not in line for line in lines)


def test_the_summary_of_no_tracks_is_empty():
    assert qc.summary_lines({}, {}) == []


def test_the_summary_of_computed_flags_counts_what_the_cells_say():
    # A shrimp at dish scale without a head click, lost on frame 120, next to a second one on
    # frames 180 and 240 (5 px of water; two grid cells of the full frame are 15 px).
    def small(frame, dv=0.0):
        return q.ellipse(frame, (CENTER[0], CENTER[1] + dv), 7.0, 3.0, angle=0.0, box=FULL_HD)

    records = {"A": [small(0), small(60), q.lost(120, FULL_HD), small(180), small(240)],
               "B": [small(frame, 200.0 if frame < 180 else 11.0) for frame in FRAMES]}
    derived, arrays, processing = q.tracks(records, {"B": q.head((CENTER[0], CENTER[1] + 200.0), 0.0)})
    flags = qc.compute_flags(derived, arrays, q.WORLD, processing)
    assert qc.summary_lines(flags, derived) == [
        "A: LOST in 1 of 5 frames (first at frame 120, t = 0.500 s)",
        "A: CONTACT in 2 of 5 frames (first at frame 180, t = 0.750 s)",
        "A: LOWRES in 4 of 5 frames (first at frame 0, t = 0.000 s)",
        "A: HEADGUESS in 5 of 5 frames (first at frame 0, t = 0.000 s)",
        "B: CONTACT in 2 of 5 frames (first at frame 180, t = 0.750 s)",
        "B: LOWRES in 5 of 5 frames (first at frame 0, t = 0.000 s)",
    ]


def test_cells_that_do_not_fit_the_track_are_refused():
    derived = derived_tracks("A")
    with pytest.raises(ValueError, match="5"):
        qc.summary_lines({"A": ["", "LOST"]}, derived)
    with pytest.raises(ValueError, match=r"\bB\b"):
        qc.summary_lines({"B": [""] * 5}, derived)
