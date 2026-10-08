"""Tests for outline_tracker.qc: the nine quality flags of SPEC 9, each on exactly the frames
where it belongs (SPEC 13.1 Flags), the rules taken over from last week's `_flags` (JUMP, SIZE)
and the text of a flags cell.

Tracks are analytic shapes or rectangles of pixels sent through measure_mask and derive_track
(tests/qc_helpers.py). The expected frames follow from the shapes: sizes and places in image px,
speeds in mm per s in the world frame (0.0324 mm per px, 240 frames per s unless a test says
otherwise), areas in px. A list of cells is one text per row of the track, in frame order.
"""

import analytic_shapes as shapes
import derive_helpers as h
import numpy as np
import pytest
import qc_helpers as q
from qc_helpers import CENTER, CLOSE, FULL_HD, TILT, WORLD

from outline_tracker import qc
from outline_tracker.geometry import WorldFrame
from outline_tracker.measure import measure_mask

USUAL = "LOWRES;ORIENT;HEADGUESS"   # a small square of pixels in the full frame, no head click


def test_a_track_with_nothing_wrong_has_an_empty_cell_on_every_row():
    records = [q.ellipse(2 * i, (CENTER[0] + 3.0 * i, CENTER[1] - 2.0 * i)) for i in range(6)]
    flags = q.flags_of({"A": records}, {"A": q.head()})
    assert flags == {"A": [""] * 6}
    assert all(type(cell) is str for cell in flags["A"])


def test_no_tracks_give_no_flags():
    assert q.flags_of({}) == {}


# --------------------------------------------------------------------------- LOST, HEADGUESS, LOWRES


def test_lost_marks_the_frames_with_an_empty_mask():
    records = [q.ellipse(0), q.lost(2), q.ellipse(4), q.lost(6), q.lost(8), q.ellipse(10)]
    assert q.flags_of({"A": records}, {"A": q.head()}) == {"A": ["", "LOST", "", "LOST", "LOST", ""]}


def test_a_lost_row_carries_lost_and_headguess_and_nothing_else():
    # A shrimp at dish scale, 14 px long, no head click: LOWRES and HEADGUESS on every frame that
    # has a mask. A lost row has shape_ok 0 and area 0 too, and is neither LOWRES nor SIZE.
    def small(frame):
        return q.ellipse(frame, (CENTER[0] + 0.75 * frame, CENTER[1]), 7.0, 3.0, box=FULL_HD)

    records = [small(0), small(2), q.lost(4, FULL_HD), small(6), q.lost(8, FULL_HD)]
    derived, arrays, processing = q.tracks({"A": records})
    assert derived["A"].shape_ok.tolist() == [0, 0, 0, 0, 0]
    assert qc.compute_flags(derived, arrays, WORLD, processing) == {
        "A": ["LOWRES;HEADGUESS", "LOWRES;HEADGUESS", "LOST;HEADGUESS", "LOWRES;HEADGUESS", "LOST;HEADGUESS"]}


def test_headguess_is_on_every_frame_of_a_track_without_a_head_click():
    up, down = (CENTER[0], CENTER[1] - 60.0), (CENTER[0], CENTER[1] + 60.0)   # 83 px of water between them
    clicked = [q.ellipse(2 * i, up) for i in range(4)]
    guessed = [q.ellipse(0, down), q.lost(2), q.ellipse(4, down), q.ellipse(6, down)]
    flags = q.flags_of({"A": clicked, "B": guessed}, {"A": q.head(up)})
    assert flags == {"A": [""] * 4, "B": ["HEADGUESS", "LOST;HEADGUESS", "HEADGUESS", "HEADGUESS"]}


@pytest.mark.parametrize("a, b, box, settings, cell", [
    (7.0, 3.0, q.window(CENTER, 96), {}, "LOWRES"),      # 14 px long: too few camera pixels, 37 grid cells
    (40.0, 15.0, FULL_HD, {}, "LOWRES"),                 # 80 px long, but 10.7 grid cells of 7.5 px
    (40.0, 15.0, CLOSE, {}, ""),                         # 80 px and 80 grid cells
    (40.0, 15.0, FULL_HD, {"shape_ok_min": 10}, ""),     # the setting moves the limit
], ids=["14px-in-96px-window", "80px-full-frame", "80px-close", "limit-10"])
def test_lowres_is_on_the_frames_whose_shape_is_not_ok(a, b, box, settings, cell):
    records = [q.ellipse(2 * i, CENTER, a, b, box=box) for i in range(3)]
    assert q.flags_of({"A": records}, {"A": q.head()}, **settings) == {"A": [cell] * 3}


def test_two_stray_pixels_do_not_switch_lowres_off():
    # A body of 6 rows x 15 columns of pixels with grid cells of 3.9 px, no head click: 17 px and 4.4
    # cells along its major axis, so LOWRES and HEADGUESS on every row. On the second row the mask
    # also holds 2 pixels 240 columns away (h.stray_pixel_records). They are 2 / 90 = 2.2% of the
    # body, under the 10% of MULTI, and the size check is of the body, the largest piece: still
    # LOWRES. ORIENT is right there: the core was stored at tracking time with an opening radius
    # from the whole mask, 146 px long (radius 15 px against a body 6 px wide), so it fell back.
    records = h.stray_pixel_records()
    assert [one.n_components for one in records] == [1, 2, 1]
    assert [one.second_fraction for one in records] == pytest.approx([0.0, 2 / 90, 0.0])
    assert [(one.core_r_px, one.core_fallback) for one in records] == [(2, False), (15, True), (2, False)]
    assert q.flags_of({"A": records}) == {"A": ["LOWRES;HEADGUESS", "LOWRES;ORIENT;HEADGUESS", "LOWRES;HEADGUESS"]}


# --------------------------------------------------------------------------- EDGE, MULTI, ORIENT


def test_edge_marks_the_frame_on_which_the_mask_touches_the_border_of_the_models_input():
    # The model sees CLOSE, whose first column is 572. The ellipse reaches 40 px to the left of
    # its center: at u = 605.37 the model's image cuts it, at 620.37 and beyond it is clear.
    def seen(frame, u):
        cols, rows = shapes.pixel_centers(CLOSE[3], CLOSE[2], CLOSE[:2])
        inside = shapes.ellipse(cols, rows, (u, CENTER[1]), 40.0, 15.0, 0.0)
        return measure_mask(shapes.to_result(inside, origin=CLOSE[:2]), frame, CLOSE, "fine")

    places = [640.37, 630.37, 620.37, 605.37, 620.37, 630.37]
    records = [seen(2 * i, u) for i, u in enumerate(places)]
    flags = q.flags_of({"A": records}, {"A": q.head((places[0], CENTER[1]), 0.0)})
    assert flags == {"A": ["", "", "", "EDGE", "", ""]}


def test_multi_marks_a_second_piece_of_at_least_a_tenth_of_the_largest():
    # The ellipse has 1885 px. A disk 55 px below it: radius 10 is 314 px (17%), radius 5 is 79 px (4%).
    def with_piece(frame, radius):
        below = (CENTER[0], CENTER[1] + 55.0)
        return h.record(lambda u, v: shapes.union(shapes.ellipse(u, v, CENTER, 40.0, 15.0, TILT),
                                                  shapes.disk(u, v, below, radius)), frame, CENTER, half=80,
                        input_box=CLOSE)

    records = [q.ellipse(0), with_piece(2, 10.0), q.ellipse(4), with_piece(6, 5.0), q.ellipse(8)]
    assert [one.n_components for one in records] == [1, 2, 1, 2, 1]
    assert q.flags_of({"A": records}, {"A": q.head()}) == {"A": ["", "MULTI", "", "", ""]}


def test_a_mask_with_a_large_second_piece_is_multi_and_is_sized_by_its_largest_piece():
    # The ellipse of pi x 40 x 15 = 1885 px and, 55 px below it, a disk of radius 10 px: 314 px, 17%
    # of the ellipse, so MULTI. The size check is of the ellipse alone, whose L1 is 2a = 80 px (the
    # second moment of a uniform ellipse along its major axis is a^2 / 4); over both pieces
    # together the major axis would read 88 px.
    below = (CENTER[0], CENTER[1] + 55.0)
    both = h.record(lambda u, v: shapes.union(shapes.ellipse(u, v, CENTER, 40.0, 15.0, TILT),
                                              shapes.disk(u, v, below, 10.0)), 2, CENTER, half=80, input_box=CLOSE)
    assert both.n_components == 2 and both.second_fraction == pytest.approx(314 / 1885, abs=0.01)
    derived, arrays, processing = q.tracks({"A": [q.ellipse(0), both, q.ellipse(4)]}, {"A": q.head()})
    assert qc.compute_flags(derived, arrays, WORLD, processing) == {"A": ["", "MULTI", ""]}
    assert derived["A"].px_along_major == pytest.approx([80.0] * 3, rel=0.02)


def test_multi_starts_at_exactly_ten_percent():
    def two_blocks(frame, rows, cols):   # 100 px, and a second piece of rows x cols px apart from it
        return q.pixels(frame, shapes.blocks((40, 60), (0, 10, 0, 10), (20, 20 + rows, 20, 20 + cols)), (600, 300))

    records = [q.block(0, 10, 10), two_blocks(2, 2, 5), two_blocks(4, 3, 3), q.block(6, 10, 10)]
    assert [one.area_px for one in records] == [100, 110, 109, 100]
    assert q.rows_with("MULTI", q.flags_of({"A": records})["A"]) == [1]


def test_orient_marks_a_round_core_a_core_fallback_and_an_axis_jump():
    turned = TILT + np.radians(75.0)
    records = [q.ellipse(2 * i) for i in range(9)] + [q.ellipse(2 * i, angle=turned) for i in range(9, 12)]
    # frame 6: a disk of the ellipse's area; its core has no long axis (lambda2 / lambda1 = 1 > 0.8)
    records[3] = q.disk(6, CENTER, 24.5, box=CLOSE)
    # frame 12: 160 x 15 px. The opening disk has radius 16 px, the shape is 15 px wide: nothing is
    # left of it, so the core falls back to the whole mask.
    records[6] = q.ellipse(12, a=80.0, b=7.5)
    assert [one.core_fallback for one in records] == [False] * 6 + [True] + [False] * 5
    # frame 18: the axis has turned by 75 degrees since frame 16 and stays there (X17: one frame)
    assert q.flags_of({"A": records}, {"A": q.head()}) == {
        "A": ["", "", "", "ORIENT", "", "", "ORIENT", "", "", "ORIENT", "", ""]}


# --------------------------------------------------------------------------- JUMP (last week's rule)


def upright(frame, u):
    """The usual ellipse standing on end (30 px wide in u), its center at `u` px."""
    return q.ellipse(frame, (u, CENTER[1]), angle=np.pi / 2)


def test_jump_marks_the_later_frame_of_a_step_faster_than_100_mm_per_s():
    # 1/120 s between rows: 15 px are 58 mm/s, 35 px are 136 mm/s.
    places = [620.37, 635.37, 650.37, 685.37, 700.37, 715.37]
    records = [upright(2 * i, u) for i, u in enumerate(places)]
    flags = q.flags_of({"A": records}, {"A": q.head((places[0], CENTER[1]), np.pi / 2)})
    assert flags == {"A": ["", "", "", "JUMP", "", ""]}


def test_jump_divides_by_the_real_time_between_two_visible_frames():
    click = {"A": q.head((620.37, CENTER[1]), np.pi / 2)}
    # 30 px across a lost frame take 1/60 s: 58 mm/s. 60 px across the next one: 117 mm/s, and
    # the frame that gets the flag is the one where the track is seen again.
    records = [upright(0, 620.37), upright(2, 635.37), q.lost(4), upright(6, 665.37), upright(8, 680.37),
               q.lost(10), upright(12, 740.37)]
    assert q.flags_of({"A": records}, click) == {"A": ["", "", "LOST", "", "", "LOST", "JUMP"]}
    # a track whose frames are not evenly spaced: 60 px in 10 frames are 47 mm/s, 35 px in 2 are 136
    records = [upright(0, 620.37), upright(2, 635.37), upright(12, 695.37), upright(14, 730.37)]
    assert q.flags_of({"A": records}, click) == {"A": ["", "", "", "JUMP"]}


@pytest.mark.parametrize("limit, rows", [(100.0, [2]), (99.5, [1, 2, 3]), (101.7, [])])
def test_jump_needs_a_speed_strictly_above_the_limit(limit, rows):
    # 1 mm per px, axes along the image, 2 frames per s: a 4 x 4 block's center moves by whole
    # pixels, so the speeds are exact. (30, 40) px in 0.5 s is 100 mm/s, (30, 41) px is 101.6 mm/s.
    world = WorldFrame(1.0, 0.0, 0.0, 0.0)
    corners = [(100, 100), (130, 140), (160, 181), (190, 221)]
    records = [q.block(i, 4, 4, at) for i, at in enumerate(corners)]
    flags = q.flags_of({"A": records}, world=world, fps=2.0, jump_mm_s=limit)
    assert q.rows_with("JUMP", flags["A"]) == rows


@pytest.mark.parametrize("limit", [0.0, -5.0, float("nan"), float("inf")])
def test_a_jump_limit_that_makes_no_sense_is_refused_by_name(limit):
    with pytest.raises(ValueError, match="jump_mm_s"):
        q.flags_of({"A": [q.ellipse(0), q.ellipse(2)]}, jump_mm_s=limit)


# --------------------------------------------------------------------------- SIZE (last week's rule)


def test_size_marks_the_frames_whose_area_is_outside_half_to_twice_the_median():
    # the usual ellipse with its area scaled; the median over the 11 frames is the unscaled one
    scales = [1.0, 1.0, 2.25, 1.0, 1.9, 1.0, 0.4, 1.0, 0.55, 1.0, 1.0]
    records = [q.ellipse(2 * i, a=40.0 * np.sqrt(s), b=15.0 * np.sqrt(s)) for i, s in enumerate(scales)]
    assert q.flags_of({"A": records}, {"A": q.head()}) == {
        "A": ["", "", "SIZE", "", "", "", "SIZE", "", "", "", ""]}


BLOCKS = {100: (10, 10), 200: (10, 20), 50: (5, 10), 201: (3, 67), 49: (7, 7), 300: (15, 20), 99: (9, 11),
          260: (13, 20)}   # a rectangle of pixels for each area in px


@pytest.mark.parametrize("areas, rows", [
    ([100, 100, 100, 200, 50, 201, 49], [5, 6]),   # median 100: exactly twice and exactly half are inside
    ([100, 100, 300, 300], []),                    # median 200, the middle of the two middle values
    ([99, 100, 300, 300], [0]),
    ([100, 100, 260, None, None, None, None], [2]),   # lost frames (area 0) are not part of the median
])
def test_size_compares_with_the_median_of_the_visible_frames_strictly(areas, rows):
    records = [q.lost(i, FULL_HD) if area is None else q.block(i, *BLOCKS[area]) for i, area in enumerate(areas)]
    assert [one.area_px for one in records] == [area or 0 for area in areas]
    assert q.rows_with("SIZE", q.flags_of({"A": records})["A"]) == rows


def test_a_later_piece_of_an_animal_is_a_track_of_its_own():
    # A ends on frame 8; A2 starts on frame 10, 120 px away and a quarter of A's area. Taken as
    # one track that would be a jump of 466 mm/s, and A2 would be under half the median area.
    first, second = (640.37, CENTER[1]), (760.37, CENTER[1])
    records = {"A": [q.ellipse(2 * i, first) for i in range(5)],
               "A2": [q.ellipse(2 * i, second, 20.0, 7.5) for i in range(5, 10)]}
    flags = q.flags_of(records, {"A": q.head(first), "A2": q.head(second)})
    assert flags == {"A": [""] * 5, "A2": [""] * 5}


# --------------------------------------------------------------------------- CONTACT


def test_contact_marks_both_tracks_on_the_frames_where_their_outlines_come_close():
    # Two disks of radius 10 px in the full frame: two grid cells are 15 px. The water between
    # them is 60, 17, 13, 5 px, then they overlap by 5 px, then 13, 17, 60 px.
    gaps = [60.0, 17.0, 13.0, 5.0, -5.0, 13.0, 17.0, 60.0]
    records = {"A": [q.disk(2 * i, CENTER) for i in range(8)],
               "B": [q.disk(2 * i, (CENTER[0] + 20.0 + gap, CENTER[1])) for i, gap in enumerate(gaps)]}
    flags = q.flags_of(records)
    assert q.rows_with("CONTACT", flags["A"]) == [2, 3, 4, 5]
    assert q.rows_with("CONTACT", flags["B"]) == [2, 3, 4, 5]


# --------------------------------------------------------------------------- the text of a cell


def test_a_cell_lists_its_codes_in_the_order_of_spec_9_joined_by_semicolons():
    # Squares of pixels in the full frame, no head click: 10 px across (LOWRES), a round core
    # (ORIENT), HEADGUESS. On frame 5, A is a 20 x 20 px square on the first row of the frame
    # (EDGE), 400 px against a median of 100 (SIZE), with a second piece of 50 px (MULTI), 13.7 mm
    # from where it was 1/240 s before (JUMP) and 6 px from B (CONTACT; the limit is 15 px).
    together = shapes.blocks((40, 20), (0, 20, 0, 20), (30, 35, 0, 10))
    records = {"A": [q.block(i, 10, 10, (600, 300)) for i in range(5)] + [q.pixels(5, together, (900, 0))],
               "B": [q.block(i, 10, 10, (925, 5)) for i in range(6)]}
    flags = q.flags_of(records)
    assert flags["A"] == [USUAL] * 5 + ["JUMP;SIZE;CONTACT;EDGE;MULTI;LOWRES;ORIENT;HEADGUESS"]
    assert flags["B"] == [USUAL] * 5 + ["CONTACT;LOWRES;ORIENT;HEADGUESS"]


# --------------------------------------------------------------------------- inputs that do not fit


def test_tracks_that_do_not_belong_together_are_refused():
    derived, arrays, processing = q.tracks({"A": [q.ellipse(0), q.ellipse(2)], "B": [q.ellipse(0), q.ellipse(4)]})
    with pytest.raises(ValueError, match=r"\bB\b"):   # a derived track without its records
        qc.compute_flags(derived, {"A": arrays["A"]}, WORLD, processing)
    with pytest.raises(ValueError, match="frames"):   # records of other frames than the derived track
        qc.compute_flags({"A": derived["A"]}, {"A": arrays["B"]}, WORLD, processing)
