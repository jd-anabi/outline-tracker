"""Tests for the CONTACT flag (SPEC 9, decision X6): the distance between two outlines, the limit
of a pair of tracks from their grid cell sizes, the frames that count, and how many frames are
compared exactly (counted, not timed).

Distances are in px in the image plane. Polygons are closed (the last point joined to the first).
The tracks are disks with analytic logits or rectangles of pixels without logits, whose stored
outline runs through the centers of their border pixels, so the distance between two rectangles
side by side is a whole number of px. The limit of a pair is max(3 px, 2 grid cells of the coarser
of the two model inputs): 15 px in the full frame (cells of 7.5 px), 3 px between two 96 px windows.
"""

import dataclasses

import cv2
import numpy as np
import pytest
import qc_helpers as q
from qc_helpers import CENTER, CLOSE, FULL_HD, WORLD

from outline_tracker import qc, qc_contact

DISH_CROP = (404, 0, 1112, 1080)   # a dish crop: grid cells of 1112 / 256 = 4.34 px, two of them 8.69 px


def square(x, y, side):
    """A square with its corner of smallest x and y at (x, y), as a polygon of 4 points."""
    return np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], float)


def circle(center, radius, count=256):
    angles = 2.0 * np.pi * np.arange(count) / count
    return np.column_stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)])


def both_ways(first, second):
    """The distance, asked in both orders and with one polygon turned the other way round."""
    values = [qc.polygon_distance(first, second), qc.polygon_distance(second, first),
              qc.polygon_distance(first[::-1], second)]
    assert max(values) - min(values) <= 1e-12, values
    return values[0]


# --------------------------------------------------------------------------- the distance between two polygons


def test_the_distance_between_two_squares_is_the_water_between_them():
    assert both_ways(square(0, 0, 10), square(13, 2, 10)) == 3.0                       # side by side
    assert both_ways(square(0, 0, 10), square(13, 14, 10)) == pytest.approx(5.0, abs=1e-12)   # corner to corner: 3, 4


def test_the_nearest_point_may_lie_inside_an_edge():
    # the tip of a diamond 2 px from the middle of the square's right edge, 5.4 px from its corners
    diamond = np.array([[12.0, 5.0], [17.0, 0.0], [22.0, 5.0], [17.0, 10.0]])
    assert both_ways(square(0, 0, 10), diamond) == 2.0


@pytest.mark.parametrize("first, second", [
    (square(0, 0, 10), square(5, 5, 10)),                                # they overlap; each has a corner in the other
    (np.array([[-10.0, -1], [10, -1], [10, 1], [-10, 1]]),
     np.array([[-1.0, -10], [1, -10], [1, 10], [-1, 10]])),              # a cross: no corner of one lies in the other
    (square(0, 0, 100), square(40, 40, 10)),                             # one inside the other, the edges 40 px apart
    (square(0, 0, 10), square(10, 10, 5)),                               # they touch at one corner
    (square(0, 0, 10), square(10, 2, 5)),                                # they share a piece of an edge
    (square(0, 0, 10), square(0, 0, 10)),                                # the same polygon twice
], ids=["overlap", "cross", "nested", "corner", "edge", "same"])
def test_polygons_that_overlap_or_touch_are_0_apart(first, second):
    assert both_ways(first, second) == pytest.approx(0.0, abs=1e-12)


def test_an_outline_without_extent_is_a_point():
    point = np.tile([15.0, 5.0], (6, 1))   # the stored outline of a mask of one pixel
    assert both_ways(point, square(0, 0, 10)) == 5.0
    assert both_ways(np.tile([4.0, 5.0], (6, 1)), square(0, 0, 10)) == 0.0   # inside the square
    assert both_ways(point, np.tile([18.0, 9.0], (3, 1))) == pytest.approx(5.0, abs=1e-12)   # two points: 3, 4


def test_two_circles_of_256_points_are_their_center_distance_minus_two_radii_apart():
    assert both_ways(circle((0.0, 0.0), 10.0), circle((26.0, 0.0), 10.0)) == pytest.approx(6.0, abs=1e-9)
    # along (3, 4) / 5 no point lies on the line of centers: each polygon is up to
    # R (1 - cos(pi / 256)) = 0.0008 px inside its circle there
    slanted = both_ways(circle((0.0, 0.0), 10.0), circle((15.6, 20.8), 10.0))
    assert 6.0 <= slanted < 6.002


def test_random_convex_polygons_agree_with_an_answer_found_another_way():
    # Two convex polygons overlap unless the direction across one of their edges separates them
    # (the separating-axis rule). If one does, they are as far apart as the nearest corner of
    # either is from the other polygon, which OpenCV measures (pointPolygonTest, negative outside).
    rng = np.random.default_rng(7)

    def convex(center):
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, int(rng.integers(3, 12))))
        a, b, turn = rng.uniform(5.0, 30.0), rng.uniform(5.0, 30.0), rng.uniform(0.0, np.pi)
        x, y = a * np.cos(angles), b * np.sin(angles)
        points = np.column_stack([center[0] + x * np.cos(turn) - y * np.sin(turn),
                                  center[1] + x * np.sin(turn) + y * np.cos(turn)])
        return points.astype(np.float32).astype(np.float64)   # the numbers OpenCV works with

    def separation(p, r):
        """The widest gap between p and r along the normals of their edges; negative when none separates them."""
        widest = -np.inf
        for polygon in (p, r):
            edges = np.roll(polygon, -1, axis=0) - polygon
            edges = edges[np.hypot(edges[:, 0], edges[:, 1]) > 0]
            normals = np.column_stack([-edges[:, 1], edges[:, 0]]) / np.hypot(edges[:, 0], edges[:, 1])[:, None]
            on_p, on_r = p @ normals.T, r @ normals.T
            gaps = np.maximum(on_r.min(axis=0) - on_p.max(axis=0), on_p.min(axis=0) - on_r.max(axis=0))
            widest = max(widest, float(gaps.max()))
        return widest

    def outside(points, polygon):
        contour = polygon.astype(np.float32).reshape(-1, 1, 2)
        return min(-cv2.pointPolygonTest(contour, (float(x), float(y)), True) for x, y in points)

    apart = overlapping = 0
    for _ in range(400):
        p, r = convex((0.0, 0.0)), convex(rng.uniform(-70.0, 70.0, 2))
        gap = separation(p, r)
        if abs(gap) < 1e-3:
            continue   # too close to call
        expected = min(outside(p, r), outside(r, p)) if gap > 0 else 0.0
        apart, overlapping = apart + (gap > 0), overlapping + (gap < 0)
        assert qc.polygon_distance(p, r) == pytest.approx(expected, abs=1e-4)
    assert apart > 200 and overlapping > 30


# --------------------------------------------------------------------------- the limit of a pair (X6)


def pair(gap, box_a, box_b, radius=10.0):
    """Two disks of `radius` px with `gap` px of water between them, one frame each, seen through
    the model inputs `box_a` and `box_b`: {track id: [record]}."""
    other = (CENTER[0] + 2.0 * radius + gap, CENTER[1])
    return {"A": [q.disk(0, CENTER, radius, box_a)], "B": [q.disk(0, other, radius, box_b)]}


FINE_A, FINE_B = q.window(CENTER, 96), q.window((CENTER[0] + 24.0, CENTER[1]), 96)


@pytest.mark.parametrize("gap, box_a, box_b, contact", [
    (13.5, FULL_HD, FULL_HD, True), (16.5, FULL_HD, FULL_HD, False),       # two cells of 7.5 px
    (7.5, DISH_CROP, DISH_CROP, True), (10.0, DISH_CROP, DISH_CROP, False),   # two cells of 4.34 px
    (2.5, CLOSE, CLOSE, True), (3.5, CLOSE, CLOSE, False),                 # two cells are 2 px: 3 px is the least
    (2.0, FINE_A, FINE_B, True), (4.0, FINE_A, FINE_B, False),             # two fine tracks: 3 px
    (13.5, FULL_HD, FINE_B, True), (13.5, FINE_A, FULL_HD, True),          # the coarser of the two decides,
    (16.5, FULL_HD, FINE_B, False),                                        # whichever track it belongs to
])
def test_the_limit_is_two_grid_cells_of_the_coarser_input_and_at_least_3_px(gap, box_a, box_b, contact):
    flags = q.flags_of(pair(gap, box_a, box_b))
    assert q.rows_with("CONTACT", flags["A"]) == q.rows_with("CONTACT", flags["B"]) == ([0] if contact else [])


@pytest.mark.parametrize("box_a, box_b, apart, contact", [
    (FULL_HD, FULL_HD, 15, True), (FULL_HD, FULL_HD, 16, False),
    ((560, 260, 96, 96), (580, 265, 96, 96), 3, True), ((560, 260, 96, 96), (580, 265, 96, 96), 4, False),
])
def test_an_outline_exactly_at_the_limit_is_in_contact(box_a, box_b, apart, contact):
    # Two 20 x 20 px squares side by side: the outline of A runs through the pixel centers of
    # column 619, that of B through those of column 619 + apart.
    records = {"A": [q.block(0, 20, 20, (600, 300), box_a)], "B": [q.block(0, 20, 20, (619 + apart, 305), box_b)]}
    flags = q.flags_of(records)
    assert q.rows_with("CONTACT", flags["A"]) == q.rows_with("CONTACT", flags["B"]) == ([0] if contact else [])


def test_one_mask_inside_another_is_in_contact():
    # Two tracks on one animal: the small disk lies in the large one, their outlines 30 px apart.
    records = {"A": [q.disk(2 * i, CENTER, 40.0) for i in range(3)],
               "B": [q.disk(2 * i, CENTER, 10.0) for i in range(3)]}
    flags = q.flags_of(records)
    assert q.rows_with("CONTACT", flags["A"]) == q.rows_with("CONTACT", flags["B"]) == [0, 1, 2]


# --------------------------------------------------------------------------- which frames count


def test_contact_needs_both_tracks_visible_on_the_same_video_frame():
    near, far = (CENTER[0] + 25.0, CENTER[1]), (CENTER[0] + 300.0, CENTER[1])   # 5 px and 280 px of water
    # A: frames 0 to 10, lost on frame 6. B: frames 4 to 14, next to A's place on frames 6, 8 and 12.
    a_rows = [q.lost(frame, FULL_HD) if frame == 6 else q.disk(frame, CENTER) for frame in range(0, 12, 2)]
    b_rows = [q.disk(frame, near if frame in (6, 8, 12) else far) for frame in range(4, 16, 2)]
    flags = q.flags_of({"A": a_rows, "B": b_rows})
    assert q.rows_with("CONTACT", flags["A"]) == [4]   # frame 8
    assert q.rows_with("CONTACT", flags["B"]) == [2]   # frame 8
    assert flags["A"][3] == "LOST;HEADGUESS"


def test_three_tracks_are_compared_pair_by_pair():
    left, right = CENTER, (CENTER[0] + 200.0, CENTER[1])
    path = [(CENTER[0] + 100.0, CENTER[1] + 150.0), (left[0] + 25.0, left[1]), (CENTER[0] + 100.0, CENTER[1]),
            (right[0] - 25.0, right[1]), (CENTER[0] + 100.0, CENTER[1] - 150.0)]
    records = {"A": [q.disk(2 * i, left) for i in range(5)], "B": [q.disk(2 * i, at) for i, at in enumerate(path)],
               "C": [q.disk(2 * i, right) for i in range(5)]}
    flags = q.flags_of(records)
    assert [q.rows_with("CONTACT", flags[name]) for name in "ABC"] == [[1], [1, 3], [3]]


# --------------------------------------------------------------------------- how many frames are compared exactly


def test_of_ten_tracks_over_1200_frames_only_the_200_frames_with_near_boxes_are_compared_exactly(monkeypatch):
    # Counted, not timed (docs/ROADMAP.md, section 2, rule 6). Comparing two outlines exactly takes
    # every point of one against every edge of the other, and ten tracks over 1200 frames are 45
    # pairs on 1200 frames each: 54000 comparisons, were every one made. The boxes around the
    # outlines say where none is needed.
    # Ten disks of radius 7 px in the full frame (limit 15 px), 170 px apart, each going round a
    # circle of 20 px. Track 1 sits 22 px from track 0 on frames 400 to 499 (8 px of water:
    # contact). Track 3 sits (24, 24) px from track 2 on frames 800 to 899: their bounding boxes
    # are 14.1 px apart, their outlines 19.9 px (no contact). Every other pair is far apart.
    n, home = 1200, np.array([200.0, 200.0])
    base_derived, base_arrays, processing = q.tracks(
        {"A": [dataclasses.replace(q.disk(0, home, 7.0), frame=i) for i in range(n)]})
    one, rows = base_arrays["A"], np.arange(n)
    wobble = 20.0 * np.column_stack([np.cos(rows / 30.0), np.sin(rows / 30.0)])
    derived, arrays, centers = {}, {}, []
    for j in range(10):
        at = np.array([150.0 + 170.0 * j, 300.0]) + wobble
        if j == 1:
            at[400:500] = np.array([150.0 + 22.0, 300.0]) + wobble[400:500]
        if j == 3:
            at[800:900] = np.array([490.0 + 24.0, 300.0 + 24.0]) + wobble[800:900]
        centers.append(at)
        # the same disk moved from `home` to `at`: centroids and outlines move with it (the mask
        # crops stay behind: no flag reads them)
        move = at - home
        moved = dataclasses.replace(
            one, u=one.u + move[:, 0], v=one.v + move[:, 1], core_u=one.core_u + move[:, 0],
            core_v=one.core_v + move[:, 1], outline_px=(one.outline_px + move[:, None, :]).astype(np.float32))
        x, y = WORLD.to_world(moved.u, moved.v)
        arrays[f"T{j}"] = moved
        derived[f"T{j}"] = dataclasses.replace(base_derived["A"], track_id=f"T{j}", x_mm=x, y_mm=y, u_px=moved.u,
                                               v_px=moved.v)

    # Which pairs come near on which frames, worked out from the scene. A disk of radius 7 px fills a
    # box of 14 px, so the boxes of two disks whose centers are (du, dv) apart are (|du| - 14, |dv| - 14)
    # apart, where that is positive. T0 and T1 on frames 400 to 499: (22 - 14, 0), 8 px. T2 and T3 on
    # frames 800 to 899: (10, 10), 14.1 px. Both are within the 15 px. Nothing else is nearer than
    # 156 px (two neighbours in the row: 170 - 14). So 100 + 100 frames are compared exactly.
    # (The stored outline lies within 0.05 px of the disk's circle, asserted here; the nearest case has
    # 0.8 px to spare.)
    assert np.abs(np.hypot(*(one.outline_px[0] - home).T) - 7.0).max() < 0.05
    near = {}
    for a in range(10):
        for b in range(a + 1, 10):
            between = np.maximum(np.abs(centers[a] - centers[b]) - 14.0, 0.0)
            frames = np.flatnonzero(np.hypot(between[:, 0], between[:, 1]) <= 15.0).tolist()
            if frames:
                near[a, b] = frames
    assert near == {(0, 1): list(range(400, 500)), (2, 3): list(range(800, 900))}

    compared = []   # the frames that each call of the exact comparison got
    exact = qc_contact._distances

    def counted(p, other):
        compared.append(len(p))
        return exact(p, other)

    monkeypatch.setattr(qc_contact, "_distances", counted)
    flags = qc.compute_flags(derived, arrays, WORLD, processing)
    touching = list(range(400, 500))
    assert [q.rows_with("CONTACT", flags[f"T{j}"]) for j in range(10)] == [touching, touching] + [[]] * 8
    assert all(len(flags[f"T{j}"]) == n for j in range(10))
    assert sum(compared) == 200   # of the 54000
