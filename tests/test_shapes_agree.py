"""The two shape modules give the same disk and the same ellipse (docs/ROADMAP.md, W1 step 6).

tests/analytic_shapes.py (functions of the pixel centers of a raster) and
outline_tracker/synthetic_shapes.py (classes in a body frame, which outline_tracker/synthetic.py puts
on the pixel grid) overlap in two shapes: the disk and the ellipse. Two tests:

1. the anchor: each of the two is held to geometry by itself. A point of the closed-form outline (the
   parametric circle and ellipse) is at distance 0; a point half a pixel inside or outside a circle is
   at +0.5 or -0.5; the point on the first axis of an ellipse, half a pixel inside its end, is at 0.5
   (the first-order distance is exact along the axes); and a disk smaller than a pixel is the one pixel
   whose center it covers;
2. the agreement: at the pixel centers of a raster the two give the same signed distances within
   1e-9 px and the same masks, and the package's own ground truth (`GroundTruth.mask`) is that mask.
   Two implementations that agree prove nothing by themselves: the anchor holds both.

Units and coordinates (SPEC 3.1): px in the full frame, u to the right, v downward; the pixel in
column c and row r has its center at (c + 0.5, r + 0.5); arrays are indexed [row, column]. A signed
distance is in px, positive inside. Angles are in rad, and the two modules turn them the other way: an
`angle` of tests/analytic_shapes.py goes from +u toward +v, clockwise on screen; a heading of the
package is counterclockwise on screen (y up). So the same ellipse has heading = -angle. The package's
body frame has xi toward the head and eta 90 degrees counterclockwise from xi.
"""

import numpy as np
import pytest
from analytic_shapes import disk, ellipse, pixel_centers

from outline_tracker.synthetic import GroundTruth, Scene, SceneObject
from outline_tracker.synthetic_shapes import Disk, Ellipse, Straight

RASTER = (120, 140)  # rows, columns of the raster of the second test, px: every case lies inside it
# (center (u, v), radius), px; the centers are at sub-pixel offsets
DISKS = [((60.3, 50.7), 10.0), ((57.25, 49.8), 6.25), ((63.71, 52.14), 22.4), ((70.9, 41.35), 2.5)]
# (center (u, v), a, b, angle): the semi-axes in px, a along the angle and b across it; the angle in rad,
# from +u toward +v
ELLIPSES = [((70.3, 60.7), 40.0, 15.0, 0.0), ((68.25, 57.8), 23.5, 10.0, 0.7), ((71.71, 62.14), 8.0, 3.0, -1.1),
            ((69.4, 61.35), 12.0, 30.0, 2.4)]
AROUND = np.linspace(0.0, 2 * np.pi, 97)  # where on an outline the first test looks, rad


def body_frame(u, v, center, heading: float):
    """(xi, eta) of the points (u, v), px in the full frame, in the body frame of an object at `center`
    (u, v) whose head points along `heading` (rad, counterclockwise on screen): xi toward the head,
    eta 90 degrees counterclockwise from xi, px. Written out from the frame's definition
    (outline_tracker/synthetic_shapes.py), not taken from the package's code."""
    x, y = u - center[0], -(v - center[1])  # y up
    return x * np.cos(heading) + y * np.sin(heading), -x * np.sin(heading) + y * np.cos(heading)


def ground_truth_mask(shape, center, heading: float) -> np.ndarray:
    """The package's true mask [row, column] of one object that stands still in a frame of the size of
    `RASTER`: `shape` at `center` (u, v), px, with the heading `heading`, rad."""
    still = Straight(center, (0.0, 0.0), heading_rad=heading)
    scene = Scene(size=RASTER[::-1], fps=240.0, n_frames=1, objects=(SceneObject("A", shape, still),), mm_per_px=0.01)
    return GroundTruth(scene).mask("A", 0)


def test_disk_and_ellipse_lie_on_their_closed_form_outlines():
    for center, radius in DISKS:
        # on the circle, half a pixel inside it, half a pixel outside it: the distance to a circle is exact
        for inside in (0.0, 0.5, -0.5):
            u = center[0] + (radius - inside) * np.cos(AROUND)
            v = center[1] + (radius - inside) * np.sin(AROUND)
            assert disk(u, v, center, radius) == pytest.approx(inside, abs=1e-9)
            assert Disk(radius).distance(*body_frame(u, v, center, 0.0)) == pytest.approx(inside, abs=1e-9)

    for center, a, b, angle in ELLIPSES:
        cos, sin = np.cos(angle), np.sin(angle)
        # the outline: the points (a cos p, b sin p) of the ellipse's own axes, turned by the angle from
        # +u toward +v and moved to the center
        u = center[0] + a * np.cos(AROUND) * cos - b * np.sin(AROUND) * sin
        v = center[1] + a * np.cos(AROUND) * sin + b * np.sin(AROUND) * cos
        assert ellipse(u, v, center, a, b, angle) == pytest.approx(0.0, abs=1e-9)
        assert Ellipse(a, b).distance(*body_frame(u, v, center, -angle)) == pytest.approx(0.0, abs=1e-9)
        # On the first axis, half a pixel inside its end: 0.5 px from the outline, and there the
        # first-order distance is exact. With the angle turned the other way the point is off the axis:
        # this pins the sign of the angle, and that the package's heading is minus the angle.
        end_u, end_v = np.array([center[0] + (a - 0.5) * cos]), np.array([center[1] + (a - 0.5) * sin])
        assert ellipse(end_u, end_v, center, a, b, angle) == pytest.approx(0.5, abs=1e-9)
        assert Ellipse(a, b).distance(*body_frame(end_u, end_v, center, -angle)) == pytest.approx(0.5, abs=1e-9)

    # A disk of radius 0.6 px around the center of the pixel in column 10 and row 20: that center is
    # 0.6 px inside, and the centers of the four pixels beside it are 1 px away, 0.4 px outside. On a grid
    # without the + 0.5 the nearest centers would be 0.71 px away and the mask empty: this pins the grid.
    center, radius = (10.5, 20.5), 0.6
    u, v = pixel_centers(*RASTER)
    assert np.argwhere(disk(u, v, center, radius) > 0).tolist() == [[20, 10]]
    assert np.argwhere(Disk(radius).distance(*body_frame(u, v, center, 0.0)) > 0).tolist() == [[20, 10]]
    assert np.argwhere(ground_truth_mask(Disk(radius), center, 0.0)).tolist() == [[20, 10]]


def _assert_the_same_shape(analytic, package, truth_mask) -> None:
    """`analytic` and `package`: the signed distances (px) that the two modules give at the pixel
    centers of the raster; `truth_mask`: the package's ground-truth mask of the same shape."""
    # the precondition: no pixel center is so near the outline that rounding could decide its side
    assert np.abs(analytic).min() > 1e-6 and np.abs(package).min() > 1e-6
    assert np.abs(analytic - package).max() <= 1e-9
    assert np.array_equal(analytic > 0, package > 0)
    assert np.array_equal(truth_mask, analytic > 0)


def test_disk_and_ellipse_of_both_modules_give_the_same_masks_and_distances():
    u, v = pixel_centers(*RASTER)
    for center, radius in DISKS:
        _assert_the_same_shape(disk(u, v, center, radius), Disk(radius).distance(u - center[0], -(v - center[1])),
                               ground_truth_mask(Disk(radius), center, 0.0))
    for center, a, b, angle in ELLIPSES:
        _assert_the_same_shape(ellipse(u, v, center, a, b, angle),
                               Ellipse(a, b).distance(*body_frame(u, v, center, -angle)),
                               ground_truth_mask(Ellipse(a, b), center, -angle))
