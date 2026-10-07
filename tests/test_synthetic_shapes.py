"""Shapes and paths of the synthetic clips (decision X16: shapes are implicit functions).

Expected values come from geometry written out here, never from the module's own output. A shape's
signed distance is in px, positive inside, in its body frame: xi toward the head, eta 90 degrees
counterclockwise from it. A path's pose is the center (u, v) in px in Tracker's convention
(SPEC 3.1: u to the right, v downward) and the heading in rad, counterclockwise on screen from the
image's rightward direction (y up).
"""

import numpy as np
import pytest

from outline_tracker.synthetic_shapes import Arc, Disk, Ellipse, Shrimp, Straight

CLOSEUP_SHRIMP = Shrimp(a_px=23.5, b_px=10.0, antenna_length_px=30.0, antenna_width_px=3.0, attach_px=14.0)


# ---------------------------------------------------------------------------------------------
# Shapes: signed distance in the body frame, in px, positive inside


def test_disk_distance_is_the_exact_distance_to_the_circle():
    disk = Disk(radius_px=10.0)
    xi, eta = np.array([0.0, 10.0, 12.0, 3.0, -6.0]), np.array([0.0, 0.0, 0.0, 4.0, -8.0])
    assert disk.distance(xi, eta) == pytest.approx([10.0, 0.0, -2.0, 5.0, 0.0], abs=1e-12)
    assert disk.reach_px == 10.0


def test_ellipse_distance_is_zero_on_the_outline_and_first_order_near_it():
    ellipse = Ellipse(a_px=40.0, b_px=15.0)
    t = np.linspace(0.0, 2 * np.pi, 97)
    on = np.column_stack([40.0 * np.cos(t), 15.0 * np.sin(t)])
    normal = np.column_stack([15.0 * np.cos(t), 40.0 * np.sin(t)])  # outward, from the gradient of q
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    assert ellipse.distance(on[:, 0], on[:, 1]) == pytest.approx(0.0, abs=1e-9)
    for offset in (-0.5, 0.5):  # half a pixel outside, half a pixel inside
        near = on - offset * normal
        assert ellipse.distance(near[:, 0], near[:, 1]) == pytest.approx(offset, abs=0.03)
    # exact along the axes; never more than the semi-minor axis (the center is b from the outline)
    assert ellipse.distance(np.array([39.0, 0.0, 44.0]), np.array([0.0, 13.0, 0.0])) == pytest.approx([1.0, 2.0, -4.0])
    assert ellipse.distance(np.array([0.0]), np.array([0.0])) == pytest.approx([15.0])
    assert ellipse.reach_px == 40.0


@pytest.mark.parametrize("quarter_beats, beta_deg", [(0, 45.0), (1, 75.0), (3, 15.0)])
def test_shrimp_antennae_are_thick_segments_at_beta_from_the_head_direction(quarter_beats, beta_deg):
    shrimp = CLOSEUP_SHRIMP
    t_s = quarter_beats / (4 * 9.0)  # beta(t) = 45 deg + 30 deg sin(2 pi 9 t)
    assert shrimp.beta_rad(t_s) == pytest.approx(np.radians(beta_deg))
    beta = np.radians(beta_deg)
    for side in (1.0, -1.0):
        along = np.array([np.cos(beta), side * np.sin(beta)])
        across = np.array([-along[1], along[0]])
        tip = np.array([14.0, 0.0]) + 30.0 * along
        points = np.array([tip,                         # the far end of the segment's center line
                           tip + 1.5 * along,           # the top of its round cap
                           tip + 3.5 * along,           # 2 px beyond the cap
                           tip - 8.0 * along + 1.5 * across,   # on the flank, away from the body
                           tip - 8.0 * along + 2.5 * across])  # 1 px outside the flank
        got = shrimp.distance(points[:, 0], points[:, 1], t_s)
        assert got == pytest.approx([1.5, 0.0, -2.0, 0.0, -1.0], abs=1e-9)
    assert shrimp.reach_px >= 14.0 + 30.0 + 1.5


def test_shrimp_is_the_union_of_body_and_antennae():
    shrimp, body = CLOSEUP_SHRIMP, Ellipse(23.5, 10.0)
    xi, eta = np.meshgrid(np.linspace(-50, 50, 81), np.linspace(-50, 50, 81))
    union = shrimp.distance(xi, eta, 0.0)
    assert (union >= body.distance(xi, eta) - 1e-12).all()  # union = max
    tail = xi < 0  # the antennae are in front: behind the center the shape is the body alone
    assert union[tail] == pytest.approx(body.distance(xi, eta)[tail])
    assert shrimp.distance(np.array([40.0]), np.array([0.0]), 0.0)[0] < 0  # between the antennae: outside


# ---------------------------------------------------------------------------------------------
# Paths: where an object is in frame n, and where its head points


def test_straight_path_position_heading_and_speed():
    path = Straight(start_px=(60.5, 100.25), step_px=(0.6, -0.8))
    u, v, heading = path.pose(10)
    assert (u, v) == pytest.approx((66.5, 92.25))
    assert heading == pytest.approx(np.arctan2(0.8, 0.6))  # up and to the right on screen
    assert path.speed_px_per_frame == pytest.approx(1.0)
    assert Straight((5.0, 5.0), (0.0, 0.0), heading_rad=0.3).pose(7) == pytest.approx((5.0, 5.0, 0.3))


@pytest.mark.parametrize("step_rad", [0.01, -0.02])
def test_arc_path_stays_on_its_circle_and_heads_along_the_motion(step_rad):
    path = Arc(center_px=(200.0, 150.0), radius_px=80.0, angle0_rad=0.4, step_rad=step_rad)
    assert path.speed_px_per_frame == pytest.approx(80.0 * abs(step_rad))
    u0, v0, _ = path.pose(0)
    assert (u0, v0) == pytest.approx((200.0 + 80.0 * np.cos(0.4), 150.0 - 80.0 * np.sin(0.4)))  # y is up
    for frame in (0, 17, 400):
        u, v, heading = path.pose(frame)
        assert np.hypot(u - 200.0, v - 150.0) == pytest.approx(80.0)
        ahead, behind = path.pose(frame + 0.001), path.pose(frame - 0.001)
        motion = np.array([ahead[0] - behind[0], -(ahead[1] - behind[1])])  # y up
        motion /= np.linalg.norm(motion)
        assert (np.cos(heading), np.sin(heading)) == pytest.approx(tuple(motion), abs=1e-6)
    assert abs(path.pose(400)[2] - path.pose(0)[2]) == pytest.approx(400 * abs(step_rad))  # not wrapped
