"""Shapes and paths of the synthetic clips: implicit functions and poses (decision X16).

A shape is an implicit function in its own body frame: `distance(xi, eta, t_s)` is a signed distance
in px, positive inside, zero on the outline. The body frame has its origin at the object's center,
xi toward the head and eta 90 degrees counterclockwise from xi; lengths are in px, `t_s` is the
time in s (only the shrimp's antennae depend on it). `reach_px` bounds the shape's extent.

A path gives the pose of an object in each frame: `pose(frame)` is the center (u, v) in px in the
full frame (SPEC 3.1: u to the right, v downward) and the heading in rad, counterclockwise on
screen from the image's rightward direction (y up), which is the world angle when the axis angle
alpha is 0. Speeds are in px per frame.

outline_tracker/synthetic.py puts shapes on paths into scenes, renders them and gives the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------------------------
# Shapes: signed distance in the body frame


@dataclass(frozen=True)
class Disk:
    """A disk of radius `radius_px` (px)."""

    radius_px: float

    @property
    def reach_px(self) -> float:
        """Largest distance of a point of the shape from the body-frame origin, in px."""
        return self.radius_px

    def distance(self, xi, eta, t_s: float = 0.0) -> np.ndarray:
        """Exact signed distance to the circle, in px, positive inside, at body-frame points (px)."""
        return self.radius_px - np.hypot(xi, eta)


@dataclass(frozen=True)
class Ellipse:
    """An ellipse with semi-axis `a_px` along the head direction and `b_px` across it (px)."""

    a_px: float
    b_px: float

    @property
    def reach_px(self) -> float:
        """Largest distance of a point of the shape from the body-frame origin, in px."""
        return max(self.a_px, self.b_px)

    def distance(self, xi, eta, t_s: float = 0.0) -> np.ndarray:
        """First-order signed distance to the outline, in px, positive inside, at body-frame points (px).

        d = (1 - q) / |grad q| with q = sqrt((xi/a)^2 + (eta/b)^2): zero exactly on the outline,
        exact along both axes, and within a few hundredths of a px of the true distance one px from
        the outline. It is never more than the smaller semi-axis.
        """
        q = np.hypot(xi / self.a_px, eta / self.b_px)
        slope = np.hypot(xi / self.a_px ** 2, eta / self.b_px ** 2)  # q |grad q|
        deepest = min(self.a_px, self.b_px)
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (1.0 - q) * q / slope
        return np.minimum(np.where(slope > 0, distance, deepest), deepest)


@dataclass(frozen=True)
class Shrimp:
    """An ellipse body with two antennae: thick segments with round ends, attached on the body axis.

    Lengths in px. The antennae start at (xi, eta) = (`attach_px`, 0) and point at the angles
    +beta and -beta from the head direction, with beta(t) = beta0 + beat sin(2 pi beat_hz t).
    With the defaults (45 and 30 degrees) the sweep never crosses the sideways position, so the
    hull area, and with it the solidity, oscillates at `beat_hz` and not at twice that.
    """

    a_px: float
    b_px: float
    antenna_length_px: float
    antenna_width_px: float
    attach_px: float
    beat_hz: float = 9.0
    beta0_deg: float = 45.0
    beat_deg: float = 30.0

    @property
    def reach_px(self) -> float:
        """Largest distance of a point of the shape from the body-frame origin, in px (an upper bound)."""
        return max(self.a_px, self.b_px, abs(self.attach_px) + self.antenna_length_px + self.antenna_width_px / 2)

    def beta_rad(self, t_s: float) -> float:
        """Angle of each antenna from the head direction at time `t_s` (s), in rad."""
        return float(np.radians(self.beta0_deg + self.beat_deg * np.sin(2 * np.pi * self.beat_hz * t_s)))

    def distance(self, xi, eta, t_s: float = 0.0) -> np.ndarray:
        """Signed distance in px, positive inside, at body-frame points (px) at time `t_s` (s).

        The union of the three parts: the largest of the body's first-order distance (see `Ellipse`)
        and the exact distances to the two thick segments.
        """
        distance = Ellipse(self.a_px, self.b_px).distance(xi, eta)
        beta, from_base = self.beta_rad(t_s), xi - self.attach_px
        for side in (1.0, -1.0):
            along_xi, along_eta = np.cos(beta), side * np.sin(beta)
            s = np.clip(from_base * along_xi + eta * along_eta, 0.0, self.antenna_length_px)
            to_center_line = np.hypot(from_base - s * along_xi, eta - s * along_eta)
            distance = np.maximum(distance, self.antenna_width_px / 2 - to_center_line)
        return distance


def shrimp_shape(mm_per_px: float) -> Shrimp:
    """The spec's shrimp at a given scale (mm per px): a 0.47 x 0.20 mm body with two antennae
    0.30 mm long and 0.03 mm wide, attached 0.14 mm in front of the center, beating at 9 Hz.

    At 0.010 mm/px the body is 47 x 20 px and the antennae 3 px wide; at 0.0324 mm/px the body is
    14.5 x 6.2 px.
    """
    return Shrimp(a_px=0.235 / mm_per_px, b_px=0.10 / mm_per_px, antenna_length_px=0.30 / mm_per_px,
                  antenna_width_px=0.03 / mm_per_px, attach_px=0.14 / mm_per_px)


# ---------------------------------------------------------------------------------------------
# Paths: the pose of an object in each frame


@dataclass(frozen=True)
class Straight:
    """A straight path at constant speed: `start_px` (u, v) in frame 0, `step_px` (du, dv) per frame.

    The head points along the motion unless `heading_rad` is given (needed for an object at rest).
    """

    start_px: tuple[float, float]
    step_px: tuple[float, float]
    heading_rad: float | None = None

    @property
    def speed_px_per_frame(self) -> float:
        """Speed in px per frame."""
        return float(np.hypot(*self.step_px))

    def pose(self, frame: float) -> tuple[float, float, float]:
        """Center (u, v) in px and heading in rad (see the module text) in frame `frame`."""
        heading = self.heading_rad
        if heading is None:
            heading = float(np.arctan2(0.0 - self.step_px[1], self.step_px[0]))
        return self.start_px[0] + self.step_px[0] * frame, self.start_px[1] + self.step_px[1] * frame, heading


@dataclass(frozen=True)
class Arc:
    """A circular path at constant speed around `center_px` (u, v) with radius `radius_px`.

    In frame n the object is at the angle `angle0_rad` + n `step_rad` around the center, measured
    counterclockwise on screen from the image's rightward direction; a positive `step_rad` (rad per
    frame) moves it counterclockwise. The head points along the motion; the heading is not wrapped.
    """

    center_px: tuple[float, float]
    radius_px: float
    angle0_rad: float
    step_rad: float

    @property
    def speed_px_per_frame(self) -> float:
        """Speed along the arc in px per frame."""
        return self.radius_px * abs(self.step_rad)

    def pose(self, frame: float) -> tuple[float, float, float]:
        """Center (u, v) in px and heading in rad (see the module text) in frame `frame`."""
        angle = self.angle0_rad + self.step_rad * frame
        heading = angle + (np.pi / 2 if self.step_rad >= 0 else -np.pi / 2)
        return (float(self.center_px[0] + self.radius_px * np.cos(angle)),
                float(self.center_px[1] - self.radius_px * np.sin(angle)), float(heading))
