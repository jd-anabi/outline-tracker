"""Measuring a mask: where an object is. Starts with the ported `mask_center` (A11 adds the rest).

`mask_center` is last week's function from shrimp.segment, moved over unchanged and checked against
the reference copy by tests/test_port_fidelity.py and tests/test_port_equivalence.py.

Coordinates: masks are arrays indexed [row, column]. Positions are in Tracker's image coordinates,
in px: the origin is the top-left corner of the frame, u grows to the right, v downward, and the
pixel in column c and row r has its center at (c + 0.5, r + 0.5) (SPEC 3.1).
"""

from __future__ import annotations

import numpy as np


def mask_center(mask: np.ndarray) -> tuple[float, float, int]:
    """Center of a boolean mask in Tracker's image coordinates, and its area in pixels.

    The mean column and row of the mask's pixels, plus 0.5: in Tracker, pixel (column c, row r) is the
    square from c to c + 1 and r to r + 1, so its center is at (c + 0.5, r + 0.5). NaN if empty.
    """
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return float("nan"), float("nan"), 0
    return float(cols.mean() + 0.5), float(rows.mean() + 0.5), int(len(rows))
