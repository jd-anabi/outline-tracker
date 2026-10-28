"""The dish circle and a tilted ellipse record, shared by the two files of the rules of `derive`.

tests/measure/test_derive_rules.py (lost rows, the heading reference, the head guess, settings) and
tests/measure/test_derive_rules_shapes.py (shapes without an extent or a hull, the size check) make the records of
their tracks with `tilted` and use `DISH` as the dish circle. Shapes and clicks are in image px, angles of shapes in
rad from +u toward +v.
"""

import derive_helpers as h
import numpy as np
from derive_helpers import CENTER

from outline_tracker.core.session import Boundary

DISH = Boundary(center_px=[900.2, 520.7], radius_px=400.0)


def tilted(frame, angle_deg, a=40.0, b=15.0):
    return h.ellipse_record(frame, CENTER, a, b, np.radians(angle_deg))
