"""A tilted ellipse record, shared by the two files of the rules of `derive`.

tests/measure/test_derive_rules.py (lost rows, the heading reference, the head guess, settings) and
tests/measure/test_derive_rules_shapes.py (shapes without an extent or a hull, the size check) make the records of
their tracks with `tilted`. Shapes and clicks are in image px, angles of shapes in rad from +u toward +v.
"""

import derive_helpers as h
import numpy as np
from derive_helpers import CENTER


def tilted(frame, angle_deg, a=40.0, b=15.0):
    return h.ellipse_record(frame, CENTER, a, b, np.radians(angle_deg))
