"""What the tests of core/schema.py share: the column lists and dtypes of the output files, typed from SPEC 8.

Imported by name from tests/core/test_schema.py and tests/core/test_schema_texts.py.
"""

NAN = float("nan")


# ---------------------------------------------------------------------------------------------
# Column lists typed from the specification

POSITIONS_NAMES = [
    "track_id", "frame", "t_s", "x_mm", "y_mm", "u_px", "v_px", "area_mm2", "visible", "mode", "flags",
]
SHAPES_NAMES = [
    "track_id", "frame", "t_s", "area_mm2", "perimeter_mm", "major_mm", "minor_mm", "eccentricity",
    "theta_rad", "core_x_mm", "core_y_mm", "core_frac", "solidity", "circularity", "feret_max_mm",
    "n_components", "largest_fraction", "px_along_major", "cells_along_major", "shape_ok",
    "wall_dist_centroid_mm", "wall_dist_min_mm", "flags",
]
RADIAL_NAMES = ["track_id", "frame", "t_s"] + [f"r_{deg:03d}" for deg in range(0, 360, 5)]
PROBES_NAMES = ["frame", "t_s", "probe", "r", "g", "b", "gray"]

POSITIONS_DTYPES = dict(
    track_id="str", frame="int", t_s="float", x_mm="float", y_mm="float", u_px="float", v_px="float",
    area_mm2="float", visible="int", mode="str", flags="str",
)
SHAPES_DTYPES = dict(
    track_id="str", frame="int", t_s="float", area_mm2="float", perimeter_mm="float", major_mm="float",
    minor_mm="float", eccentricity="float", theta_rad="float", core_x_mm="float", core_y_mm="float",
    core_frac="float", solidity="float", circularity="float", feret_max_mm="float", n_components="int",
    largest_fraction="float", px_along_major="float", cells_along_major="float", shape_ok="int",
    wall_dist_centroid_mm="float", wall_dist_min_mm="float", flags="str",
)
RADIAL_DTYPES = dict(track_id="str", frame="int", t_s="float", **{n: "float" for n in RADIAL_NAMES[3:]})
PROBES_DTYPES = dict(frame="int", t_s="float", probe="str", r="float", g="float", b="float", gray="float")


def names(columns) -> list[str]:
    return [c.name for c in columns]
