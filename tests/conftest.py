"""Shared test helpers (provided by the course; you do not need to change this file)."""


def java_sci(v):
    """A number the way Tracker writes it with Number Format "Full Precision" (Java's 0.000000E0)."""
    if v == 0:
        return "0.000000E0"
    mantissa, exponent = f"{v:.6E}".split("E")
    return f"{mantissa}E{int(exponent)}"
