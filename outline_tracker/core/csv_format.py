"""Write the rows of a table as CSV text: the header line, one line for a row, a whole file (SPEC 8, X11).

The columns (`Column` of core/schema.py) say each cell's type, unit and format; the functions here follow them. A
value is in the unit its column names (s, mm, mm^2, rad, px); no coordinate transformation happens here. The text
rules are those of schema.py: UTF-8, LF line ends on every platform, no quoting unless a cell holds a comma, a
quote or a line break, a missing number is an empty cell, integers are written as integers.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from outline_tracker.core.schema import CSV_NEWLINE, Column


def header_line(columns: Sequence[Column]) -> str:
    """The header line of a CSV file (no line end): the column names, comma separated. No units involved."""
    return ",".join(c.name for c in columns)


def format_row(columns: Sequence[Column], row: Mapping[str, object] | Sequence[object]) -> str:
    """One CSV line (no line end) for `row`, in the units and formats of `columns`.

    `row` is a mapping from column name to value (every column must be present; other keys are
    ignored) or a sequence in column order. Floats are written with the column's decimals; NaN, None
    and infinities become an empty cell. Integer columns must hold a whole number (a lost row keeps
    0): NaN or a fraction raises ValueError. Text is written as is, quoted only if it holds a comma,
    a quote or a line break. Values are in the units named by the columns (s, mm, mm^2, rad, px);
    no coordinate transformation happens here.
    """
    if isinstance(row, Mapping):
        for col in columns:
            if col.name not in row:
                raise KeyError(f"row has no value for column {col.name!r}")
        values = [row[col.name] for col in columns]
    else:
        values = list(row)
        if len(values) != len(columns):
            raise ValueError(f"row has {len(values)} values for {len(columns)} columns")
    return ",".join(_cell(col, value) for col, value in zip(columns, values))


def csv_text(columns: Sequence[Column], rows: Iterable[Mapping[str, object] | Sequence[object]]) -> str:
    """A whole CSV file as text: the header line, then one line per row, each ended by "\\n".

    Values are in the units of the columns (see `format_row`). Encode it with CSV_ENCODING and write
    the bytes (or open with newline="\\n"), so that the file has LF line ends on Windows too. With no
    rows the file is the header line alone.
    """
    lines = [header_line(columns)] + [format_row(columns, row) for row in rows]
    return CSV_NEWLINE.join(lines) + CSV_NEWLINE


def _cell(col: Column, value: object) -> str:
    if col.dtype == "str":
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if not isinstance(value, str):
            raise ValueError(f"column {col.name}: expected text, got {value!r}")
        if any(ch in value for ch in ',"\r\n'):
            return '"' + value.replace('"', '""') + '"'
        return value
    if col.dtype == "int":
        return format(_whole_number(col, value), col.fmt)
    if value is None:
        return ""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"column {col.name}: expected a number, got {value!r}")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as err:
        raise ValueError(f"column {col.name}: expected a number, got {value!r}") from err
    return format(number, col.fmt) if math.isfinite(number) else ""


def _whole_number(col: Column, value: object) -> int:
    if isinstance(value, (bool, int)):
        return int(value)
    problem = f"column {col.name}: an integer column needs a whole number (a lost row keeps 0), got {value!r}"
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(problem)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as err:
        raise ValueError(problem) from err
    if not math.isfinite(number) or number != int(number):
        raise ValueError(problem)
    return int(number)
