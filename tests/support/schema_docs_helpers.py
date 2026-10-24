"""What the tests of the two texts of core/schema_docs.py share: reading Markdown, and the radius columns.

Imported by name from tests/core/test_schema_docs_readme.py and tests/core/test_schema_docs_outputs.py.
"""

import re

from helpers import normalized


RADIAL_COLUMNS = [f"r_{deg:03d}" for deg in range(0, 360, 5)]  # SPEC 8.5: r_000 ... r_355


def md_section(text: str, heading: str) -> list[str]:
    """Lines under `heading` (such as "## Conventions") up to the next heading of the same or a higher level."""
    lines = text.splitlines()
    level = len(heading) - len(heading.lstrip("#"))
    start = lines.index(heading)
    end = next((i for i in range(start + 1, len(lines)) if re.match(rf"#{{1,{level}}} ", lines[i])), len(lines))
    return lines[start + 1 : end]


def cells(line: str) -> list[str]:
    """The cells of one table row, `\\|` read as a bar inside a cell."""
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def plain(text: str) -> str:
    """Markdown text as README text: no code marks, whitespace collapsed."""
    return normalized(text.replace("`", ""))
