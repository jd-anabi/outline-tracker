"""The round number badge of a panel (task C0b; SPEC 10.1): where its digit is drawn, and its fill.

The badge is a circle 20 px across with an outline ring of 1 px; inside it are the fill of the
panel's state and the panel's number (gui/panel.py, the badge rules of gui/theme.py). What a
well-drawn badge has does not depend on the system's font, it is geometry: the digit lies inside
the circle with 2 px or more to the circle's edge, its middle is within 1.5 px of the badge's
middle, and the rest of the circle shows the state's fill, which is a theme token.

The badge is read from the drawn window (`grab`, offscreen), never from the style sheet's text.
Lengths are Qt's device-independent px, x to the right and y down from the badge's top left
corner; the device pixel (column, row) covers [column, column + 1) x [row, row + 1), divided by
the image's device pixel ratio.
"""

import math
import os
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QFont, QFontDatabase, QFontInfo, QGuiApplication

from outline_tracker.gui import theme
from test_theme import look  # noqa: F401  the fixture that puts the application's look back

SIZE = 20           # the badge: a circle this many px across
MIDDLE = SIZE / 2   # its centre, px from its left and from its top edge; also the circle's radius
RING = 1            # the outline ring's width, px
MARGIN = 2          # from the digit's bounding box to the circle's edge: at least this, px
OFF_CENTRE = 1.5    # from the digit's middle to the badge's middle: at most this, px, in x and in y
# A pixel whose centre is nearer to the badge's centre than this lies wholly inside the ring: the
# ring's inner edge is 9 px from the centre, and half the diagonal of a device pixel is 0.71 px.
INSIDE = MIDDLE - RING - 0.75

STATES = ("todo", "done", "attention")
# state -> the tokens of the fill and of the digit (the design note's three badge rules)
TOKENS = {"todo": ("panel", "text"), "done": ("success", "panel"), "attention": ("warningBg", "warning")}


@dataclass
class Reading:
    """What is drawn in a badge. `box`: the digit's bounding box (left, top, right, bottom; px from the
    badge's top left corner, x right, y down), None when no pixel has the digit's colour. `fill`: the
    most frequent colour inside the ring beside the digit, `#RRGGBB`. `picture`: one text row per row
    of device pixels (`#` digit, `.` fill, `:` another colour inside the ring; outside it `+` where
    the colour is nearer to the digit's than to the fill's)."""

    box: tuple[float, float, float, float] | None
    fill: str
    picture: str


def nearer(color: str, to: str, than: str) -> bool:
    """True when `color` is nearer to the colour `to` than to the colour `than` (all `#RRGGBB`;
    distance in the cube of red, green and blue)."""
    def distance(other: str) -> int:
        return sum((int(color[i:i + 2], 16) - int(other[i:i + 2], 16)) ** 2 for i in (1, 3, 5))

    return distance(to) < distance(than)


def read(window, badge, fill: str, digit: str) -> Reading:
    """Read `badge` as `window` draws it, given its state's `fill` colour and its `digit` colour
    (`#RRGGBB`). A pixel of the digit is one inside the ring whose colour is nearer to `digit` than
    to `fill`; lengths in the result are px from the badge's top left corner."""
    image = window.grab().toImage()
    ratio = image.devicePixelRatio()
    corner = badge.mapTo(window, QPoint(0, 0)) * ratio
    side = round(SIZE * ratio)
    columns, rows, others, picture = [], [], Counter(), []
    for row in range(side):
        line = ""
        for column in range(side):
            color = image.pixelColor(corner.x() + column, corner.y() + row).name().upper()
            x, y = (column + 0.5) / ratio, (row + 0.5) / ratio  # the pixel's centre
            like_digit = nearer(color, digit, fill)
            if math.hypot(x - MIDDLE, y - MIDDLE) >= INSIDE:
                line += "+" if like_digit else " "
            elif like_digit:
                columns.append(column)
                rows.append(row)
                line += "#"
            else:
                others[color] += 1
                line += "." if color == fill else ":"
        picture.append(line)
    box = None
    if columns:
        box = (min(columns) / ratio, min(rows) / ratio, (max(columns) + 1) / ratio, (max(rows) + 1) / ratio)
    return Reading(box, others.most_common(1)[0][0], "\n".join(picture))


def faults(seen: Reading, fill: str) -> list[str]:
    """What a well-drawn badge would not have, in words; empty for a well-drawn one. `seen` is the
    badge's `Reading` (px), `fill` the fill its state asks for (`#RRGGBB`)."""
    found = []
    if seen.box is None:
        found.append("no pixel inside the circle has the digit's colour")
    else:
        left, top, right, bottom = seen.box
        reach = max(math.hypot(x - MIDDLE, y - MIDDLE) for x in (left, right) for y in (top, bottom))
        if reach > MIDDLE - MARGIN:
            found.append(f"a corner of the digit's box is {reach:.2f} px from the centre: more than "
                         f"{MIDDLE - MARGIN:g} px, so less than {MARGIN} px from the circle's edge")
        off = ((left + right) / 2 - MIDDLE, (top + bottom) / 2 - MIDDLE)
        if max(abs(off[0]), abs(off[1])) > OFF_CENTRE:
            found.append(f"the digit's middle is ({off[0]:+.2f}, {off[1]:+.2f}) px (x right, y down) from the "
                         f"badge's middle: more than {OFF_CENTRE} px")
    if seen.fill != fill:
        found.append(f"the most frequent colour beside the digit is {seen.fill}, not the fill {fill}")
    return found


def described(badge, seen: Reading) -> str:
    """The facts of one badge for a failure message: the digit's box, the font really used, the
    platform, and the picture. Lengths in px; the font's size in px and in points."""
    used, asked = QFontInfo(badge.font()), badge.font()
    screen = QGuiApplication.primaryScreen()
    return (f"badge {badge.text()!r}, state {badge.property('state')!r}: digit box (left, top, right, bottom) = "
            f"{seen.box} px of {SIZE} x {SIZE}; fill seen {seen.fill}\n"
            f"font family {used.family()!r}, pixel size {used.pixelSize()} px ({used.pointSizeF():g} pt, weight "
            f"{used.weight()}, exact match {used.exactMatch()}; asked for {asked.family()!r}, "
            f"{asked.pointSizeF():g} pt); {len(QFontDatabase.families())} font families known\n"
            f"platform {QGuiApplication.platformName()!r} on {sys.platform}, logical dpi "
            f"{screen.logicalDotsPerInch():g}, device pixel ratio {screen.devicePixelRatio():g}\n{seen.picture}")


@pytest.mark.parametrize("number", [1, 9])
@pytest.mark.parametrize("state", STATES)
def test_the_digit_is_in_the_middle_of_a_circle_of_the_states_fill(state, number, window, qapp, qtbot, look):  # noqa: F811
    theme.apply(qapp, False)
    panel = window.panels[number - 1]
    assert panel.badge.text() == str(number)
    assert (panel.badge.width(), panel.badge.height()) == (SIZE, SIZE)
    panel.set_state(state)
    with qtbot.waitExposed(window):
        window.show()
    fill, digit = (theme.LIGHT[token] for token in TOKENS[state])

    seen = read(window, panel.badge, fill, digit)

    facts = described(panel.badge, seen)
    print(facts)
    warnings.warn(facts, stacklevel=1)  # C0b step 1 only: a passing run on the test machines shows the numbers
    assert faults(seen, fill) == [], facts


# ---------------------------------------------------------------------------------------------
# C0b step 1 only (taken out again with the fix): the same reading in fresh processes, to learn
# what the Windows test machine draws when text has a font. No window is shown by any of them.

PROBE = "import sys; sys.path.insert(0, sys.argv[1]); import test_badge; test_badge.probe(sys.argv[2])"


def probe(variant: str) -> None:
    """Print the facts of the six badges (three states, numbers 1 and 9) of windows that are made
    and read but never shown, in this process's platform; `variant` "segoe" first loads Windows'
    own interface font from its font folder."""
    from PySide6.QtWidgets import QApplication

    from outline_tracker.gui.main_window import MainWindow

    app = QApplication([])
    if variant == "segoe":
        folder = Path(os.environ["WINDIR"]) / "Fonts"
        ids = [QFontDatabase.addApplicationFont(str(folder / name)) for name in ("segoeui.ttf", "seguisb.ttf", "segoeuib.ttf")]
        print("added fonts:", [(i, QFontDatabase.applicationFontFamilies(i)) for i in ids])
        app.setFont(QFont("Segoe UI", 9))
    theme.apply(app, False)
    print(f"application font: {app.font().family()!r} {app.font().pointSizeF():g} pt -> "
          f"{QFontInfo(app.font()).family()!r} {QFontInfo(app.font()).pixelSize()} px")
    for state in STATES:
        for number in (1, 9):
            window = MainWindow()
            panel = window.panels[number - 1]
            panel.set_state(state)
            fill, digit = (theme.LIGHT[token] for token in TOKENS[state])
            seen = read(window, panel.badge, fill, digit)
            print(f"faults: {faults(seen, fill)}\n{described(panel.badge, seen)}")
            window.close()


def test_facts_from_fresh_processes():
    fonts = str(Path(os.environ.get("WINDIR", "")) / "Fonts")
    variants = {"offscreen": {"QT_QPA_PLATFORM": "offscreen"}}
    if sys.platform == "win32":
        variants |= {
            "offscreen, QT_QPA_FONTDIR": {"QT_QPA_PLATFORM": "offscreen", "QT_QPA_FONTDIR": fonts},
            "segoe": {"QT_QPA_PLATFORM": "offscreen"},
            "native": {"QT_QPA_PLATFORM": "windows"},
            "native, 125 %": {"QT_QPA_PLATFORM": "windows", "QT_SCALE_FACTOR": "1.25"},
            "native, 150 %": {"QT_QPA_PLATFORM": "windows", "QT_SCALE_FACTOR": "1.5"},
        }
    for name, settings in variants.items():
        try:
            done = subprocess.run(
                [sys.executable, "-c", PROBE, str(Path(__file__).parent), name], capture_output=True, timeout=120,
                encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8", **settings})
            told = f"exit code {done.returncode}\n{done.stdout}\nstderr:\n{done.stderr}"
        except Exception as error:  # a probe never fails the run: it only tells
            told = repr(error)
        warnings.warn(f"PROBE {name} {settings}\n{told[:30000]}", stacklevel=1)
