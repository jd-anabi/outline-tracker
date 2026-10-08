"""The round number badge of a panel (task C0b; SPEC 10.1): where its digit is drawn, and its fill.

The badge is a circle 20 px across with an outline ring of 1 px; inside it are the fill of the
panel's state and the panel's number (gui/panel.py, the badge rules of gui/theme.py). What a
well-drawn badge has does not depend on the system's font, it is geometry: the digit lies inside
the circle with 2 px or more to the circle's edge, its middle is within 1.5 px of the badge's
middle, and the rest of the circle shows the state's fill, which is a theme token.

The badge is read from the drawn window (`grab`, offscreen), never from the style sheet's text.
Lengths are Qt's device-independent px, x to the right and y down from the badge's top left
corner.
"""

import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from PySide6.QtGui import QFont, QFontDatabase, QFontInfo, QGuiApplication

from gui_helpers import badge_pixels, inside_badge
from outline_tracker.gui import theme

SIZE = 20           # the badge: a circle this many px across
MIDDLE = SIZE / 2   # its centre, px from its left and from its top edge; also the circle's radius
MARGIN = 2          # from the digit's bounding box to the circle's edge: at least this, px
OFF_CENTRE = 1.5    # from the digit's middle to the badge's middle: at most this, px, in x and in y

STATES = ("todo", "done", "attention")
# state -> the tokens of the fill and of the digit (the design note's three badge rules)
TOKENS = {"todo": ("panel", "text"), "done": ("success", "panel"), "attention": ("warningBg", "warning")}

# What a Windows screen gives the application: Segoe UI at 9 points, which is 12 px. The files are
# the regular cut and the semibold one, which the badge's weight of 600 asks for.
WINDOWS_FONT = ("Segoe UI", 9, ("segoeui.ttf", "seguisb.ttf"))


@pytest.fixture
def text_font(qapp):
    """A font for the text of this test where the platform has none; taken away again afterwards.

    Measured on the Windows test machine (task C0b): the offscreen platform looks for font files in
    a folder of Qt's own that PySide6 does not have, knows 0 font families, and draws an empty square
    of 8 x 8 px in place of every character. There the application gets the font it has on a Windows
    screen (`WINDOWS_FONT`), from Windows' font folder; with it the offscreen platform drew the digits
    1 and 9 within 1 px of the boxes that the Windows platform drew. On macOS and Linux the offscreen
    platform has the system's fonts and nothing is done.
    """
    if sys.platform != "win32" or QFontDatabase.families():
        yield
        return
    family, points, files = WINDOWS_FONT
    folder = Path(os.environ["WINDIR"]) / "Fonts"
    loaded = [QFontDatabase.addApplicationFont(str(folder / name)) for name in files]
    before = qapp.font()
    qapp.setFont(QFont(family, points))
    yield
    qapp.setFont(before)
    for font in loaded:
        QFontDatabase.removeApplicationFont(font)
    # Taking a font away makes Qt forget its list of fonts; it reads the list again when text is next
    # drawn, and says once more that it found no font folder. Have that said here, not in a later test.
    QFontDatabase.families()


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
    drawn = badge_pixels(window, badge)
    half = drawn[0][0]  # the first pixel's centre is half a device pixel from the badge's left edge
    of_digit, others, rows = [], Counter(), {}
    for x, y, color in drawn:
        like_digit = nearer(color, digit, fill)
        if not inside_badge(x, y):
            mark = "+" if like_digit else " "
        elif like_digit:
            of_digit.append((x, y))
            mark = "#"
        else:
            others[color] += 1
            mark = "." if color == fill else ":"
        rows[y] = rows.get(y, "") + mark
    box = None
    if of_digit:
        across, down = zip(*of_digit)
        box = (min(across) - half, min(down) - half, max(across) + half, max(down) + half)
    return Reading(box, others.most_common(1)[0][0], "\n".join(rows.values()))


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
def test_the_digit_is_in_the_middle_of_a_circle_of_the_states_fill(state, number, text_font, window, qapp, qtbot,
                                                                   look):
    theme.apply(qapp, False)
    panel = window.panels[number - 1]
    assert panel.badge.text() == str(number)
    assert (panel.badge.width(), panel.badge.height()) == (SIZE, SIZE)
    panel.set_state(state)
    with qtbot.waitExposed(window):
        window.show()
    fill, digit = (theme.LIGHT[token] for token in TOKENS[state])

    seen = read(window, panel.badge, fill, digit)

    assert faults(seen, fill) == [], described(panel.badge, seen)
