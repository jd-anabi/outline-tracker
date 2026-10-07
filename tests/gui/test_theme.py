"""The window's look (task C0; the design note's colour tokens): tokens, palette, style sheet.

Expected values are the note's: its token tables (a few are spelled out here, and the contrast it
states for a pair is recomputed with the WCAG 2 formula), its palette roles, the first rule of its
style-sheet skeleton, and which token colours which part of a panel. Colours are read from the
drawn window (`grab`, offscreen), never from the style sheet's text. Qt's own messages are
collected by the `qt_messages` fixture: a rule Qt cannot read is a message (one test shows that
such a message does arrive there).
"""

import math
import re
from collections import Counter

import pytest
from PySide6.QtCore import QPoint, Qt, qInstallMessageHandler
from PySide6.QtGui import QPalette

from outline_tracker.gui import theme

BOTH = pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])


def luminance(color: str) -> float:
    """Relative luminance of `#RRGGBB` (WCAG 2)."""
    channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(one: str, other: str) -> float:
    """Contrast ratio of two `#RRGGBB` colours (WCAG 2): 1 to 21."""
    high, low = sorted((luminance(one), luminance(other)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.fixture
def look(qapp):
    """Put the application's style, palette and style sheet back after the test."""
    style, palette, sheet = qapp.style().name(), QPalette(qapp.palette()), qapp.styleSheet()
    yield
    qapp.setStyleSheet(sheet)
    qapp.setPalette(palette)
    if style:  # no name: a style sheet was in place before the test, and its wrapper cannot be asked
        qapp.setStyle(style)


@pytest.fixture
def qt_messages():
    """Every message Qt writes during the test, as text (its warnings about a style sheet among them)."""
    seen = []
    previous = qInstallMessageHandler(lambda mode, context, text: seen.append(text))
    yield seen
    qInstallMessageHandler(previous)


def about_the_look(messages: list[str]) -> list[str]:
    """`messages` without Qt's notice that the system lacks the font family it asked for: that one is
    about the computer's fonts (it comes on macOS, offscreen), not about the style sheet."""
    return [text for text in messages if "font family" not in text]


def color_at(window, widget, point: QPoint) -> str:
    """The colour drawn at `point` of `widget` (its own px), as `#RRGGBB` in capitals."""
    image = window.grab().toImage()
    at = widget.mapTo(window, point) * image.devicePixelRatio()
    return image.pixelColor(at).name().upper()


def badge_pixels(window, badge) -> list[tuple[float, float, str]]:
    """Every device pixel that `window` draws in `badge`'s square, row by row: (x, y, colour). x and y
    are the pixel's centre in px from the badge's top left corner (x right, y down); the colour is
    `#RRGGBB` in capitals."""
    image = window.grab().toImage()
    ratio = image.devicePixelRatio()
    corner = badge.mapTo(window, QPoint(0, 0)) * ratio
    return [((column + 0.5) / ratio, (row + 0.5) / ratio,
             image.pixelColor(corner.x() + column, corner.y() + row).name().upper())
            for row in range(round(badge.height() * ratio)) for column in range(round(badge.width() * ratio))]


def inside_badge(x: float, y: float) -> bool:
    """True for a pixel with its centre at (x, y), px from the top left corner of a badge, that lies
    wholly inside the badge's outline ring. The badge is a circle 20 px across, so its centre is at
    (10, 10) and the 1 px ring's inner edge 9 px from it; half the diagonal of a pixel is 0.71 px."""
    return math.hypot(x - 10, y - 10) < 9 - 0.75


def badge_fill(window, badge) -> str:
    """The most frequent colour inside `badge`'s outline ring, as `#RRGGBB` in capitals: the fill,
    whatever the font of the digit is and wherever the digit is drawn (it covers the smaller part)."""
    return Counter(color for x, y, color in badge_pixels(window, badge) if inside_badge(x, y)).most_common(1)[0][0]


# Why the tests that probe one pixel beside a badge's digit cannot pass on Windows (task C0b).


# ---------------------------------------------------------------------------------------------
# Tokens


def test_light_and_dark_name_the_same_tokens_as_hex_colours():
    assert theme.tokens(False) is theme.LIGHT and theme.tokens(True) is theme.DARK
    assert set(theme.LIGHT) == set(theme.DARK)
    for table in (theme.LIGHT, theme.DARK):
        assert [name for name, color in table.items() if not re.fullmatch(r"#[0-9A-F]{6}", color)] == []
    # around the video the colours are the same in both themes
    assert theme.LIGHT["canvas"] == theme.DARK["canvas"] == "#1B1D20"
    assert theme.LIGHT["onCanvas"] == theme.DARK["onCanvas"] == "#E7EAEE"


def test_the_surfaces_and_the_text_are_the_notes():
    wanted = {"window": ("#F1F3F5", "#1E2126"), "panel": ("#FFFFFF", "#272B31"), "text": ("#1B1F24", "#E7EAEE"),
              "border": ("#D0D5DC", "#3A4048"), "borderStrong": ("#7C8591", "#7D8793"),
              "accent": ("#1F5FBF", "#7AB0FF"), "warning": ("#8A5A00", "#F2BD4B")}
    assert {name: (theme.LIGHT[name], theme.DARK[name]) for name in wanted} == wanted


@pytest.mark.parametrize(
    "front, back, light, dark",
    [  # the ratios the note states for light and dark
        ("text", "panel", 16.6, 11.8),
        ("text", "window", 14.9, 13.4),
        ("muted", "panel", 6.4, 6.3),
        ("muted", "window", 5.7, 7.1),
        ("accent", "panel", 6.1, 6.4),
        ("onAccent", "accent", 6.1, 7.8),
        ("success", "panel", 5.8, 7.3),
        ("success", "successBg", 5.0, 7.0),
        ("warning", "panel", 5.9, 8.2),
        ("warning", "warningBg", 5.4, 7.7),
        ("problem", "panel", 6.6, 6.4),
        ("problem", "problemBg", 5.6, 6.5),
        ("onCanvas", "canvas", 14.0, 14.0),
    ],
)
def test_a_text_pair_has_the_contrast_the_note_states(front, back, light, dark):
    # stated to one decimal, so within 0.05 (and a little for the last digit)
    assert contrast(theme.LIGHT[front], theme.LIGHT[back]) == pytest.approx(light, abs=0.06)
    assert contrast(theme.DARK[front], theme.DARK[back]) == pytest.approx(dark, abs=0.06)


@BOTH
def test_every_text_pair_meets_wcag_aa(dark):
    colors = theme.tokens(dark)
    pairs = [("text", "panel"), ("text", "window"), ("text", "field"), ("text", "control"), ("text", "accentSoft"),
             ("muted", "panel"), ("muted", "window"), ("accent", "panel"),
             ("onAccent", "accent"), ("onAccent", "accentHover"), ("onAccent", "accentPressed"),
             ("success", "panel"), ("success", "successBg"), ("warning", "panel"), ("warning", "warningBg"),
             ("problem", "panel"), ("problem", "problemBg"), ("onCanvas", "canvas"),
             ("text", "successBg"), ("text", "warningBg"), ("text", "problemBg")]
    assert [pair for pair in pairs if contrast(colors[pair[0]], colors[pair[1]]) < 4.5] == []
    assert contrast(colors["borderStrong"], colors["panel"]) >= 3.0  # an outline needs 3:1


# ---------------------------------------------------------------------------------------------
# Style sheet and palette


def test_fill_replaces_whole_token_names():
    colors = {"border": "#111111", "borderStrong": "#222222"}
    filled = theme.fill("a { color: @borderStrong; border-color: @border; } b { color: @border; }", colors)
    assert filled == "a { color: #222222; border-color: #111111; } b { color: #111111; }"


def test_fill_refuses_a_name_that_is_not_a_token():
    with pytest.raises(KeyError, match="nosuch"):
        theme.fill("a { color: @nosuch; }", theme.LIGHT)


@BOTH
def test_the_style_sheet_holds_the_themes_colours(dark):
    colors = theme.tokens(dark)
    sheet = theme.style_sheet(colors)
    assert "@" not in sheet
    panel_rule = (f'QFrame[role="panel"] {{ background: {colors["panel"]}; '
                  f'border: 1px solid {colors["border"]}; border-radius: 6px; }}')
    assert panel_rule in sheet
    assert f'QFrame[role="panel"][state="attention"] {{ border-color: {colors["warning"]}; }}' in sheet


@BOTH
def test_the_palette_gives_each_role_its_token(dark, qapp):
    colors = theme.tokens(dark)
    palette = theme.palette(colors)
    role = QPalette.ColorRole
    wanted = {role.Window: "window", role.WindowText: "text", role.Text: "text", role.ButtonText: "text",
              role.Base: "field", role.AlternateBase: "window", role.Button: "control",
              role.Highlight: "accent", role.Accent: "accent", role.Link: "accent",
              role.HighlightedText: "onAccent", role.ToolTipBase: "panel", role.ToolTipText: "text",
              role.PlaceholderText: "muted"}
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        drawn = {r: palette.color(group, r).name().upper() for r in wanted}
        assert drawn == {r: colors[name] for r, name in wanted.items()}
    disabled = QPalette.ColorGroup.Disabled
    for text_role in (role.WindowText, role.Text, role.ButtonText):
        assert palette.color(disabled, text_role).name().upper() == colors["disabled"]
    assert palette.color(disabled, role.Window).name().upper() == colors["window"]


# ---------------------------------------------------------------------------------------------
# On the window


@BOTH
def test_the_theme_draws_the_window_without_a_qt_message(dark, window, qapp, qtbot, qt_messages, look):
    colors = theme.tokens(dark)
    theme.apply(qapp, dark)
    assert qapp.palette().color(QPalette.ColorRole.Window).name().upper() == colors["window"]
    for panel in window.panels:  # every part is drawn, so every rule of the sheet meets its widget
        panel.set_expanded(True)
    first, done, attention = window.panels[0], window.panels[1], window.panels[2]
    done.set_state("done")
    attention.set_state("attention")
    with qtbot.waitExposed(window):
        window.show()

    column = window.scroll.widget()
    assert color_at(window, first, QPoint(6, first.height() - 6)) == colors["panel"]  # the panel's surface
    assert color_at(window, column, QPoint(50, first.geometry().top() + first.height() + 4)) == colors["window"]
    assert color_at(window, first, QPoint(first.width() // 2, 0)) == colors["border"]  # the 1 px outline
    assert color_at(window, attention, QPoint(attention.width() // 2, 0)) == colors["warning"]
    # the badge is a 20 px circle: 4 px in from its left edge at mid height is inside it, beside the digit
    assert color_at(window, first.badge, QPoint(4, 10)) == colors["panel"]  # outlined only
    assert color_at(window, done.badge, QPoint(4, 10)) == colors["success"]  # filled
    assert color_at(window, attention.badge, QPoint(4, 10)) == colors["warningBg"]
    assert (first.badge.width(), first.badge.height()) == (20, 20)
    assert color_at(window, window.video_area, QPoint(5, 5)) == colors["canvas"]
    assert about_the_look(qt_messages) == []


@BOTH
def test_the_theme_draws_the_window_and_fills_the_badges_without_a_qt_message(dark, window, qapp, qtbot, qt_messages,
                                                                              look):
    # the test above, with a badge's fill read as the most frequent colour inside its ring
    colors = theme.tokens(dark)
    theme.apply(qapp, dark)
    assert qapp.palette().color(QPalette.ColorRole.Window).name().upper() == colors["window"]
    for panel in window.panels:  # every part is drawn, so every rule of the sheet meets its widget
        panel.set_expanded(True)
    first, done, attention = window.panels[0], window.panels[1], window.panels[2]
    done.set_state("done")
    attention.set_state("attention")
    with qtbot.waitExposed(window):
        window.show()

    column = window.scroll.widget()
    assert color_at(window, first, QPoint(6, first.height() - 6)) == colors["panel"]  # the panel's surface
    assert color_at(window, column, QPoint(50, first.geometry().top() + first.height() + 4)) == colors["window"]
    assert color_at(window, first, QPoint(first.width() // 2, 0)) == colors["border"]  # the 1 px outline
    assert color_at(window, attention, QPoint(attention.width() // 2, 0)) == colors["warning"]
    assert badge_fill(window, first.badge) == colors["panel"]  # outlined only
    assert badge_fill(window, done.badge) == colors["success"]  # filled
    assert badge_fill(window, attention.badge) == colors["warningBg"]
    assert (first.badge.width(), first.badge.height()) == (20, 20)
    assert color_at(window, window.video_area, QPoint(5, 5)) == colors["canvas"]
    assert about_the_look(qt_messages) == []


def test_a_rule_qt_cannot_read_is_a_message(window, qapp, qtbot, qt_messages, look, monkeypatch):
    # the check above is worth something only if such a message reaches `qt_messages`
    monkeypatch.setattr(theme, "STYLE", theme.STYLE + 'QLabel[role="hint"] { colr: red; }\n')
    theme.apply(qapp, False)
    with qtbot.waitExposed(window):
        window.show()
    assert "Unknown property colr" in about_the_look(qt_messages)


def test_the_style_is_fusion_on_every_system(qapp, look):
    theme.apply(qapp, False)
    qapp.setStyleSheet("")  # under a style sheet Qt hands out its style-sheet wrapper, which has no name
    assert qapp.style().name() == "fusion"


def test_a_state_change_is_drawn_at_once(window, qapp, qtbot, look):
    theme.apply(qapp, False)
    with qtbot.waitExposed(window):
        window.show()
    panel = window.panels[0]
    assert color_at(window, panel, QPoint(panel.width() // 2, 0)) == theme.LIGHT["border"]
    panel.set_state("attention")
    assert color_at(window, panel, QPoint(panel.width() // 2, 0)) == theme.LIGHT["warning"]
    panel.set_state("done")
    assert color_at(window, panel, QPoint(panel.width() // 2, 0)) == theme.LIGHT["border"]
    assert color_at(window, panel.badge, QPoint(4, 10)) == theme.LIGHT["success"]


def test_a_state_change_is_drawn_at_once_on_the_edge_and_in_the_badge(window, qapp, qtbot, look):
    # the test above, with the badge's fill read as the most frequent colour inside its ring
    theme.apply(qapp, False)
    with qtbot.waitExposed(window):
        window.show()
    panel = window.panels[0]
    assert color_at(window, panel, QPoint(panel.width() // 2, 0)) == theme.LIGHT["border"]
    panel.set_state("attention")
    assert color_at(window, panel, QPoint(panel.width() // 2, 0)) == theme.LIGHT["warning"]
    panel.set_state("done")
    assert color_at(window, panel, QPoint(panel.width() // 2, 0)) == theme.LIGHT["border"]
    assert badge_fill(window, panel.badge) == theme.LIGHT["success"]


def test_the_title_keeps_its_font_under_the_style_sheet(window, qapp, qtbot, look):
    base = qapp.font().pointSizeF()
    theme.apply(qapp, False)
    with qtbot.waitExposed(window):
        window.show()
    assert window.panels[0].header.font().pointSizeF() == base + 1
    assert window.panels[0].hint.font().pointSizeF() == base  # nothing is smaller than the base size


def test_the_theme_follows_the_systems_light_or_dark_setting(qapp, look):
    def window_color():
        return qapp.palette().color(QPalette.ColorRole.Window).name().upper()

    hints = qapp.styleHints()
    theme.follow_system(qapp)
    try:
        at_start = theme.DARK if hints.colorScheme() == Qt.ColorScheme.Dark else theme.LIGHT
        assert window_color() == at_start["window"]
        assert qapp.styleSheet() == theme.style_sheet(at_start)
        hints.colorSchemeChanged.emit(Qt.ColorScheme.Dark)
        assert window_color() == "#1E2126"
        assert qapp.styleSheet() == theme.style_sheet(theme.DARK)
        hints.colorSchemeChanged.emit(Qt.ColorScheme.Light)
        assert window_color() == "#F1F3F5"
        hints.colorSchemeChanged.emit(Qt.ColorScheme.Dark)
        hints.colorSchemeChanged.emit(Qt.ColorScheme.Unknown)  # a system that does not say: light
        assert window_color() == "#F1F3F5"
    finally:
        hints.colorSchemeChanged.disconnect(theme.on_color_scheme_changed)
