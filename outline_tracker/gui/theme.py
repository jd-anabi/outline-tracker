"""The look of the window: colour tokens for light and dark, one palette and one style sheet.

The style is Qt's Fusion on every system, so the window is drawn the same on Windows and macOS and
follows the palette. From the system come only the choice of light or dark, the font, and the file
dialogs; every colour is fixed here, so that each pair of text and background keeps a contrast of
4.5:1 or more (tests/gui/test_theme.py recomputes it). A colour is `#RRGGBB`. Sizes in the style
sheet are Qt's device-independent px.

The style sheet styles widgets by the dynamic properties that gui/panel.py sets (`role`, `state`).
Qt reads a property for a style sheet only when a widget is polished: change one with
`set_property`. Style sheets have no variables: `fill` puts the theme's colour in for each `@name`.
Nothing here creates a Qt object when the module is imported.
"""

from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget

LIGHT = {
    "window": "#F1F3F5",         # window, dock background, status bar
    "panel": "#FFFFFF",          # panel surface
    "field": "#FFFFFF",          # inputs, tables, log
    "control": "#F7F8FA",        # spin-box arrows, combo boxes, table headers
    "hover": "#E8EBEF",
    "pressed": "#DCE0E6",
    "border": "#D0D5DC",         # panel outline, separators (decoration only)
    "borderStrong": "#7C8591",   # outline of buttons and of the badge
    "text": "#1B1F24",
    "muted": "#57606A",          # hints, read-outs
    "disabled": "#8C959F",
    "accent": "#1F5FBF",         # primary button, selection, focus ring
    "onAccent": "#FFFFFF",
    "accentHover": "#1A52A6",
    "accentPressed": "#164690",
    "accentSoft": "#DCE8FA",     # checked tool button
    "success": "#17753A",        # done, a check passes
    "successBg": "#E3F4E7",
    "warning": "#8A5A00",        # needs attention
    "warningBg": "#FFF3D1",
    "problem": "#B42318",        # problems, destructive buttons
    "problemBg": "#FDE7E4",
    "canvas": "#1B1D20",         # around the video, the same in both themes
    "onCanvas": "#E7EAEE",
}

DARK = {
    "window": "#1E2126",
    "panel": "#272B31",
    "field": "#1A1D21",
    "control": "#30353C",
    "hover": "#30353C",
    "pressed": "#3A4048",
    "border": "#3A4048",
    "borderStrong": "#7D8793",
    "text": "#E7EAEE",
    "muted": "#A3ADB8",
    "disabled": "#6E7781",
    "accent": "#7AB0FF",
    "onAccent": "#0D1B2E",
    "accentHover": "#94C0FF",
    "accentPressed": "#5F9BEE",
    "accentSoft": "#253B5C",
    "success": "#5FD080",
    "successBg": "#1C3324",
    "warning": "#F2BD4B",
    "warningBg": "#3A2E12",
    "problem": "#FF8E84",
    "problemBg": "#40211F",
    "canvas": "#1B1D20",
    "onCanvas": "#E7EAEE",
}

# The rules of the widgets that exist so far. A task that adds a kind of widget adds its rules.
STYLE = """
#Dock, #DockColumn { background: @window; }
QLabel#VideoArea { background: @canvas; color: @onCanvas; }
QFrame[role="panel"] { background: @panel; border: 1px solid @border; border-radius: 6px; }
QFrame[role="panel"][state="attention"] { border-color: @warning; }
QToolButton[role="panelHeader"] { border: none; background: transparent; color: @text; padding: 6px 4px; }
QToolButton[role="panelHeader"]:hover { background: @hover; border-radius: 4px; }
QToolButton[role="panelHeader"]:focus { border: 2px solid @accent; border-radius: 4px; padding: 4px 2px; }
QLabel[role="badge"] { border-radius: 10px; border: 1px solid @borderStrong; color: @text; font-weight: 600; }
QLabel[role="badge"][state="done"] { background: @success; border-color: @success; color: @panel; }
QLabel[role="badge"][state="attention"] { background: @warningBg; border-color: @warning; color: @warning; }
QLabel[role="state"], QLabel[role="hint"] { border: none; background: transparent; color: @muted; }
QLabel[role="state"][state="done"] { color: @success; }
QLabel[role="state"][state="attention"] { color: @warning; font-weight: 600; }
"""


def tokens(dark: bool) -> dict[str, str]:
    """The colour tokens of the dark theme (`dark` true) or the light one: name -> `#RRGGBB`."""
    return DARK if dark else LIGHT


def fill(template: str, colors: dict[str, str]) -> str:
    """`template` with every `@name` replaced by `colors[name]` (`#RRGGBB`). A name is letters only
    and is taken whole (`@borderStrong` is not `@border`); a name that is no token is a `KeyError`."""
    return re.sub(r"@([A-Za-z]+)", lambda found: colors[found.group(1)], template)


def style_sheet(colors: dict[str, str]) -> str:
    """The application's style sheet in the colours of one theme (a table from `tokens`).
    Lengths in it are device-independent px."""
    return fill(STYLE, colors)


def palette(colors: dict[str, str]) -> QPalette:
    """The palette of one theme (a table from `tokens`): Fusion draws the inputs from it.

    The shades Fusion uses for edges come from the `control` and `window` colours; each role with a
    meaning gets its token. Disabled text is the `disabled` colour.
    """
    role = QPalette.ColorRole
    made = QPalette(QColor(colors["control"]), QColor(colors["window"]))
    by_role = {
        role.Window: "window", role.WindowText: "text", role.Text: "text", role.ButtonText: "text",
        role.Base: "field", role.AlternateBase: "window", role.Button: "control",
        role.Highlight: "accent", role.Accent: "accent", role.Link: "accent", role.HighlightedText: "onAccent",
        role.ToolTipBase: "panel", role.ToolTipText: "text", role.PlaceholderText: "muted",
    }
    for which, name in by_role.items():
        made.setColor(which, QColor(colors[name]))  # every group: active, inactive, disabled
    for which in (role.WindowText, role.Text, role.ButtonText):
        made.setColor(QPalette.ColorGroup.Disabled, which, QColor(colors["disabled"]))
    return made


def apply(app: QApplication, dark: bool) -> None:
    """Give the application the Fusion style and the palette and style sheet of the dark theme
    (`dark` true) or the light one. Every widget, shown or not yet made, is drawn with them."""
    colors = tokens(dark)
    app.setStyle("Fusion")
    app.setPalette(palette(colors))
    app.setStyleSheet(style_sheet(colors))


def on_color_scheme_changed(scheme: Qt.ColorScheme) -> None:
    """Apply the theme for `scheme` to the running application: dark for `Qt.ColorScheme.Dark`, light
    for everything else (also when the system does not say)."""
    apply(QApplication.instance(), scheme == Qt.ColorScheme.Dark)


def follow_system(app: QApplication) -> None:
    """Apply the theme of the system's light or dark setting now, and again whenever it changes."""
    hints = app.styleHints()
    on_color_scheme_changed(hints.colorScheme())
    hints.colorSchemeChanged.connect(on_color_scheme_changed)


def set_property(widget: QWidget, name: str, value: str) -> None:
    """Set the dynamic property `name` of `widget` and polish it again, so that the style sheet rules
    that select by this property (`[state="done"]`) are applied at once."""
    widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()
