"""The application's settings, kept between sittings: `settings()` gives the `QSettings` of the tool.

Two things are remembered there. The time panel (gui/panels/time_panel.py) keeps the manifest the user
pointed to last (`MANIFEST_KEY`), and the overlays (gui/view/overlays.py) keep whether Fill was on (`FILL_KEY`).
Each of them owns its key; this module owns where the settings are stored. No quantities here.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings


def settings() -> QSettings:
    """The application's settings, kept between sessions: an INI file in the user's own settings
    folder, the same kind of file on every system."""
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "outline-tracker", "outline-tracker")
