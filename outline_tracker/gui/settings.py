"""`settings()` gives the application's `QSettings`, kept between sittings (the manifest's place, Fill).

Two things are remembered there. The time panel (gui/panels/time_panel.py) keeps the manifest the user
pointed to last (`MANIFEST_KEY`), and the overlays (gui/view/overlays.py) keep whether Fill was on (`FILL_KEY`).
This module owns where the settings are stored, and `MANIFEST_KEY`, so that a part that is not a panel can read
it; the overlays own their key. No quantities here.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings

MANIFEST_KEY = "manifest_path"  # the manifest the user pointed to last, kept by the time panel


def settings() -> QSettings:
    """The application's settings, kept between sessions: an INI file in the user's own settings
    folder, the same kind of file on every system."""
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "outline-tracker", "outline-tracker")
