"""The window's entry point (SPEC 10.2): `main` starts the application.

Import order is the reason this module exists as it is. With PyTorch on Windows, importing torch
after PySide6 fails (WinError 1114, c10.dll), so torch is imported first, in the main thread, before
anything of Qt, on every system; pyqtgraph comes after PySide6, so it uses that binding. The three
imports are in `import_in_order`, and nothing is imported from them at the top of this file:
importing this module loads no library (tests/gui/test_import_order.py checks both, and that `main`
takes this step before anything of Qt). This is the only place outside outline_tracker/segmenter
that imports torch. tests/gui/test_app.py runs `main` in a fresh process.
"""

from __future__ import annotations

from pathlib import Path


def import_in_order():
    """Import torch, then PySide6, then pyqtgraph, and return the `PySide6.QtWidgets` module.

    Call it before anything else of the window is imported. No quantities, so no units or frame.
    """
    import torch  # noqa: F401  first: after Qt it does not load on Windows (SPEC 10.2)
    from PySide6 import QtWidgets

    import pyqtgraph  # noqa: F401  after PySide6: it takes the Qt binding that is loaded

    return QtWidgets


def main(argv: list[str] | None = None) -> int:
    """Open the window, run until it is closed, and return the application's exit code (0 = closed
    normally).

    `argv` holds what the `gui` command was given: nothing, or the path of one video or session
    file, which is handed to the window. The theme follows the system's light or dark setting.
    No quantities here, so no units or frame.
    """
    QtWidgets = import_in_order()
    from outline_tracker.from_tracker import load_segmenter  # makes the real model when the worker asks
    from outline_tracker.gui import theme
    from outline_tracker.gui.main_window import MainWindow

    application = QtWidgets.QApplication(["outline-tracker"])
    theme.follow_system(application)
    window = MainWindow(segmenter_factory=load_segmenter)
    if argv:
        window.open_path(Path(argv[0]))
    window.show_at_start()
    return application.exec()
