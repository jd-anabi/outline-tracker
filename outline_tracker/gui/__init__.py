"""The window (SPEC 10): PySide6 and pyqtgraph are imported only in this package.

`app` is the entry point: its `main` imports torch first, then PySide6, then pyqtgraph (SPEC 10.2:
on Windows torch fails to load once Qt is loaded). No other module here imports torch, and none
creates a Qt object when it is imported: a Qt object made before the application exists can end the
process. Nothing is imported by this file itself, so `import outline_tracker.gui` loads no library.

`main_window` is the window with its dock of nine numbered panels, `panel` the frame every panel
shares (number, title, state, hint, body), `theme` the colours (light and dark), the palette and
the style sheet. No measurement and no file format is defined in this package (SPEC 12).
"""
