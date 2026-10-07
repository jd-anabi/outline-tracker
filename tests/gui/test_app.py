"""The window's entry point, `outline_tracker.gui.app.main`, run for real (SPEC 10.2; task C0).

`main` creates the application, so it cannot run in this test process (pytest-qt has made one
already): each test starts a fresh process, with the real PySide6 on the offscreen platform. Two
things are replaced there. torch is an empty stand-in, so the real one is never loaded (the entry
point only imports it, for the order of SPEC 10.2; that order is tests/gui/test_import_order.py).
`QApplication.exec`, the event loop, is a function that writes down what `main` has built, closes
the window and returns an exit code, so the process ends by itself.

The expected values: the nine panels of SPEC 10.1, the title, the factory that makes the real
segmenter for the command line (`from_tracker.load_segmenter`), and a status bar that names the
file the window was started with (its name, never its folder).
"""

import json
import subprocess
import sys
from pathlib import Path

import helpers

REPO = Path(__file__).resolve().parents[2]
EXIT_CODE = 5  # what the stand-in event loop returns: not 0, so that a constant 0 is not taken for it

_PROBE = """
import importlib.abc, importlib.machinery, json, os, sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"  # never a window on a real screen


class TorchStandIn(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    # An empty module for torch and for every name under it.
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] != "torch":
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return None  # Python's own empty module

    def exec_module(self, module):
        pass


stand_in = TorchStandIn()
sys.meta_path.insert(0, stand_in)

from PySide6 import QtWidgets  # loaded here only to replace the event loop before `main` runs

seen = {"event_loops": 0}


def event_loop(*_):
    # In place of QApplication.exec: write down what `main` has built, close it, end with a code.
    from outline_tracker import from_tracker
    from outline_tracker.gui import theme
    from outline_tracker.gui.main_window import MainWindow

    application = QtWidgets.QApplication.instance()
    windows = [widget for widget in application.topLevelWidgets() if isinstance(widget, MainWindow)]
    seen["event_loops"] += 1
    seen["windows"] = len(windows)
    seen["themed"] = application.styleSheet() in (theme.style_sheet(theme.LIGHT), theme.style_sheet(theme.DARK))
    seen["torch_is_the_stand_in"] = getattr(sys.modules.get("torch"), "__loader__", None) is stand_in
    for window in windows:
        seen["title"] = window.windowTitle()
        seen["visible"] = window.isVisible()
        seen["maximized"] = window.isMaximized()
        seen["content"] = [window.width(), window.height()]
        area = window.screen().availableGeometry()
        seen["screen_offers"] = [area.width(), area.height()]
        seen["panels"] = [panel.number for panel in window.panels]
        seen["factory_is_load_segmenter"] = window.segmenter_factory is from_tracker.load_segmenter
        seen["status"] = window.statusBar().currentMessage()
        window.close()
        seen["closed"] = not window.isVisible()
    return int(sys.argv[1])


QtWidgets.QApplication.exec = event_loop

from outline_tracker.gui import app

seen["returned"] = app.main(sys.argv[2:])  # what the `gui` command hands on: nothing, or one path
print(json.dumps(seen))
"""


def run_main(argv: list[str]) -> dict:
    """What `app.main(argv)` built and returned in a fresh process (the probe above)."""
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, str(EXIT_CODE), *argv],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def assert_the_window_was_built(seen: dict) -> None:
    assert seen["event_loops"] == 1, seen  # `main` ran the application's event loop, once
    assert seen["torch_is_the_stand_in"]  # the real torch was not loaded for this test
    assert seen["windows"] == 1, seen
    assert seen["title"] == "Outline Tracker"
    assert seen["visible"]  # shown before the event loop starts
    # ... and as the window starts (tests/gui/test_shell.py): maximized, unless the screen of this
    # run has room for 1440 x 900 and its frame (the offscreen screen is 800 x 800)
    width, height = seen["screen_offers"]
    room = width >= 1456 and height >= 940
    assert seen["maximized"] == (not room)
    assert not room or seen["content"] == [1440, 900]
    assert seen["panels"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert seen["factory_is_load_segmenter"]  # the real model's factory, kept for the worker
    assert seen["themed"]  # the application has the style sheet of the light or of the dark theme
    assert seen["closed"]
    assert seen["returned"] == EXIT_CODE  # the application's exit code is the entry point's


def test_the_entry_point_shows_the_window_and_returns_the_applications_exit_code():
    seen = run_main([])
    assert_the_window_was_built(seen)
    assert seen["status"] == ""  # no file was named


def test_the_entry_point_hands_a_named_file_to_the_window(tmp_path):
    folder = tmp_path / helpers.ODD_FOLDER  # a space and non-ASCII characters
    seen = run_main([str(folder / "dish_tracker.mp4")])
    assert_the_window_was_built(seen)
    assert "dish_tracker.mp4" in seen["status"]
    assert helpers.ODD_FOLDER not in seen["status"] and str(tmp_path) not in seen["status"]
