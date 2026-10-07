"""How a panel's task plugs its controls into the window (task C1): `gui/panels`.

`PANEL_MODULES` names one module per panel. `build_bodies(window)` calls `build(window)` of every
module that exists and puts the widget it returns into that panel's body, under the hint line; a
panel whose module does not exist keeps its hint line alone; an error inside a module that exists
is not hidden. The table is the plan's.

These tests must not depend on which panels are built so far: for each of them the package looks
for its modules in a folder of the test (`panel_folder`), which is empty until the test writes one.
"""

import importlib
import sys

import pytest
from PySide6.QtWidgets import QLabel

from outline_tracker.gui import panels

BUILDS_A_LABEL = '''
from PySide6.QtWidgets import QLabel

built = []


def build(window):
    widget = QLabel("the controls of this panel")
    built.append((window, widget))
    return widget
'''


@pytest.fixture
def panel_folder(tmp_path, monkeypatch):
    """A folder in which the `panels` package looks for its modules during the test, and nowhere
    else. Returns `write(name, source)`, which puts a module there and returns its full name."""
    names = [f"{panels.__name__}.{name}" for name in panels.PANEL_MODULES.values()]
    set_aside = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    monkeypatch.setattr(panels, "__path__", [str(tmp_path)])

    def write(name: str, source: str) -> str:
        (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
        importlib.invalidate_caches()  # the folder was looked at before this file was in it
        return f"{panels.__name__}.{name}"

    yield write
    for name in names:
        sys.modules.pop(name, None)
        if hasattr(panels, name.rsplit(".", 1)[1]):
            delattr(panels, name.rsplit(".", 1)[1])
    sys.modules.update(set_aside)
    for name, module in set_aside.items():
        setattr(panels, name.rsplit(".", 1)[1], module)
    importlib.invalidate_caches()


@pytest.fixture
def calibration_module(panel_folder):
    """The module of panel 3, written before the window is made; returns its full name."""
    return panel_folder("calibration_panel", BUILDS_A_LABEL)


def body_widgets(panel) -> list:
    """The widgets in a panel's body, top to bottom."""
    return [panel.body.itemAt(index).widget() for index in range(panel.body.count())]


def test_the_table_names_one_module_for_each_of_the_nine_panels():
    assert panels.PANEL_MODULES == {
        1: "video_panel", 2: "time_panel", 3: "calibration_panel", 4: "dish_panel", 5: "probes_panel",
        6: "objects_panel", 7: "track_panel", 8: "review_panel", 9: "export_panel",
    }


def test_a_panel_without_a_module_keeps_its_hint_line(panel_folder, window):
    for panel in window.panels:
        assert body_widgets(panel) == [panel.hint]
        assert panel.hint.text() != ""


def test_the_window_puts_a_modules_widget_into_its_panel(calibration_module, window):
    (given, widget), = sys.modules[calibration_module].built  # built once, when the window was made
    assert given is window
    # the module was offered the parts it works with
    assert all(hasattr(given, part) for part in ("controller", "view", "navigation", "panels"))
    calibration = window.panels[2]
    assert calibration.number == 3
    assert body_widgets(calibration) == [calibration.hint, widget]  # under the hint line
    assert isinstance(widget, QLabel) and widget.text() == "the controls of this panel"
    for panel in window.panels:
        if panel is not calibration:
            assert body_widgets(panel) == [panel.hint]


def test_build_bodies_fills_each_panel_whose_module_exists(panel_folder, window):
    first, last = panel_folder("video_panel", BUILDS_A_LABEL), panel_folder("export_panel", BUILDS_A_LABEL)
    panels.build_bodies(window)
    for name, number in ((first, 1), (last, 9)):
        (given, widget), = sys.modules[name].built
        assert given is window
        assert body_widgets(window.panels[number - 1]) == [window.panels[number - 1].hint, widget]
    assert all(body_widgets(panel) == [panel.hint] for panel in window.panels[1:8])


def test_an_error_inside_build_is_not_hidden(panel_folder, window):
    panel_folder("time_panel", "def build(window):\n    raise RuntimeError('the time panel is broken')\n")
    with pytest.raises(RuntimeError, match="the time panel is broken"):
        panels.build_bodies(window)


def test_an_error_while_a_module_is_imported_is_not_taken_for_a_missing_module(panel_folder, window):
    panel_folder("dish_panel", "import a_module_that_does_not_exist_anywhere\n\n\ndef build(window):\n    pass\n")
    with pytest.raises(ModuleNotFoundError, match="a_module_that_does_not_exist_anywhere"):
        panels.build_bodies(window)


def test_a_module_without_build_is_an_error(panel_folder, window):
    panel_folder("objects_panel", "nothing = None\n")
    with pytest.raises(AttributeError, match="build"):
        panels.build_bodies(window)
