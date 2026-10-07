"""Import rules of the window's code (SPEC 10.2, 12; task C0), each checked in a fresh process.

- The launcher asks for torch before anything of PySide6, and for pyqtgraph after PySide6. A
  recording import hook serves empty stand-ins for the three, so no real torch is needed here (the
  real libraries in this order are a step of the Windows CI job). This is checked twice: for the
  import step alone (`app.import_in_order`), and for the entry point itself (`app.main`), which
  must take that step before it touches anything of Qt. With empty stand-ins `main` cannot build
  a window (it ends with an error where the window's code first uses a Qt class); what it builds
  with the real PySide6 is in tests/gui/test_app.py.
- Importing the launcher's module asks for none of them: the imports are inside its functions.
- Every module of `outline_tracker.gui` imports without torch, creates no application, leaves no Qt
  object in a module or in a class body, and makes Qt say nothing.

This test process has PySide6 loaded already (pytest-qt), so nothing about imports can be seen in it.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_RECORDER = """
import importlib.abc, importlib.machinery, json, sys

STAND_INS = ("torch", "PySide6", "pyqtgraph")
asked = []  # the stand-ins' names, in the order the import system asked for them


class Recorder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] not in STAND_INS:
            return None
        asked.append(fullname)
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return None  # Python's own empty module

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, Recorder())
from outline_tracker.gui import app

at_import = list(asked)
"""

_ORDER_PROBE = _RECORDER + """
app.import_in_order()
print(json.dumps({"at_import": at_import, "asked": asked}))
"""

# The entry point itself. A stand-in is an empty package that serves every name asked of it as one
# more empty module, so `main` imports its way to the first use of a Qt class (a class with a Qt
# base) and ends there with an error. What was asked for until then is the order of its imports.
_MAIN_PROBE = _RECORDER + """
try:
    ended = "returned %r" % (app.main([]),)
except Exception as error:
    ended = type(error).__name__ + ": " + str(error)
print(json.dumps({"at_import": at_import, "asked": asked, "ended": ended}))
"""

_PACKAGE_PROBE = """
import importlib, json, pkgutil, sys
from PySide6 import QtCore
import shiboken6

messages = []
QtCore.qInstallMessageHandler(lambda mode, context, text: messages.append(text))
import outline_tracker.gui as gui

names = ["outline_tracker.gui"] + [m.name for m in pkgutil.walk_packages(gui.__path__, "outline_tracker.gui.")]
objects = []
for name in names:
    module = importlib.import_module(name)
    for key, value in list(vars(module).items()):
        held = [(key, value)]
        if isinstance(value, type) and value.__module__ == name:
            # staticMetaObject is what the binding itself adds to every class of a Qt widget
            held += [(key + "." + k, v) for k, v in vars(value).items() if k != "staticMetaObject"]
        objects += [name + "." + k for k, v in held if isinstance(v, shiboken6.Object)]
print(json.dumps({
    "modules": names,
    "torch": "torch" in sys.modules,
    "application": QtCore.QCoreApplication.instance() is not None,
    "objects": objects,
    "messages": messages,
    "scan_sees_a_qt_object": isinstance(QtCore.QSize(1, 2), shiboken6.Object),
}))
"""


def _probe(script: str) -> dict:
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=120
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_launcher_asks_for_torch_before_pyside6_and_for_pyqtgraph_last():
    seen = _probe(_ORDER_PROBE)
    assert seen["at_import"] == []  # importing the module loads nothing: the imports are in its functions
    tops = [name.split(".")[0] for name in seen["asked"]]
    assert set(tops) == {"torch", "PySide6", "pyqtgraph"}
    assert tops[0] == "torch"
    assert tops.index("torch") < tops.index("PySide6") < tops.index("pyqtgraph")
    last_pyside = max(i for i, top in enumerate(tops) if top == "PySide6")
    assert last_pyside < tops.index("pyqtgraph")  # pyqtgraph finds the binding already loaded


def test_the_entry_point_asks_for_torch_before_it_touches_anything_of_qt():
    seen = _probe(_MAIN_PROBE)
    assert seen["at_import"] == []
    tops = [name.split(".")[0] for name in seen["asked"]]
    assert tops[:1] == ["torch"], seen  # the first of the three that `main` asks for: nothing of Qt before it
    assert "PySide6" in tops and "pyqtgraph" in tops, seen
    assert tops.index("PySide6") < tops.index("pyqtgraph")
    assert not seen["ended"].startswith("returned"), seen  # empty stand-ins cannot make a window


def test_the_gui_package_imports_without_torch_and_creates_no_qt_object():
    seen = _probe(_PACKAGE_PROBE)
    wanted = {"outline_tracker.gui", "outline_tracker.gui.app", "outline_tracker.gui.main_window",
              "outline_tracker.gui.panel", "outline_tracker.gui.theme"}
    assert wanted <= set(seen["modules"])  # the scan looked at the real package
    assert seen["scan_sees_a_qt_object"]
    assert seen["torch"] is False
    assert seen["application"] is False
    assert seen["objects"] == []
    assert seen["messages"] == []
