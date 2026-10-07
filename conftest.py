"""Settings for every test run (repo root, loaded before pytest-qt creates anything)."""

import os
import sys

# GUI tests never open windows.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# pytest-qt imports PySide6 when pytest starts. On Windows, torch must be imported before Qt
# (SPEC 10.2), so import it here when it is installed. No fast test needs it.
if sys.platform == "win32":
    try:
        import torch  # noqa: F401
    except Exception:  # not installed, or its DLLs failed to load: fast tests still run
        pass

# Shared fixtures live in tests/helpers.py. No conftest.py may be added below tests/ other than
# tests/conftest.py: the ported tests import helpers from it by name.
pytest_plugins = ["helpers"]
