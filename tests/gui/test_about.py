"""Help > About (SPEC 10.1: "About with all versions"; task C8a).

Expected values come from where each version is defined, not from the About text: the tool's line
is `provenance.tool_version()`; Python's version is the running interpreter's; PySide6, pyqtgraph
and numpy say their own version as `__version__` (they are loaded here anyway); the versions of
OpenCV, torch and transformers are read from the installed packages' metadata, since no test of
the window may import torch. The model's folders are where the segmenter keeps EdgeTAM
(`$SHRIMP_MODEL_CACHE/edgetam`) and where the Hugging Face Hub keeps SAM 2.1.
"""

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy
import PySide6
import pyqtgraph

from finish_helpers import record_every_dialog
from outline_tracker import provenance
from outline_tracker.gui import about

REPO = Path(__file__).resolve().parents[2]
NAMES = ("Python", "PySide6", "pyqtgraph", "numpy", "OpenCV", "torch", "transformers")


def installed(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not installed"


def test_about_lists_the_tool_and_every_library_with_its_version():
    facts = dict(about.facts())
    assert list(facts)[:8] == ["Outline Tracker", *NAMES]
    assert facts["Outline Tracker"] == provenance.tool_version()
    assert facts["Python"] == platform.python_version()
    assert facts["PySide6"] == PySide6.__version__
    assert facts["pyqtgraph"] == pyqtgraph.__version__
    assert facts["numpy"] == numpy.__version__
    assert facts["OpenCV"] == installed("opencv-python-headless")
    assert facts["torch"] == installed("torch")
    assert facts["transformers"] == installed("transformers")
    assert all(value.strip() for value in facts.values())  # none is empty


def test_about_says_where_the_model_is_stored(monkeypatch, tmp_path):
    from huggingface_hub import constants

    monkeypatch.setenv("SHRIMP_MODEL_CACHE", str(tmp_path / "models"))
    facts = dict(about.facts())
    assert facts["EdgeTAM is stored in"] == str(tmp_path / "models" / "edgetam")
    assert facts["SAM 2.1 is stored in"] == str(constants.HF_HUB_CACHE)


def test_the_text_has_a_heading_and_one_line_for_each_fact():
    heading, *lines = about.about_text().splitlines()
    assert heading == "About Outline Tracker"
    assert lines == [f"{name}: {value}" for name, value in about.facts()]


def test_help_about_shows_the_text_in_a_dialog(window, monkeypatch):
    asked = record_every_dialog(monkeypatch)
    window.menus.about_action.trigger()
    assert asked.messages == [(window, "info", about.about_text())]
    for name in NAMES:
        assert f"\n{name}: " in asked.messages[0][2]


def test_about_reads_the_versions_without_loading_torch():
    probe = ("import json, sys\n"
             "from outline_tracker.gui import about\n"
             "text = about.about_text()\n"
             "print(json.dumps({'torch': 'torch' in sys.modules, 'transformers': 'transformers' in sys.modules,\n"
             "                  'text': text}))\n")
    done = subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                          timeout=120)
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    assert seen["torch"] is False and seen["transformers"] is False
    assert f"torch: {installed('torch')}" in seen["text"].splitlines()
