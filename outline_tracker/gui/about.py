"""Help > About (SPEC 10.1): which tool this is, the versions of what it runs on, and where the
model is stored.

The versions come from the installed packages' metadata (`provenance.library_versions`), so asking
for them loads neither torch nor transformers: the window must stay usable while the model loads,
and a version is also wanted when torch cannot be loaded at all. The model's folders are the ones
the segmenter reads: the converted EdgeTAM in its own cache folder, SAM 2.1 in the Hugging Face
cache. A folder is written with its whole path here, since the path is what is asked for.

Text only: no quantities, so no units and no coordinate frame.
"""

from __future__ import annotations

import platform

from outline_tracker import provenance
from outline_tracker.gui import dialogs
from outline_tracker.segmenter.edgetam_convert import edgetam_cache

HEADING = "About Outline Tracker"
# The name shown for each library, and the distribution whose metadata holds its version.
LIBRARIES = {"PySide6": "PySide6-Essentials", "pyqtgraph": "pyqtgraph", "numpy": "numpy",
             "OpenCV": "opencv-python-headless", "torch": "torch", "transformers": "transformers"}


def facts() -> list[tuple[str, str]]:
    """What About tells, as (name, value) pairs in the order shown: the tool's version line
    (`provenance.tool_version`), the versions of Python, PySide6, pyqtgraph, numpy, OpenCV, torch
    and transformers ("not installed" for one that is missing), and the folders the models are
    stored in. Nothing is imported or loaded to find a version."""
    versions = provenance.library_versions(LIBRARIES.values())
    listed = [("Outline Tracker", provenance.tool_version()), ("Python", platform.python_version())]
    listed += [(name, versions[distribution]) for name, distribution in LIBRARIES.items()]
    listed.append(("EdgeTAM is stored in", str(edgetam_cache())))
    try:
        from huggingface_hub import constants  # a small package; it does not load torch
    except ImportError:  # the hub's package is missing: so is every model that would come from it
        return listed
    return [*listed, ("SAM 2.1 is stored in", str(constants.HF_HUB_CACHE))]


def about_text() -> str:
    """The text of the About dialog: the heading, then one line "name: value" for each of `facts`."""
    return "\n".join([HEADING, *(f"{name}: {value}" for name, value in facts())])


def show(parent) -> None:
    """Show About in a dialog over `parent` (a widget) and return at once."""
    dialogs.message(parent, "info", about_text())
