"""The controls of the nine panels (SPEC 10.1): one module per panel, found by its name.

`PANEL_MODULES` gives each panel's number the name of its module in this package. A panel's task
adds that module, with one function, `build(window)`, which returns the widget that holds the
panel's controls; nothing else has to be edited for the panel to show them. The window offers a
module what it works with: `window.controller` (the open video and its session),
`window.view` (the video view: tools, clicks, graphics), `window.navigation` (the bottom bar) and
`window.panels` (the nine `Panel`s, for a panel's own state and hint line).

Importing this package imports none of the panels' modules; `build_bodies` does. No quantities
here, so no units and no coordinate frame.
"""

from __future__ import annotations

import importlib
import importlib.util

PANEL_MODULES = {1: "video_panel", 2: "time_panel", 3: "calibration_panel", 4: "dish_panel",
                 5: "probes_panel", 6: "objects_panel", 7: "track_panel", 8: "review_panel", 9: "export_panel"}


def build_bodies(window) -> None:
    """Put each panel's controls into the body of `window.panels`, under the hint line.

    For every number whose module `outline_tracker.gui.panels.<name>` exists, that module's
    `build(window)` is called and the widget it returns is added to the panel's body. A panel
    whose module does not exist keeps its hint line alone. Whether a module exists is asked
    without importing it, so an error raised while it is imported, or inside `build`, is the
    caller's to see: it is never taken for a missing module.
    """
    for number, name in PANEL_MODULES.items():
        full_name = f"{__name__}.{name}"
        if importlib.util.find_spec(full_name) is None:
            continue
        module = importlib.import_module(full_name)
        window.panels[number - 1].body.addWidget(module.build(window))
