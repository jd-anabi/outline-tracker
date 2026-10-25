"""The worker when the model is changed (task C8b; SPEC 10.1, 15; X7): its two files, its device, and the outlines
asked for while another model loads.

- gui/worker.py was split: the engine of the thread is in gui/worker_engine.py, and what
  gui/worker.py offered is still importable from it.
- The worker's device is the loaded model's; of two loads asked for in a row only the second makes
  the worker ready; an outline asked for, or awaited, when another model is asked for is made by
  that model. Which stand-in made an outline is read from where the outline is: a `ThresholdFake`
  outlines the disk under the click, an `ExactFake` the scene's object of the track's name wherever
  the click is (the disk scene's centers are its ground truth).

The choice in panel 7, what is on the picture and the estimate are in tests/gui/test_model_choice.py.

No test here imports torch: the factory is a stand-in.
"""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QCoreApplication

from helpers import click
from last_controls_helpers import OnDevice
from prompt_helpers import Gate, Watched, asked_of

# What gui/worker.py defined before it was split (the module as of task C5), private names too.
OFFERED = ["STACK_BYTES", "NO_FACTORY", "REASON_LENGTH", "TRACES_KEPT", "log", "Preview", "_Request", "Found",
           "found_in", "plain", "traced", "before_load_for", "_Engine", "Worker", "worker_of"]


# ---------------------------------------------------------------------------------------------
# The worker: its two files, and the device of the loaded model


def test_every_name_the_worker_module_offered_is_still_importable_from_it():
    from outline_tracker.gui import worker

    assert [name for name in OFFERED if not hasattr(worker, name)] == []
    assert worker.log.name == "outline_tracker.gui.worker"  # the log the traces of failures go to


def test_the_engine_is_in_its_own_file_and_both_files_are_under_400_lines():
    from outline_tracker.gui import worker, worker_engine

    assert worker._Engine is worker_engine._Engine and worker._Engine.__module__ == worker_engine.__name__
    for module in (worker, worker_engine):
        lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
        assert len(lines) < 400, f"{Path(module.__file__).name} has {len(lines)} lines"


def test_the_workers_device_is_none_before_a_model_is_loaded_and_the_models_afterwards(worker, qtbot):
    from helpers import stand_in_segmenter

    assert worker.device is None
    worker.start(lambda model, device: OnDevice("cpu"), "edgetam", "auto")
    qtbot.waitUntil(lambda: worker.ready)
    assert worker.device == "cpu"
    worker.start(stand_in_segmenter, "edgetam", "auto")  # a stand-in that names no device
    qtbot.waitUntil(lambda: worker.ready and worker.device is None)
    worker.stop()
    assert worker.device is None


def test_of_two_loads_asked_for_in_a_row_only_the_second_makes_the_worker_ready(worker, qtbot):
    """The answer of the load before must not say "ready" while the newer model is still loading."""
    states = []
    worker.state_changed.connect(lambda state, message: states.append(state))
    with Gate() as gate:
        def slow(model, device):
            if model == "sam2":
                gate.park()
            return OnDevice(device)

        worker.start(slow, "edgetam", "cpu")
        worker.start(slow, "sam2", "cpu")
        qtbot.waitUntil(gate.parked.is_set)
        # the first load answered before the second one began: hand that answer over now, not later
        QCoreApplication.sendPostedEvents()
        assert worker.state == "loading" and "ready" not in states
        gate.open()
        qtbot.waitUntil(lambda: worker.ready)
    assert states == ["loading", "loading", "ready"]


def two_blocks() -> np.ndarray:
    """A light image of 60 x 40 px, an RGB array [row, column, 3], with two dark blocks of 10 x 10 px
    whose centers are (15, 15) and (45, 15), px in Tracker's convention."""
    image = np.full((40, 60, 3), 255, np.uint8)
    image[10:20, 10:20] = image[10:20, 40:50] = 0
    return image


def test_an_outline_asked_for_while_another_model_loads_is_made_by_that_model(worker, qtbot):
    """Also while the model before is still at work on an older request: it is given no newer one."""
    image, done = two_blocks(), []
    worker.preview_done.connect(lambda preview: done.append((worker.state, preview.serial)))
    with Gate() as gate:
        models = {"edgetam": Watched(gate=gate, park_at={0}), "sam2": Watched()}
        worker.start(lambda model, device: models[model], "edgetam", "cpu")
        qtbot.waitUntil(lambda: worker.ready)
        worker.request_preview(0, image, (0, 0), [click("A", 15.0, 15.0)])
        qtbot.waitUntil(gate.parked.is_set)  # the model before stands inside preview(), for A
        worker.start(lambda model, device: models[model], "sam2", "cpu")
        newest = worker.request_preview(0, image, (0, 0), [click("B", 45.0, 15.0)])
        assert worker.busy and worker.state == "loading"
        gate.open()
        qtbot.waitUntil(lambda: worker.ready and not worker.busy)
    assert asked_of(models["edgetam"]) == [["A"]] and asked_of(models["sam2"]) == [["B"]]
    assert done == [("ready", newest)]
    assert models["edgetam"].closed == 1 and models["sam2"].closed == 0


def test_an_outline_that_is_awaited_when_another_model_is_asked_for_is_made_by_that_model(worker, qtbot):
    """What the model before made of it is not reported: it arrives while the other model loads."""
    image, done, failed = two_blocks(), [], []
    worker.preview_done.connect(lambda preview: done.append((worker.state, preview.serial)))
    worker.preview_failed.connect(lambda serial, reason: failed.append(serial))
    with Gate() as gate:
        models = {"edgetam": Watched(gate=gate, park_at={0}), "sam2": Watched()}
        worker.start(lambda model, device: models[model], "edgetam", "cpu")
        qtbot.waitUntil(lambda: worker.ready)
        awaited = worker.request_preview(0, image, (0, 0), [click("A", 15.0, 15.0)])
        qtbot.waitUntil(gate.parked.is_set)
        worker.start(lambda model, device: models[model], "sam2", "cpu")
        assert worker.busy
        gate.open()
        qtbot.waitUntil(lambda: worker.ready and not worker.busy)
    assert asked_of(models["edgetam"]) == [["A"]] and asked_of(models["sam2"]) == [["A"]]
    assert done == [("ready", awaited)] and failed == []  # reported once, and that one is the new model's
