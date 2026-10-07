"""The model and the device in panel 7, and the worker's two files (task C8b; SPEC 10.1, 15; X7).

- Panel 7 offers the model (EdgeTAM, the default, and SAM 2.1 tiny) and the device (auto, cpu, and
  the one this system can have besides). A choice goes into `session.processing` and the worker
  loads that model: the window's factory is asked for it, which a recording factory shows. Both
  boxes are off while a model loads and during a run.
- gui/worker.py was split: the engine of the thread is in gui/worker_engine.py, and what
  gui/worker.py offered is still importable from it.

No test here imports torch: the factory is a stand-in, and which devices a system can have is
decided from the system's name alone.
"""

import platform
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication

from gui_helpers import picture, record_dialogs
from last_controls_helpers import RUNNING, OnDevice, choose, let_run_end, parked, press_track
from outline_tracker.gui.panels.track_panel import devices_for
from outline_tracker.gui.worker import Worker
from prompt_helpers import Gate
from session_helpers import body, read_json
from track_helpers import NAME, own_copy, ready_to_track, track_panel

DEVICE_TEXTS = {"auto": "auto", "cpu": "cpu", "mps": "mps (Apple GPU)", "cuda": "cuda (NVIDIA GPU)"}
# What gui/worker.py defined before it was split (the module as of task C5), private names too.
OFFERED = ["STACK_BYTES", "NO_FACTORY", "REASON_LENGTH", "TRACES_KEPT", "log", "Preview", "_Request", "Found",
           "found_in", "plain", "traced", "before_load_for", "_Engine", "Worker", "worker_of"]


class Recording:
    """A factory for the window that keeps what it was asked for: `asked` holds (model, device),
    `made` the stand-ins it gave, in order. The call number `park_at` (from 1) waits at `gate`
    first; a model in `broken` raises `OSError(broken[model])` instead."""

    def __init__(self, gate=None, park_at=None, broken=None):
        self.asked, self.made = [], []
        self.gate, self.park_at, self.broken = gate, park_at, broken or {}

    def __call__(self, model, device):
        self.asked.append((model, device))
        if len(self.asked) == self.park_at:
            self.gate.park()
        if model in self.broken:
            raise OSError(self.broken[model])
        self.made.append(OnDevice("cpu" if device == "auto" else device))
        return self.made[-1]


def opened(window, qtbot, clip, factory):
    """The window with `factory`, a named student and `clip` open, shown, the model ready: its
    `TrackPanel`."""
    window.segmenter_factory = factory
    window.controller.set_student(NAME)
    picture(window, qtbot, clip)
    panel = track_panel(window)
    qtbot.waitUntil(lambda: panel.worker.ready)
    return panel


def entries(box) -> list[tuple[str, str]]:
    return [(box.itemData(index), box.itemText(index)) for index in range(box.count())]


# ---------------------------------------------------------------------------------------------
# What is offered


def test_the_devices_are_auto_cpu_and_the_one_the_system_can_have_besides():
    assert devices_for("darwin", "arm64") == ["auto", "cpu", "mps"]  # a Mac with an Apple chip
    assert devices_for("win32", "AMD64") == ["auto", "cpu", "cuda"]
    assert devices_for("linux", "x86_64") == ["auto", "cpu", "cuda"]


def test_the_boxes_offer_the_two_models_and_this_systems_devices(window, qtbot, clip_in_odd_folder):
    from outline_tracker.segmenter import hf  # its table of models; torch is not loaded by this

    panel = opened(window, qtbot, clip_in_odd_folder, Recording())
    assert entries(panel.model_box) == [("edgetam", "EdgeTAM"), ("sam2", "SAM 2.1 tiny")]
    assert {key for key, _ in entries(panel.model_box)} <= set(hf.MODELS)  # keys the real factory knows
    here = devices_for(sys.platform, platform.machine())
    assert entries(panel.device_box) == [(key, DEVICE_TEXTS[key]) for key in here]
    assert (panel.model_box.currentData(), panel.device_box.currentData()) == ("edgetam", "auto")  # the defaults
    assert panel.model_box.toolTip() and panel.device_box.toolTip()


def test_without_a_video_there_is_no_session_to_keep_a_choice_so_the_boxes_are_off(window):
    panel = track_panel(window)
    assert not panel.model_box.isEnabled() and not panel.device_box.isEnabled()
    assert panel.model_box.toolTip() == panel.device_box.toolTip() == "Open your video in panel 1 first."


# ---------------------------------------------------------------------------------------------
# A choice is stored and loaded


def test_another_model_is_stored_asked_of_the_factory_and_shown_as_loading_then_ready(window, qtbot,
                                                                                    clip_in_odd_folder):
    with Gate() as gate:
        factory = Recording(gate, park_at=2)
        panel = opened(window, qtbot, clip_in_odd_folder, factory)
        model_label = body(window, 6).model_label  # the model's state in the status bar
        assert factory.asked == [("edgetam", "auto")] and model_label.text() == "Model ready"
        assert panel.model_box.isEnabled() and panel.device_box.isEnabled()
        choose(panel.model_box, "sam2")
        assert window.controller.session.processing.model == "sam2"
        qtbot.waitUntil(gate.parked.is_set)
        assert factory.asked == [("edgetam", "auto"), ("sam2", "auto")]
        assert panel.worker.state == "loading" and model_label.text() == "Model loading"
        assert not panel.progress_bar.isHidden() and panel.progress_label.text() == "The model is loading."
        assert not panel.track_button.isEnabled()
        # off while a model loads, and each says why
        assert not panel.model_box.isEnabled() and not panel.device_box.isEnabled()
        assert panel.model_box.toolTip() == panel.device_box.toolTip() == "The model is loading."
        assert factory.made[0].closed == 1  # one copy of the weights: the model before was closed first
        gate.open()
        qtbot.waitUntil(lambda: panel.worker.ready)
    assert model_label.text() == "Model ready" and panel.progress_bar.isHidden()
    assert panel.model_box.isEnabled() and panel.device_box.isEnabled()
    assert panel.model_box.currentData() == "sam2"


def test_another_device_is_stored_loaded_and_saved_with_the_session(window, qtbot, clip_in_odd_folder):
    factory = Recording()
    panel = opened(window, qtbot, clip_in_odd_folder, factory)
    assert panel.worker.device == "cpu" and window.readouts.device_label.text() == "Device cpu"
    other = panel.device_box.itemData(2)  # mps on a Mac with an Apple chip, cuda elsewhere
    choose(panel.device_box, other)
    qtbot.waitUntil(lambda: panel.worker.ready and len(factory.made) == 2)
    assert factory.asked == [("edgetam", "auto"), ("edgetam", other)]
    assert panel.worker.device == other and window.readouts.device_label.text() == f"Device {other}"
    window.controller.save_now()
    processing = read_json(window.controller.run_folder / "session.json")["processing"]
    assert (processing["model"], processing["device"]) == ("edgetam", other)
    choose(panel.device_box, other)  # the same again: nothing to load
    assert factory.asked == [("edgetam", "auto"), ("edgetam", other)] and panel.worker.ready


def test_a_model_that_cannot_be_loaded_gives_its_plain_reason_and_another_can_be_chosen(window, qtbot,
                                                                                      clip_in_odd_folder, monkeypatch):
    shown = record_dialogs(monkeypatch)
    factory = Recording(broken={"sam2": "The weights of this model are not on this computer."})
    panel = opened(window, qtbot, clip_in_odd_folder, factory)
    choose(panel.model_box, "sam2")
    qtbot.waitUntil(lambda: panel.worker.state == "failed")
    assert panel.worker.message == "The weights of this model are not on this computer."
    assert panel.worker.device is None and body(window, 6).model_label.text() == "Model not loaded"
    (_, kind, text), = shown.messages
    assert kind == "problem" and panel.worker.message in text and "Traceback" not in text
    assert panel.model_box.isEnabled() and panel.device_box.isEnabled()  # so that another can be chosen
    choose(panel.model_box, "sam2")  # the same again, after the internet came back, say: it is tried again
    qtbot.waitUntil(lambda: panel.worker.state == "failed" and len(factory.asked) == 3)
    choose(panel.model_box, "edgetam")
    qtbot.waitUntil(lambda: panel.worker.ready)
    assert factory.asked == [("edgetam", "auto"), ("sam2", "auto"), ("sam2", "auto"), ("edgetam", "auto")]
    assert window.controller.session.processing.model == "edgetam"


def test_during_a_run_both_boxes_are_off_and_no_other_model_is_loaded(window, qtbot, clip_in_odd_folder):
    clip, asked = clip_in_odd_folder, []
    with Gate() as gate:
        segmenter = parked(clip, gate)

        def factory(model, device):
            asked.append((model, device))
            return segmenter

        window.segmenter_factory = factory
        panel, _ = ready_to_track(window, qtbot, clip, end=10)
        assert panel.model_box.isEnabled() and panel.device_box.isEnabled()
        press_track(qtbot, panel, gate)
        assert not panel.model_box.isEnabled() and not panel.device_box.isEnabled()
        assert panel.model_box.toolTip() == panel.device_box.toolTip() == RUNNING
        panel.set_model("sam2")  # the function behind the box
        panel.set_device("cpu")
        processing = window.controller.session.processing
        assert (processing.model, processing.device) == ("edgetam", "auto") and asked == [("edgetam", "auto")]
        let_run_end(qtbot, panel, gate)
    assert panel.model_box.isEnabled() and panel.device_box.isEnabled()
    assert panel.jobs.status == "complete" and asked == [("edgetam", "auto")]  # the run ended with its own model


def test_a_session_that_is_opened_loads_its_own_model_and_device(window, qtbot, clip_in_odd_folder, disk_clip,
                                                              tmp_path):
    factory = Recording()
    panel = opened(window, qtbot, clip_in_odd_folder, factory)
    choose(panel.model_box, "sam2")
    choose_device = panel.device_box.itemData(1)
    qtbot.waitUntil(lambda: panel.worker.ready and len(factory.made) == 2)
    choose(panel.device_box, choose_device)
    qtbot.waitUntil(lambda: panel.worker.ready and len(factory.made) == 3)
    window.controller.save_now()
    session_file = window.controller.run_folder / "session.json"
    # another video: a new session, with a new session's model and device
    window.open_path(own_copy(disk_clip, tmp_path / "other").path)
    qtbot.waitUntil(lambda: panel.worker.ready and len(factory.made) == 4)
    assert factory.asked[-1] == ("edgetam", "auto")
    assert (panel.model_box.currentData(), panel.device_box.currentData()) == ("edgetam", "auto")
    window.open_path(session_file)
    qtbot.waitUntil(lambda: panel.worker.ready and len(factory.made) == 5)
    assert factory.asked[-1] == ("sam2", "cpu")
    assert (panel.model_box.currentData(), panel.device_box.currentData()) == ("sam2", "cpu")
    assert [made.closed for made in factory.made] == [1, 1, 1, 1, 0]  # one model at a time


def test_a_sessions_model_or_device_that_is_not_in_the_list_is_shown_as_it_is(window, qtbot, clip_in_odd_folder):
    factory = Recording()
    panel = opened(window, qtbot, clip_in_odd_folder, factory)
    processing = window.controller.session.processing
    processing.model, processing.device = "sam2-small", "another"  # as a session from elsewhere may say
    window.controller.touch()
    assert (panel.model_box.currentData(), panel.device_box.currentData()) == ("sam2-small", "another")
    assert (processing.model, processing.device) == ("sam2-small", "another")  # shown, never changed


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


@pytest.fixture
def worker(qtbot):
    """A worker of its own, not started; its thread is stopped after the test."""
    made = Worker()
    yield made
    made.stop()


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
