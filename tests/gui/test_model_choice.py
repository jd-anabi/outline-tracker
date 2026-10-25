"""The model and the device in panel 7 (task C8b; SPEC 10.1, 15; X7): what is offered, chosen, stored and loaded.

- Panel 7 offers the model (EdgeTAM, the default, and SAM 2.1 tiny) and the device (auto, cpu, and
  the one this system can have besides). A choice goes into `session.processing` and the worker
  loads that model: the window's factory is asked for it, which a recording factory shows. Both
  boxes are off while a model loads and during a run.
- What is on the picture and in the estimate is the chosen model's: the outlines of the frame shown
  are made again by the model that is loaded in place of the one before, and the time per frame of
  the model before is not kept. Which stand-in made an outline is read from where the outline is:
  a `ThresholdFake` outlines the disk under the click, an `ExactFake` the scene's object of the
  track's name wherever the click is (the disk scene's centers are its ground truth).
- A model that cannot be loaded says what to do next: at the first start, check the internet and
  start again; after a choice, choose another in panel 7.

The worker's two files, its device, and the outlines asked for while another model loads are in
tests/gui/test_model_choice_worker.py.

No test here imports torch: the factory is a stand-in, and which devices a system can have is
decided from the system's name alone.
"""

import platform
import sys

import pytest
from PySide6.QtCore import QCoreApplication

from gui_helpers import picture, record_dialogs, show
from last_controls_helpers import OnDevice, choose
from outline_tracker.gui.panels.track_panel import devices_for
from outline_tracker.segmenter.fake import ExactFake, ThresholdFake
from prompt_helpers import Gate, Watched, asked_of, objects_panel
from session_helpers import body, read_json
from track_helpers import NAME, own_copy, ready_to_track, track_panel, window_hint
from tracking_helpers import center

DEVICE_TEXTS = {"auto": "auto", "cpu": "cpu", "mps": "mps (Apple GPU)", "cuda": "cuda (NVIDIA GPU)"}
# The last sentences of the dialog about a model that could not be loaded: what to do next.
CHOOSE_ANOTHER = ("Choose another model or device in panel 7 (EdgeTAM and auto are the defaults). A model needs the "
                  "internet the first time it is loaded.")
FIRST_START = "Check the internet connection, which the first start needs, and start the app again."
NOT_ON_THIS_COMPUTER = "The weights of this model are not on this computer."


class Recording:
    """A factory for the window that keeps what it was asked for: `asked` holds (model, device),
    `made` the stand-ins it gave, in order. The call number `park_at` (from 1) waits at `gate`
    first; a model in `broken` raises `OSError(broken[model])` instead. A model in `others` is
    made by `others[model]()`, in place of the `OnDevice` every other model gets."""

    def __init__(self, gate=None, park_at=None, broken=None, others=None):
        self.asked, self.made = [], []
        self.gate, self.park_at, self.broken, self.others = gate, park_at, broken or {}, others or {}

    def __call__(self, model, device):
        self.asked.append((model, device))
        if len(self.asked) == self.park_at:
            self.gate.park()
        if model in self.broken:
            raise OSError(self.broken[model])
        make = self.others.get(model, lambda: OnDevice("cpu" if device == "auto" else device))
        self.made.append(make())
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


def middle(outline) -> tuple[float, float]:
    """The middle of the box around an outline, (N, 2) points (u, v) in px of the frame: for the
    outline of a disk, the disk's center."""
    low, high = outline.min(axis=0), outline.max(axis=0)
    return float(low[0] + high[0]) / 2, float(low[1] + high[1]) / 2


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
    # what to do next is what helps here: another choice. Starting again would load the same model again.
    assert text.endswith(CHOOSE_ANOTHER) and "start the app again" not in text
    assert panel.model_box.isEnabled() and panel.device_box.isEnabled()  # so that another can be chosen
    choose(panel.model_box, "sam2")  # the same again, after the internet came back, say: it is tried again
    qtbot.waitUntil(lambda: panel.worker.state == "failed" and len(factory.asked) == 3)
    choose(panel.model_box, "edgetam")
    qtbot.waitUntil(lambda: panel.worker.ready)
    assert factory.asked == [("edgetam", "auto"), ("sam2", "auto"), ("sam2", "auto"), ("edgetam", "auto")]
    assert window.controller.session.processing.model == "edgetam"


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


def test_a_first_load_that_fails_says_to_check_the_internet_and_a_failed_choice_says_to_choose_another(
        window, qtbot, clip_in_odd_folder, monkeypatch):
    shown = record_dialogs(monkeypatch)
    factory = Recording(broken={"edgetam": "The model files could not be downloaded."})
    window.segmenter_factory = factory
    window.controller.set_student(NAME)
    picture(window, qtbot, clip_in_odd_folder)
    panel = track_panel(window)
    qtbot.waitUntil(lambda: panel.worker.state == "failed")
    (_, kind, text), = shown.messages
    assert kind == "problem" and "The model files could not be downloaded." in text
    assert text.endswith(FIRST_START) and "panel 7" not in text  # the defaults, at the start: nothing was chosen
    choose(panel.device_box, "cpu")  # a choice that fails too
    qtbot.waitUntil(lambda: panel.worker.state == "failed" and len(shown.messages) == 2)
    assert factory.asked == [("edgetam", "auto"), ("edgetam", "cpu")]
    assert shown.messages[1][2].endswith(CHOOSE_ANOTHER) and "start the app again" not in shown.messages[1][2]


def test_a_session_whose_own_model_cannot_be_loaded_says_to_choose_another_and_that_helps(window, qtbot,
                                                                                         clip_in_odd_folder,
                                                                                         monkeypatch):
    """What a student meets who starts the app again after a choice that failed: the session still
    names that model, so "start the app again" would lead nowhere."""
    shown = record_dialogs(monkeypatch)
    factory = Recording(broken={"sam2": NOT_ON_THIS_COMPUTER})
    window.segmenter_factory = factory
    window.controller.set_student(NAME)
    window.open_path(clip_in_odd_folder.path)  # not shown yet: nothing is loaded
    window.controller.session.processing.model = "sam2"  # as the session of the run before says
    window.controller.touch()
    show(window, qtbot)
    panel = track_panel(window)
    qtbot.waitUntil(lambda: panel.worker.state == "failed")
    assert factory.asked == [("sam2", "auto")]  # the first load of this window is the session's own
    text = shown.messages[-1][2]
    assert NOT_ON_THIS_COMPUTER in text and text.endswith(CHOOSE_ANOTHER) and "start the app again" not in text
    assert panel.model_box.isEnabled()
    choose(panel.model_box, "edgetam")
    qtbot.waitUntil(lambda: panel.worker.ready)
    assert window.controller.session.processing.model == "edgetam"


# ---------------------------------------------------------------------------------------------
# What is on the picture, and the estimate, are the chosen model's


def test_after_another_model_is_chosen_the_outline_on_the_picture_is_made_again_by_that_model(window, qtbot,
                                                                                             disk_clip, tmp_path):
    clip = own_copy(disk_clip, tmp_path / "disks")
    with Gate() as gate:
        first, second = Watched(), Watched(ExactFake(clip))
        factory = Recording(gate, park_at=2, others={"edgetam": lambda: first, "sam2": lambda: second})
        panel = opened(window, qtbot, clip, factory)
        objects = objects_panel(window)
        prompts = objects.prompts
        objects.add_object()
        assert prompts.add_point(*center(clip, "B", 0), 1)  # object A, clicked on the scene's disk B
        qtbot.waitUntil(lambda: "A" in prompts.outlines)
        assert middle(prompts.outlines["A"]) == pytest.approx(center(clip, "B", 0), abs=1.0)  # under the click
        choose(panel.model_box, "sam2")
        qtbot.waitUntil(gate.parked.is_set)
        QCoreApplication.sendPostedEvents()  # whatever the model before had to report is handed over now
        # while the other model loads, the outline of the model before is not left on the picture
        assert panel.worker.state == "loading" and prompts.outlines == {}
        assert prompts.busy and not objects.busy_label.isHidden()
        assert objects.busy_label.text() == "The model is loading. The outline appears when it is ready."
        gate.open()
        qtbot.waitUntil(lambda: panel.worker.ready and not prompts.busy)
    # no click and no other frame since: the outline of the frame shown was made again, by the second model
    assert middle(prompts.outlines["A"]) == pytest.approx(center(clip, "A", 0), abs=1.0)
    assert asked_of(first) == [["A"]] and asked_of(second) == [["A"]]  # the model before was not asked again
    assert prompts.message == ("", "") and objects.busy_label.isHidden()


def test_after_a_model_that_could_not_be_loaded_the_next_ones_outline_appears_and_the_message_goes(
        window, qtbot, disk_clip, tmp_path, monkeypatch):
    record_dialogs(monkeypatch)
    clip = own_copy(disk_clip, tmp_path / "disks")
    panel = opened(window, qtbot, clip, Recording(broken={"sam2": NOT_ON_THIS_COMPUTER}))
    objects = objects_panel(window)
    prompts = objects.prompts
    objects.add_object()
    assert prompts.add_point(*center(clip, "A", 0), 1)
    qtbot.waitUntil(lambda: "A" in prompts.outlines)
    choose(panel.model_box, "sam2")
    qtbot.waitUntil(lambda: panel.worker.state == "failed")
    # no model: the outline of the model before is gone, and the panel says why none is shown
    assert prompts.outlines == {} and not prompts.busy
    assert prompts.message == ("problem", "The model could not be loaded, so no outline can be shown. "
                                          + NOT_ON_THIS_COMPUTER)
    choose(panel.model_box, "edgetam")
    qtbot.waitUntil(lambda: panel.worker.ready and not prompts.busy)
    assert middle(prompts.outlines["A"]) == pytest.approx(center(clip, "A", 0), abs=1.0)  # by itself: no click
    assert prompts.message == ("", "")


def test_clicks_whose_outline_one_model_could_not_make_are_tried_on_the_next_model(window, qtbot, disk_clip,
                                                                                  tmp_path, monkeypatch):
    class Broken(ThresholdFake):
        def preview(self, image, prompts):
            raise RuntimeError("The model ran out of memory.")

    record_dialogs(monkeypatch)
    clip = own_copy(disk_clip, tmp_path / "disks")
    panel = opened(window, qtbot, clip, Recording(others={"edgetam": Broken}))
    objects = objects_panel(window)
    prompts = objects.prompts
    objects.add_object()
    assert prompts.add_point(*center(clip, "A", 0), 1)
    qtbot.waitUntil(lambda: prompts.message[0] == "problem" and not prompts.busy)
    assert prompts.message == ("problem", "The outline could not be made. The model ran out of memory.")
    assert prompts.outlines == {}
    choose(panel.model_box, "sam2")  # the same clicks, another model
    qtbot.waitUntil(lambda: panel.worker.ready and not prompts.busy)
    assert middle(prompts.outlines["A"]) == pytest.approx(center(clip, "A", 0), abs=1.0)
    assert prompts.message == ("", "")


def test_the_time_per_frame_of_the_model_before_is_not_kept_and_the_new_models_outline_is_timed(window, qtbot,
                                                                                             disk_clip, tmp_path):
    """By hand: 2.0 s per frame for 60 frames of one object are 120 s, "about 2 min". The second model
    is ready at 200.0 s and its outline of the one object is there at 203.0 s: 3.0 s per frame,
    180 s, "about 3 min"."""
    clip = own_copy(disk_clip, tmp_path / "disks")
    now = [100.0]
    with Gate() as gate:
        second = Watched(gate=gate, park_at={0})
        window.segmenter_factory = Recording(others={"sam2": lambda: second})
        panel, objects = ready_to_track(window, qtbot, clip)  # object A, clicked, its outline shown
        panel.clock = lambda: now[0]
        panel.seconds_per_frame = 2.0  # as the first model was timed
        panel.refresh()
        assert window_hint(panel) == "Estimated time: about 2 min for 60 frames and 1 object."
        now[0] = 200.0
        choose(panel.model_box, "sam2")
        assert panel.seconds_per_frame is None  # what the model before took says nothing about this one
        qtbot.waitUntil(gate.parked.is_set)  # the second model is loaded, and at work on the outline
        assert panel.worker.ready and panel.seconds_per_frame is None
        assert window_hint(panel) == "No time estimate yet: 60 frames and 1 object are ready to track."
        now[0] = 203.0
        gate.open()
        qtbot.waitUntil(lambda: not objects.prompts.busy)
    assert panel.seconds_per_frame == pytest.approx(3.0)
    assert window_hint(panel) == "Estimated time: about 3 min for 60 frames and 1 object."
    choose(panel.model_box, "sam2")  # the same again: nothing changed, so the time stays
    assert panel.seconds_per_frame == pytest.approx(3.0)
    choose(panel.device_box, "cpu")  # another device is another speed
    assert panel.seconds_per_frame is None
    qtbot.waitUntil(lambda: panel.worker.ready and not objects.prompts.busy)


def test_a_sessions_model_or_device_that_is_not_in_the_list_is_shown_as_it_is(window, qtbot, clip_in_odd_folder):
    factory = Recording()
    panel = opened(window, qtbot, clip_in_odd_folder, factory)
    processing = window.controller.session.processing
    processing.model, processing.device = "sam2-small", "another"  # as a session from elsewhere may say
    window.controller.touch()
    assert (panel.model_box.currentData(), panel.device_box.currentData()) == ("sam2-small", "another")
    assert (processing.model, processing.device) == ("sam2-small", "another")  # shown, never changed
