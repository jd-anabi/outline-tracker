"""The joins between the panels (2, 3): the one lock, and the model after results.

Task C9; SPEC 6.4, 8.1, 10.1, 10.2; decision X7. Panels 5 to 9 were built side by side. These tests hold the places
where two of them meet, numbered as that build step numbered them. This file holds the second and the third:

2. One lock: what a tracking run switches off is also off during an export, with the reason
   "An export is running."; Track is off during an export and Export all during a run. One
   function, `Jobs.writing()`, says whether and why.
3. Once a track has stored frames, the model of the session is fixed; the device is not.

The other joins are in tests/gui/test_joins.py (1: Export all and the flags need no model) and in
tests/gui/test_joins_exported_files.py (4 to 6: exported files older than what they come from, the Quickstart,
the two files that were split).

Expected values are the brief's sentences and SPEC 8.1's list of files, never the panels' own
output. A factory, a run or an export is parked on a gate (tests/support/prompt_helpers.py), so
"during" is a state the test holds; where a file's time decides, the test sets it. Frames are
video frame numbers; (u, v) is in px of the video frame (SPEC 3.1).
"""

import re

from PySide6.QtCore import Qt

from export_helpers import calibrate
from export_panel_helpers import END, FRAMES, export_panel, tracked, watch_export
from joins_helpers import EXPORTING, RUNNING, WAIT_FOR_EXPORT, corrections, lockable, not_saying, off
from last_controls_helpers import PACKAGE, choose, let_run_end, parked, press_track, record_confirm
from layout_helpers import below
from layout_helpers import path as table_path
from outline_tracker.gui.worker import worker_of
from outline_tracker.gui.worker_jobs import jobs_of
from outline_tracker.segmenter.fake import ExactFake
from prompt_helpers import LEFT, NO_KEY, Gate, objects_panel
from review_helpers import review_panel
from session_helpers import body
from track_helpers import Tracked, ids, ready_to_track, results_of, run_to_end, track_panel, window_hint
from tracking_helpers import center

MODEL_TIP = "The model that finds the outlines. EdgeTAM is the default."
DEVICE_TIP = "What the model runs on. auto takes the graphics processor if it works, else the processor (cpu)."
MODEL_FIXED = "The results were made with EdgeTAM. For another model, remove the objects or start a new session."


# ---------------------------------------------------------------------------------------------
# 2. One lock


def a_tracked_object_and_a_new_one(window, qtbot, clip):
    """A is tracked on `FRAMES` with a scale; B is added after the run, fine with a typed window,
    selected, with the Positive tool chosen. Returns (ExportPanel, ObjectsPanel, ReviewPanel)."""
    panel = tracked(window, qtbot, clip)
    objects, review = objects_panel(window), review_panel(window)
    objects.add_object()
    objects.set_mode("fine")
    objects.set_fine_window(120)
    assert objects.prompts.selected == "B" and objects.prompts.tool_kind == "positive"
    return panel, objects, review


def press_export(qtbot, panel, gate) -> None:
    """Press Export all and wait until the export stands at `gate`, inside `export_all`."""
    panel.export_button.click()
    assert panel.exporting, panel.message.text() or panel.export_button.toolTip()
    qtbot.waitUntil(gate.parked.is_set)


def let_export_end(qtbot, panel, gate) -> None:
    gate.open()
    qtbot.waitUntil(lambda: not panel.exporting, timeout=30_000)


def test_the_one_function_gives_the_runs_reason_the_exports_reason_and_none_otherwise(window, qtbot,
                                                                                      clip_in_odd_folder,
                                                                                      monkeypatch):
    clip = clip_in_odd_folder
    with Gate() as run_gate:
        track, _ = ready_to_track(window, qtbot, clip, parked(clip, run_gate), end=END)
        calibrate(window.controller.session)
        window.controller.touch()
        jobs, panel = jobs_of(window), export_panel(window)
        assert jobs.writing() is None
        press_track(qtbot, track, run_gate)
        assert jobs.writing() == RUNNING
        # Export all is off during a run, with the run's reason
        assert not panel.export_button.isEnabled() and panel.export_button.toolTip().startswith(RUNNING)
        panel.export()
        assert not panel.exporting and jobs.writing() == RUNNING
        let_run_end(qtbot, track, run_gate)
    assert jobs.writing() is None and jobs.status == "complete"
    assert window.panels[6].state == "done"
    with Gate() as export_gate:
        watch_export(monkeypatch, gate=export_gate)
        press_export(qtbot, panel, export_gate)
        assert jobs.writing() == EXPORTING
        # Track is off during an export, with the export's reason, and the panel stays done
        assert not track.track_button.isEnabled() and track.track_button.toolTip() == EXPORTING
        assert window_hint(track) == EXPORTING and window.panels[6].state == "done"
        assert jobs.start() == EXPORTING and not jobs.running
        let_export_end(qtbot, panel, export_gate)
    assert jobs.writing() is None and panel.message.kind == "success"
    assert EXPORTING not in track.track_button.toolTip() and EXPORTING not in window_hint(track)


def test_during_an_export_every_part_a_run_switches_off_is_off_and_says_why(window, qtbot, clip_in_odd_folder,
                                                                            monkeypatch):
    panel, objects, review = a_tracked_object_and_a_new_one(window, qtbot, clip_in_odd_folder)
    parts, fixes, model_box = lockable(window), corrections(window), body(window, 7).model_box
    assert off(parts) == []  # B is selected: all on before the export
    objects.prompts.select("A")
    assert off(fixes) == []  # A is tracked: it can be fixed
    with Gate() as gate:
        watch_export(monkeypatch, gate=gate)
        press_export(qtbot, panel, gate)
        everything = {**parts, **fixes, "model": model_box}
        assert sorted(off(everything)) == sorted(everything)
        assert not_saying(everything, EXPORTING) == []
        assert not panel.folder_button.isEnabled()  # Change folder is File > Save session as
        # what an export does not read stays usable, and so does looking at the flags
        assert window.menus.save_action.isEnabled() and window.menus.quit_action.isEnabled()
        assert body(window, 8).shape_box.isEnabled()
        let_export_end(qtbot, panel, gate)
    assert panel.message.kind == "success"
    assert off(fixes) == [] and not_saying(fixes, EXPORTING) == list(fixes)
    objects.prompts.select("B")
    assert off(parts) == [] and not_saying(parts, EXPORTING) == list(parts)
    assert all(part.toolTip() for part in parts.values())  # each has its own tip back
    assert panel.folder_button.isEnabled()


def test_during_an_export_a_click_on_the_picture_adds_no_point_and_the_tool_says_to_wait(window, qtbot,
                                                                                        clip_in_odd_folder,
                                                                                        monkeypatch):
    clip = clip_in_odd_folder
    panel, objects, _ = a_tracked_object_and_a_new_one(window, qtbot, clip)
    prompts, tool = objects.prompts, window.view.tool
    u, v = center(clip, "B", 0)
    with Gate() as gate:
        watch_export(monkeypatch, gate=gate)
        press_export(qtbot, panel, gate)
        assert window.view.tool is tool  # the tool stays chosen, and says that it does nothing now
        assert window.tool_text.text() == WAIT_FOR_EXPORT and tool.cursor == Qt.CursorShape.ForbiddenCursor
        tool.click(u, v, LEFT, NO_KEY)  # what the view calls for a click on the picture
        assert not prompts.add_point(u, v, 1) and not prompts.add_point(u + 3, v, 0)
        assert not prompts.set_head(u, v)
        prompts.select("A")
        assert not prompts.undo()  # the Undo key
        a, b = window.controller.session.tracks
        assert b.prompts == [] and b.head_px is None
        assert [[tuple(point) for point in prompt.points_px] for prompt in a.prompts] == [[center(clip, "A", 0)]]
        let_export_end(qtbot, panel, gate)
    prompts.select("B")
    prompts.choose_tool("positive")
    assert window.tool_text.text() == "Positive: click on animal B."
    assert window.view.tool.cursor == Qt.CursorShape.CrossCursor
    window.view.tool.click(u, v, LEFT, NO_KEY)
    b = window.controller.session.tracks[1]
    assert [[tuple(point) for point in prompt.points_px] for prompt in b.prompts] == [[(u, v)]]
    qtbot.waitUntil(lambda: not prompts.busy)


def test_during_an_export_a_correction_is_refused_and_nothing_else_changes_its_input(window, qtbot,
                                                                                     clip_in_odd_folder,
                                                                                     monkeypatch):
    """A key, code, or the answer to a question asked before the export began, that reaches a
    function behind a button that is off."""
    clip, questions = clip_in_odd_folder, record_confirm(monkeypatch)
    loads = []
    panel, objects, review = a_tracked_object_and_a_new_one(window, qtbot, clip)
    worker_of(window).state_changed.connect(lambda state, message: loads.append(state))
    track = track_panel(window)
    objects.prompts.select("A")
    window.show_frame(0)
    objects.ask_remove()   # three questions asked before the export, answered during it
    review.retrack()       # A has its click on frame 0
    window.show_frame(10)
    review.end_track()
    assert [action for _, _, action, _ in questions.asked] == ["Remove", "Re-track from here", "End track here"]
    with Gate() as gate:
        watch_export(monkeypatch, gate=gate)
        press_export(qtbot, panel, gate)
        for _, _, _, answer_yes in questions.asked:
            answer_yes()
        review.retrack()
        review.end_track()
        review.continue_as_new()
        assert len(questions.asked) == 3  # no correction asks now
        objects.add_object()
        objects.remove("A")
        objects.remove_button.click()
        objects.prompts.select("B")
        objects.set_mode("coarse")
        objects.set_fine_window(64)
        track.set_model("sam2")
        track.set_device("cpu")
        a, b = window.controller.session.tracks
        processing = window.controller.session.processing
        assert ids(window) == ["A", "B"] and a.ended_at is None and (b.mode, b.fine_window_px) == ("fine", 120)
        assert (processing.model, processing.device) == ("edgetam", "auto") and loads == []
        assert not review.jobs.running and review.jobs.start() == EXPORTING
        assert results_of(window).arrays("A").frames.tolist() == FRAMES
        let_export_end(qtbot, panel, gate)
    assert panel.message.kind == "success" and results_of(window).arrays("A").frames.tolist() == FRAMES


def test_no_other_module_of_the_window_names_panel_9_for_the_lock():
    """Every panel asks `Jobs.writing()`; none reads another panel's attribute."""
    gui = PACKAGE / "gui"
    others = [path for path in gui.rglob("*.py") if path != table_path("export panel")]
    naming = [path.relative_to(gui).as_posix() for path in others
              if re.search(r"ExportPanel|export_panel import|\.exporting\b", path.read_text(encoding="utf-8"))]
    assert naming == sorted(below(gui, "worker jobs"))  # which keeps the lock: `Jobs.exporting`


# ---------------------------------------------------------------------------------------------
# 3. The model after results


def test_with_results_the_model_box_is_off_with_its_sentence_and_the_device_box_is_on(window, qtbot,
                                                                                    clip_in_odd_folder):
    clip, asked = clip_in_odd_folder, []

    def factory(model, device):
        asked.append((model, device))
        return Tracked(ExactFake(clip))

    window.segmenter_factory = factory
    track, objects = ready_to_track(window, qtbot, clip, end=10)
    model_box, device_box, processing = track.model_box, track.device_box, window.controller.session.processing
    assert model_box.isEnabled() and model_box.toolTip() == MODEL_TIP  # nothing is tracked yet
    assert device_box.isEnabled() and device_box.toolTip() == DEVICE_TIP
    run_to_end(qtbot, track)
    assert results_of(window).track_ids == ["A"]
    assert not model_box.isEnabled() and model_box.toolTip() == MODEL_FIXED
    assert device_box.isEnabled() and device_box.toolTip() == DEVICE_TIP
    track.set_model("sam2")  # the function behind the box
    assert processing.model == "edgetam" and asked == [("edgetam", "auto")]
    choose(device_box, "cpu")  # another device is no other model: the results stay comparable
    qtbot.waitUntil(lambda: track.worker.ready and len(asked) == 2)
    assert asked[-1] == ("edgetam", "cpu") and processing.device == "cpu"
    assert not model_box.isEnabled() and model_box.toolTip() == MODEL_FIXED
    objects.remove("A")  # the last object: no track has stored frames now
    assert ids(window) == [] and results_of(window).track_ids == []
    assert model_box.isEnabled() and model_box.toolTip() == MODEL_TIP
    choose(model_box, "sam2")
    qtbot.waitUntil(lambda: track.worker.ready and len(asked) == 3)
    assert asked[-1] == ("sam2", "cpu") and processing.model == "sam2"


def test_during_a_run_both_boxes_are_off_and_after_it_the_device_box_is_on_again(window, qtbot, clip_in_odd_folder):
    """After the run there are results, so the model box stays off, with its own sentence."""
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
    assert panel.device_box.isEnabled() and panel.device_box.toolTip() == DEVICE_TIP
    assert not panel.model_box.isEnabled() and panel.model_box.toolTip() == MODEL_FIXED
    assert panel.jobs.status == "complete" and asked == [("edgetam", "auto")]  # the run ended with its own model
