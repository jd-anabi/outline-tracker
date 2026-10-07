"""What a job does around the frames (SPEC 6.4, 6.5): cancel, autosave, progress, the session changes
it hands over, the checks before a run, and what is kept when the model fails. The positions
themselves are checked in tests/test_tracking_coarse.py.

Expected values are counts of frames worked out from the clip (120 frames, or 210 for the autosave
clip) and the session's frame grid, and the ground truth of outline_tracker/synthetic.py. Frames
are video frame numbers; positions are px in Tracker's convention (SPEC 3.1).
"""

import copy
import errno
import os
import time
import weakref
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from helpers import SMALL
from tracking_helpers import Recorder, Watched, dish_circle, make_session, run, table_truth, track

from outline_tracker import fileio, synthetic, tracking
from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.session import Session
from outline_tracker.tracking import Job, run_job

GRID = list(range(0, 120, 2))  # `dish_clip` has 120 frames; the sessions here take every 2nd


def three_tracks(clip, tmp_path, **settings):
    """A session with a click on each of the dish scene's objects A, B and C in frame 0."""
    return make_session(clip, tmp_path, [track(clip, name) for name in "ABC"], **settings)


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory):
    """`dish_scene` at 320 x 240 px with 210 frames: more than one autosave period at step 1."""
    scene = synthetic.dish_scene(size=SMALL, n_frames=210)
    return synthetic.render(scene, tmp_path_factory.mktemp("long") / "long_tracker.mp4")


# ---------------------------------------------------------------------------------------------
# Cancel


def test_cancel_after_frame_10_keeps_the_frames_up_to_10_and_marks_the_session_partial(dish_clip, tmp_path):
    session = three_tracks(dish_clip, tmp_path)
    segmenter = Watched(ExactFake(dish_clip))
    status, seen = run(dish_clip, session, tmp_path, segmenter, Recorder(cancel_after_frame=10))
    assert status == "cancelled" and seen.finished == ["cancelled"]
    assert segmenter.calls == ["start", *["step"] * 5, "close"]  # frames 0 to 10, and no frame after the cancel

    kept = [0, 2, 4, 6, 8, 10]
    store = ResultsStore.load(tmp_path / "results.npz")
    assert store.track_ids == ["A", "B", "C"]
    for name in "ABC":
        arrays = store.arrays(name)
        assert arrays.frames.tolist() == kept
        true_u, true_v, _ = table_truth(dish_clip, name, kept)
        np.testing.assert_allclose(arrays.u, true_u, rtol=0, atol=0.01)
        np.testing.assert_allclose(arrays.v, true_v, rtol=0, atol=0.01)

    saved = Session.load(tmp_path / "session.json")
    assert saved.complete is False
    (record,) = saved.runs
    assert (record.tracks, record.frames_done) == (["A", "B", "C"], 6)
    assert record.finished is not None  # the run is over
    assert session.complete is False and session.runs[0].frames_done == 6  # the caller's session says the same


def test_a_cancel_before_the_first_frame_tracks_nothing(dish_clip, tmp_path):
    session = three_tracks(dish_clip, tmp_path)
    segmenter, recorder = Watched(ExactFake(dish_clip)), Recorder()
    callbacks = replace(recorder.callbacks(), should_cancel=lambda: True)
    status = run_job(Job(session, tmp_path, dish_clip.path, segmenter.make), callbacks)
    assert status == "cancelled" and recorder.finished == ["cancelled"]
    assert "start" not in segmenter.calls and recorder.results == []
    assert list(tmp_path.iterdir()) == []  # no run began: there is nothing to save, in results or session
    assert session.runs == [] and session.complete is True


# ---------------------------------------------------------------------------------------------
# Autosave and the session changes


def test_autosave_after_200_tracked_frames_writes_results_and_session(long_clip, tmp_path):
    assert tracking.AUTOSAVE_EVERY == 200  # SPEC 6.4
    session = make_session(long_clip, tmp_path, [track(long_clip, "B")], step=1)
    results_file, session_file = tmp_path / "results.npz", tmp_path / "session.json"
    on_disk = {}

    def look(track_id, frame, record):
        # Frame 199 is the 200th: its result is reported before the autosave. Frame 200 comes after it.
        if frame in (199, 200):
            frames = ResultsStore.load(results_file).arrays("B").frames.tolist() if results_file.exists() else None
            saved = Session.load(session_file)
            on_disk[frame] = (frames, saved.runs[-1].frames_done, saved.complete)

    callbacks = replace(Recorder().callbacks(), frame_result=look)
    status = run_job(Job(session, tmp_path, long_clip.path, lambda: ExactFake(long_clip)), callbacks)
    assert status == "complete"
    # until then: no results file, and a session that already says a run is under way
    assert on_disk[199] == (None, 0, False)
    assert on_disk[200] == (list(range(200)), 200, False)
    assert ResultsStore.load(results_file).arrays("B").frames.tolist() == list(range(210))
    saved = Session.load(session_file)
    assert saved.runs[-1].frames_done == 210 and saved.complete is True


def test_with_a_save_session_function_the_job_itself_writes_no_session_file(long_clip, tmp_path):
    session = make_session(long_clip, tmp_path, [track(long_clip, "B")], step=1)
    before = copy.deepcopy(session).to_json()
    status, seen = run(long_clip, session, tmp_path, ExactFake(long_clip), record_session=True)
    assert status == "complete"
    assert [path.name for path in tmp_path.iterdir()] == ["results.npz"]
    assert session.to_json() == before  # the job worked on a copy of the session

    # what was handed over: the run's record when it starts, at the autosave, and when it ends
    assert [(change.complete, change.run_index, change.run.frames_done, change.run.finished is None)
            for change in seen.changes] == [(False, 0, 0, True), (False, 0, 200, True), (True, 0, 210, False)]
    assert seen.changes[0].run is not seen.changes[1].run  # each is the receiver's own

    # applied in that order by the one writer of session.json, they give the finished session
    mine = copy.deepcopy(session)
    for change in seen.changes:
        change.apply(mine)
    assert mine.complete is True
    (record,) = mine.runs
    assert (record.tracks, record.start_frame, record.mode, record.frames_done) == (["B"], 0, "coarse", 210)
    assert record.seconds_per_frame > 0
    started, finished = datetime.fromisoformat(record.started), datetime.fromisoformat(record.finished)
    assert started.tzinfo is not None and started <= finished
    # a stand-in has no device and no weights
    assert (record.device, record.model_id, record.weights_sha256) == ("", None, None)
    assert mine.to_json()["runs"][0].keys() == {"tracks", "start_frame", "mode", "started", "finished",
                                                "frames_done", "seconds_per_frame", "device"}


def test_the_session_is_complete_only_when_the_last_run_of_the_job_has_ended(dish_clip, tmp_path):
    tracks = [track(dish_clip, "A"), track(dish_clip, "B"), track(dish_clip, "C", frame=60)]
    session = make_session(dish_clip, tmp_path, tracks)
    status, seen = run(dish_clip, session, tmp_path, ExactFake(dish_clip), record_session=True)
    assert status == "complete"
    # the run of A and B starts and ends, then the run of C from frame 60 starts and ends
    assert [(change.complete, change.run_index, change.run.tracks, change.run.frames_done)
            for change in seen.changes] == [(False, 0, ["A", "B"], 0), (False, 0, ["A", "B"], 60),
                                            (False, 1, ["C"], 0), (True, 1, ["C"], 30)]


def test_a_later_job_tracks_what_has_no_results_yet_and_keeps_the_rest(dish_clip, tmp_path):
    session = three_tracks(dish_clip, tmp_path)
    results_file, session_file = tmp_path / "results.npz", tmp_path / "session.json"
    first = Watched(ExactFake(dish_clip))
    assert run(dish_clip, session, tmp_path, first, track_ids=["B"])[0] == "complete"
    assert [[prompt.obj_id for prompt in prompts] for _, prompts in first.starts] == [["B"]]
    before = ResultsStore.load(results_file)
    assert before.track_ids == ["B"]

    second = Watched(ExactFake(dish_clip))
    assert run(dish_clip, session, tmp_path, second)[0] == "complete"
    assert [[prompt.obj_id for prompt in prompts] for _, prompts in second.starts] == [["A", "C"]]
    store = ResultsStore.load(results_file)
    assert store.track_ids == ["A", "B", "C"]
    np.testing.assert_array_equal(store.arrays("B").u, before.arrays("B").u)
    np.testing.assert_array_equal(store.arrays("B").mask_bits, before.arrays("B").mask_bits)
    for name in "AC":
        true_u, true_v, _ = table_truth(dish_clip, name, GRID)
        np.testing.assert_allclose(store.arrays(name).u, true_u, rtol=0, atol=0.01)
        np.testing.assert_allclose(store.arrays(name).v, true_v, rtol=0, atol=0.01)
    saved = Session.load(session_file)
    assert [run_.tracks for run_ in saved.runs] == [["B"], ["A", "C"]] and saved.complete is True
    assert session.to_json() == saved.to_json()  # the default keeps the caller's session and the file together

    # nothing is left: a third job does not load a model and changes no file
    third, written = Watched(ExactFake(dish_clip)), (results_file.read_bytes(), session_file.read_bytes())
    status, seen = run(dish_clip, session, tmp_path, third)
    assert status == "complete" and seen.finished == ["complete"]
    assert third.made == 0 and seen.progress == [] and len(seen.log) == 1
    assert (results_file.read_bytes(), session_file.read_bytes()) == written


def test_a_job_for_a_track_the_session_does_not_have_is_refused(dish_clip, tmp_path):
    session = three_tracks(dish_clip, tmp_path)
    segmenter = Watched(ExactFake(dish_clip))
    with pytest.raises(ValueError, match=r"\bZ\b"):
        run(dish_clip, session, tmp_path, segmenter, track_ids=["A", "Z"])
    assert segmenter.made == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("locked, kept_as", [("results.npz", "results.new.npz"), ("session.json", "session.new.json")])
def test_a_file_that_stays_locked_is_written_next_to_it_and_the_log_says_so(dish_clip, tmp_path, monkeypatch, locked,
                                                                            kept_as):
    real_replace = os.replace

    def replace_unless_locked(src, dst):  # what Windows does while another program holds the file open
        if Path(dst).name == locked:
            raise PermissionError(errno.EACCES, "The file is being used by another process", str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(fileio.os, "replace", replace_unless_locked)
    monkeypatch.setattr(fileio.time, "sleep", lambda seconds: None)
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "B")])
    status, seen = run(dish_clip, session, tmp_path, ExactFake(dish_clip))
    assert status == "complete"
    assert (tmp_path / kept_as).is_file() and not (tmp_path / locked).exists()
    assert any(kept_as in line for line in seen.log)


# ---------------------------------------------------------------------------------------------
# Progress


def test_progress_reports_done_total_seconds_per_frame_and_time_left(dish_clip, tmp_path):
    tracks = [track(dish_clip, "A"), track(dish_clip, "B"), track(dish_clip, "C", frame=60)]
    session = make_session(dish_clip, tmp_path, tracks)
    began = time.perf_counter()
    status, seen = run(dish_clip, session, tmp_path, ExactFake(dish_clip))
    elapsed = time.perf_counter() - began
    assert status == "complete"
    # 60 frames for A and B, then 30 for C from frame 60: one call after every tracked frame
    assert [done for done, _, _, _ in seen.progress] == list(range(1, 91))
    assert {total for _, total, _, _ in seen.progress} == {90}
    for done, total, s_per_frame, eta_s in seen.progress:
        assert 0 < s_per_frame * done <= elapsed  # s per tracked frame, over the job so far
        assert eta_s == pytest.approx(s_per_frame * (total - done))  # s left at that rate
    spent = [s_per_frame * done for done, _, s_per_frame, _ in seen.progress]
    assert spent == sorted(spent)
    assert seen.progress[-1][3] == 0.0


# ---------------------------------------------------------------------------------------------
# Checks before a run


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), None, "0.1"])
def test_a_core_open_frac_that_is_not_a_fraction_is_refused_before_the_run(dish_clip, tmp_path, value):
    session = three_tracks(dish_clip, tmp_path)
    session.processing.core_open_frac = value
    segmenter = Watched(ExactFake(dish_clip))
    with pytest.raises(ValueError, match="core_open_frac"):
        run(dish_clip, session, tmp_path, segmenter)
    assert segmenter.made == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value, radius", [(0.0, 1), (0.3, 4)], ids=["zero", "three times the default"])
def test_the_sessions_core_open_frac_is_the_one_used(dish_clip, tmp_path, value, radius):
    # B is an ellipse with semi-axes 7.25 x 3.09 px, so L1 = 4 sqrt(lambda1) = 2 x 7.25 = 14.5 px. The opening
    # disk has radius max(1, floor(frac L1 + 0.5)) px (SPEC 7.3): 1 for frac = 0, floor(4.85) = 4 for 0.3.
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "B")])
    session.processing.core_open_frac = value
    status, _ = run(dish_clip, session, tmp_path, ExactFake(dish_clip))
    assert status == "complete"
    assert set(ResultsStore.load(tmp_path / "results.npz").arrays("B").core_r_px.tolist()) == {radius}


@pytest.mark.parametrize("value", [None, 0, -240.0, float("inf"), float("nan"), "240"])
def test_an_fps_true_that_is_not_a_positive_number_is_refused_before_the_run(dish_clip, tmp_path, value):
    session = three_tracks(dish_clip, tmp_path)
    session.time.fps_true = value
    segmenter = Watched(ExactFake(dish_clip))
    with pytest.raises(ValueError, match="fps_true"):
        run(dish_clip, session, tmp_path, segmenter)
    assert segmenter.made == 0 and list(tmp_path.iterdir()) == []


def test_a_fine_object_is_not_tracked_yet(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "A"), track(dish_clip, "B", mode="fine")])
    segmenter = Watched(ExactFake(dish_clip))
    with pytest.raises(NotImplementedError, match="fine"):
        run(dish_clip, session, tmp_path, segmenter)
    assert segmenter.made == 0 and list(tmp_path.iterdir()) == []  # the coarse object was not started either


# ---------------------------------------------------------------------------------------------
# The segmenter: its log and device, its failures, its images


class WithGpu(Watched):
    """A stand-in with what the real segmenter has besides the protocol: `device`, `model_id`,
    `weights_sha256` and `log`. On its 4th image it gives up the Apple GPU, as HFSegmenter.step does."""

    log = staticmethod(print)
    GAVE_UP = "  The Apple GPU failed (RuntimeError: not implemented for MPS); using the processor instead."

    def __init__(self, inner):
        super().__init__(inner)
        self.device, self.model_id, self.weights_sha256 = "mps", "stand-in/model", "ab" * 32

    def step(self, image):
        if len(self.calls) == 3:
            self.device = "cpu"
            self.log(self.GAVE_UP)
        return super().step(image)


def test_the_segmenters_log_goes_to_the_job_and_the_run_records_the_device_used(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "B")])
    segmenter = WithGpu(ExactFake(dish_clip))
    status, seen = run(dish_clip, session, tmp_path, segmenter, record_session=True)
    assert status == "complete"
    assert WithGpu.GAVE_UP in seen.log
    assert [change.run.device for change in seen.changes] == ["mps", "cpu"]  # as it began, and as it ended
    record = seen.changes[-1].run
    assert (record.model_id, record.weights_sha256) == ("stand-in/model", "ab" * 32)
    assert segmenter.log is print  # the job gave the segmenter's log back


class Failing(Watched):
    """A stand-in that raises `error` instead of segmenting its `at`-th image (the first is 1)."""

    def __init__(self, inner, at, error):
        super().__init__(inner)
        self.at, self.error = at, error

    def step(self, image):
        if len(self.calls) + 1 == self.at:
            raise self.error
        return super().step(image)


@pytest.mark.parametrize("error, expected", [(RuntimeError("out of memory"), "failed"),
                                             (KeyboardInterrupt(), "cancelled")], ids=["an error", "Ctrl+C"])
def test_what_was_tracked_is_kept_when_the_model_fails_or_the_user_interrupts(dish_clip, tmp_path, error, expected):
    session = three_tracks(dish_clip, tmp_path)
    segmenter = Failing(ExactFake(dish_clip), at=6, error=error)
    status, seen = run(dish_clip, session, tmp_path, segmenter)
    assert status == expected and seen.finished == [expected]
    assert segmenter.calls[-1] == "close"

    kept = [0, 2, 4, 6, 8]  # five images went through
    store = ResultsStore.load(tmp_path / "results.npz")
    for name in "ABC":
        arrays = store.arrays(name)
        assert arrays.frames.tolist() == kept
        true_u, true_v, _ = table_truth(dish_clip, name, kept)
        np.testing.assert_allclose(arrays.u, true_u, rtol=0, atol=0.01)
        np.testing.assert_allclose(arrays.v, true_v, rtol=0, atol=0.01)
    saved = Session.load(tmp_path / "session.json")
    assert saved.complete is False
    assert saved.runs[0].frames_done == 5 and saved.runs[0].finished is not None
    if expected == "failed":  # the log names the error and the frame it happened on
        assert any("RuntimeError" in line and "out of memory" in line and "10" in line for line in seen.log)


class Counting(Watched):
    """Writes down, at every image, how many of the images it was given before still exist."""

    def __init__(self, inner):
        super().__init__(inner)
        self.images, self.alive = [], []

    def _see(self, image):
        self.alive.append(sum(ref() is not None for ref in self.images))
        self.images.append(weakref.ref(image))

    def start(self, image, prompts):
        self._see(image)
        return super().start(image, prompts)

    def step(self, image):
        self._see(image)
        return super().step(image)


@pytest.mark.parametrize("dish_crop", [True, False], ids=["the cut-out", "the decoded frame"])
def test_no_image_is_kept_beyond_its_frame(dish_clip, tmp_path, dish_crop):
    # SPEC 6.5: decode, process, discard. Without the crop the image is the decoded frame itself.
    session = three_tracks(dish_clip, tmp_path, circle=dish_circle(dish_clip.scene), dish_crop=dish_crop)
    segmenter = Counting(ExactFake(dish_clip))
    status, seen = run(dish_clip, session, tmp_path, segmenter)
    assert status == "complete"
    assert len(segmenter.images) == 60
    assert set(segmenter.alive) == {0}  # when an image arrives, the earlier ones are gone
    assert len(seen.results) == 180  # the records are still here: they hold crops of the masks, never the image
    assert all(ref() is None for ref in segmenter.images)

