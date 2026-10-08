"""The Qt-free edit functions the GUI panels call (outline_tracker/tracking_edit.py): the object
table (`add_object`, `remove_object`), clicks (`add_prompt`, `undo_prompt`), the head click
(`set_head`), what "Track" would start (`pending_runs`) and the flags table (`flags_table`).
The three corrections and the track ids are in tests/test_corrections.py.

Expected values: last week's overlay colors (typed in the test), frame grids worked out by hand, the
ground truth of `dish_clip` (120 frames, every 2nd tracked: frames 0, 2, ..., 118; B and C pass
each other in frame 60), and the rules of SPEC 9 applied to shapes whose flags are known: a disk
18 px across is too small for shape numbers (LOWRES) and round (ORIENT). Frames are video frame
numbers; points are px in Tracker's convention (SPEC 3.1); times are s.
"""

import shutil

import numpy as np
import pytest
from helpers import ODD_FOLDER
from results_helpers import lost, measured, read_npz
from tracking_helpers import abc_session, center, clicks_at, make_session, run, track

from outline_tracker.derive import derive_track
from outline_tracker.qc import compute_flags
from outline_tracker.results import ResultsStore
from outline_tracker.schema import FLAG_SEPARATOR, FLAGS, RESULTS_NPZ, SESSION_JSON
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.session import Clip, Prompt, Session, TimeSettings, Track, VideoRef
from outline_tracker.tracking_edit import (
    TRACK_COLORS,
    add_object,
    add_prompt,
    end_track,
    flags_table,
    pending_runs,
    remove_object,
    retrack_from,
    set_head,
    undo_prompt,
)
from outline_tracker.tracking_plan import plan_runs

GRID = list(range(0, 120, 2))
FPS = 240.0  # the synthetic clips' frame rate, and fps_true of the sessions here


@pytest.fixture(scope="module")
def tracked_folder(dish_clip, tmp_path_factory):
    """A run folder in which A, B and C of the dish clip are tracked on every grid frame."""
    folder = tmp_path_factory.mktemp("tracked")
    assert run(dish_clip, abc_session(dish_clip, folder), folder, ExactFake(dish_clip))[0] == "complete"
    return folder


@pytest.fixture
def tracked(tracked_folder, tmp_path):
    """This test's own copy of that run, in a folder named with a space and non-ASCII characters:
    (run folder, session, store)."""
    folder = tmp_path / ODD_FOLDER
    shutil.copytree(tracked_folder, folder)
    return folder, Session.load(folder / SESSION_JSON), ResultsStore.load(folder / RESULTS_NPZ)


def frames_of(folder, track_id):
    return ResultsStore.load(folder / RESULTS_NPZ).arrays(track_id).frames.tolist()


# ---------------------------------------------------------------------------------------------
# The object table


def test_track_colors_are_last_weeks_overlay_colors_as_rgb_hex():
    # Last week's ten colors, typed here as red, green, blue. Its script held them as (blue, green, red):
    # (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 255, 0), (128, 0, 255),
    # (0, 0, 255), (255, 0, 0), (128, 255, 128).
    assert list(TRACK_COLORS) == ["#FFFF00", "#FF00FF", "#00FF00", "#0080FF", "#FF8000", "#00FFFF", "#FF0080",
                                  "#FF0000", "#0000FF", "#80FF80"]
    # last week's overlay drew on a BGR image: its first color (0, 255, 255) is yellow (X14)
    assert TRACK_COLORS[0] == "#FFFF00"


def test_add_object_adds_a_track_with_the_next_id_its_color_and_no_clicks():
    session = Session(video=VideoRef(width=320, height=240), clip=Clip(start=0, end=118, step=2))
    first = add_object(session)
    second = add_object(session, mode="fine", fine_window_px=160)
    assert session.tracks == [first, second] and session.tracks[0] is first
    assert first == Track(id="A", color="#FFFF00", mode="coarse")
    assert second == Track(id="B", color="#FF00FF", mode="fine", fine_window_px=160)
    assert session.complete is True and pending_runs(session, ResultsStore()) == []  # nothing to track without clicks

    for _ in range(25):
        add_object(session)
    by_id = {a_track.id: a_track.color for a_track in session.tracks}
    assert by_id["K"] == by_id["A"] and by_id["L"] == by_id["B"]  # ten colors, then again
    assert by_id["AA"] == TRACK_COLORS[26 % 10]


def test_add_object_refuses_an_unknown_mode():
    session = Session()
    with pytest.raises(ValueError, match="'coarse' or 'fine'"):
        add_object(session, mode="medium")
    assert session.tracks == []


def test_remove_object_takes_the_track_out_of_the_session_and_the_results(tracked):
    folder, session, store = tracked
    before = read_npz(folder / RESULTS_NPZ)
    saved = remove_object(session, store, "B", folder)
    assert saved == folder / RESULTS_NPZ
    assert [a_track.id for a_track in session.tracks] == ["A", "C"] and store.track_ids == ["A", "C"]
    after = read_npz(folder / RESULTS_NPZ)
    assert sorted(after) == sorted(key for key in before if not key.startswith("B__"))
    for key in after:  # A and C as they were
        np.testing.assert_array_equal(after[key], before[key], err_msg=key)
    assert frames_of(folder, "A") == GRID and frames_of(folder, "C") == GRID
    assert session.complete is True and session.corrections == []
    assert add_object(session).id == "D"  # B was tracked once: its name is not given to another object


def test_remove_object_of_an_object_without_results_does_not_write_the_results(tracked):
    folder, session, store = tracked
    name = add_object(session).id
    add_prompt(session, store, name, 20, (30.5, 40.5), 1, None, None)
    assert session.complete is False  # D has a click and waits
    written = (folder / RESULTS_NPZ).stat().st_mtime_ns
    assert remove_object(session, store, name, folder) is None
    assert (folder / RESULTS_NPZ).stat().st_mtime_ns == written
    assert [a_track.id for a_track in session.tracks] == ["A", "B", "C"]
    assert session.complete is True  # nothing waits any more
    with pytest.raises(ValueError, match="no track D"):
        remove_object(session, store, name, folder)


# ---------------------------------------------------------------------------------------------
# Clicks


def one_object(dish_clip, tmp_path, **settings):
    """A session for the dish clip with one new object and no results: (session, store, its id)."""
    session = make_session(dish_clip, tmp_path, [], dish_crop=False, **settings)
    return session, ResultsStore(), add_object(session).id


def test_clicks_on_one_frame_share_a_prompt_and_the_first_click_sets_the_start_frame(dish_clip, tmp_path):
    session, store, name = one_object(dish_clip, tmp_path)
    add_prompt(session, store, name, 6, (30.5, 40.5), 0, "sha256:six", "here")  # a negative click may come first
    add_prompt(session, store, name, 6, (50.0, 41.25), 1, "sha256:six", "here")
    (a_track,) = session.tracks
    assert a_track.prompts == [Prompt(frame=6, frame_hash="sha256:six", decoder="here",
                                      points_px=[[30.5, 40.5], [50.0, 41.25]], labels=[0, 1])]
    assert a_track.start_frame == 6
    assert [(plan.track_ids, plan.start_frame) for plan in pending_runs(session, store)] == [(("A",), 6)]


def test_an_object_that_waits_takes_no_clicks_on_another_frame(dish_clip, tmp_path):
    # its run starts on the frame of its clicks: clicks on a second frame would leave the first ones unused
    session, store, name = one_object(dish_clip, tmp_path)
    add_prompt(session, store, name, 6, (30.5, 40.5), 1, None, None)
    before = session.to_json()
    with pytest.raises(ValueError, match="clicks on frame 6 that are not tracked yet"):
        add_prompt(session, store, name, 10, (30.5, 40.5), 1, None, None)
    assert session.to_json() == before
    # undo the clicks of frame 6, and the object can start on frame 10
    assert undo_prompt(session, store, name) is True
    add_prompt(session, store, name, 10, (30.5, 40.5), 1, None, None)
    assert session.tracks[0].start_frame == 10 and [prompt.frame for prompt in session.tracks[0].prompts] == [10]


def test_a_track_that_waits_for_a_retrack_takes_clicks_on_that_frame_only(tracked, dish_clip):
    folder, session, store = tracked
    clicks = clicks_at(dish_clip, "A", 40)
    retrack_from(session, store, ["A"], 40, {"A": clicks}, folder)
    with pytest.raises(ValueError, match="clicks on frame 40 that are not tracked yet"):
        add_prompt(session, store, "A", 60, (30.5, 40.5), 0, None, None)
    add_prompt(session, store, "A", 40, (30.5, 40.5), 0, clicks.frame_hash, clicks.decoder)  # one more, same frame
    assert session.tracks[0].prompts[-1].points_px == [list(center(dish_clip, "A", 40)), [30.5, 40.5]]
    assert session.tracks[0].prompts[-1].labels == [1, 0] and len(session.tracks[0].prompts) == 2


@pytest.mark.parametrize("track_id, frame, label, message", [
    ("X", 6, 1, "no track X"),
    ("A", 6, 2, "1 .positive. or 0 .negative."),
    ("A", 6, True, "1 .positive. or 0 .negative."),
    ("B", 52, 1, "ended at frame 50"),
], ids=["unknown track", "label 2", "label True", "after the track's end"])
def test_add_prompt_refuses_what_it_cannot_use_and_changes_nothing(tracked, track_id, frame, label, message):
    folder, session, store = tracked
    end_track(session, store, "B", 50, folder)
    before = session.to_json()
    with pytest.raises(ValueError, match=message):
        add_prompt(session, store, track_id, frame, (30.5, 40.5), label, None, None)
    assert session.to_json() == before


@pytest.mark.parametrize("point, label, message", [
    ((30.5,), 1, "two numbers"),
    ("ab", 1, "two numbers"),
    (("30.5", "40.5"), 1, "two numbers"),
    ((float("nan"), 40.5), 1, "two numbers"),
    ((30.5, float("inf")), 1, "two numbers"),
    (None, 1, "two numbers"),
    ((30.5, 40.5), 2, "1 .positive. or 0 .negative."),
], ids=["one number", "text", "numbers as text", "NaN", "infinite", "no point", "label 2"])
def test_add_prompt_checks_the_click_before_it_changes_the_session(dish_clip, tmp_path, point, label, message):
    # the first click of an object sets its start frame and opens the clicks of that frame: a click
    # that is refused must do neither
    session, store, name = one_object(dish_clip, tmp_path)
    before = session.to_json()
    with pytest.raises(ValueError, match=message):
        add_prompt(session, store, name, 6, point, label, "sha256:six", "here")
    assert session.to_json() == before
    assert (session.tracks[0].start_frame, session.tracks[0].prompts) == (0, [])
    assert session.complete is True and pending_runs(session, store) == []


def test_undo_prompt_removes_the_newest_click_and_keeps_the_earlier_frames(tracked, dish_clip):
    folder, session, store = tracked
    start = session.tracks[0].prompts[0]
    add_prompt(session, store, "A", 40, (30.5, 40.5), 1, None, None)
    add_prompt(session, store, "A", 40, (31.5, 40.5), 0, None, None)
    assert undo_prompt(session, store, "A") is True
    assert [(prompt.frame, prompt.points_px) for prompt in session.tracks[0].prompts] == [
        (0, start.points_px), (40, [[30.5, 40.5]])]
    assert undo_prompt(session, store, "A") is True
    assert session.tracks[0].prompts == [start] and session.tracks[0].start_frame == 0
    with pytest.raises(ValueError, match="no track X"):
        undo_prompt(session, store, "X")


# ---------------------------------------------------------------------------------------------
# The head click


def test_set_head_stores_the_click_of_the_start_frame(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "A", frame=6)], dish_crop=False)
    set_head(session, "A", 6, (12.5, 30.25))
    assert session.tracks[0].head_px == [12.5, 30.25]
    assert Session.from_json(session.to_json()).tracks[0].head_px == [12.5, 30.25]
    # the heading takes its side from this click on the start frame (SPEC 7.3): another frame is refused
    with pytest.raises(ValueError, match="start frame .frame 6."):
        set_head(session, "A", 8, (14.5, 30.25))
    assert session.tracks[0].head_px == [12.5, 30.25]
    set_head(session, "A", 6, None)
    assert session.tracks[0].head_px is None


def test_set_head_needs_a_track_that_was_clicked_on(dish_clip, tmp_path):
    session, _, name = one_object(dish_clip, tmp_path)
    with pytest.raises(ValueError, match="no track X"):
        set_head(session, "X", 0, (12.5, 30.25))
    with pytest.raises(ValueError, match="Click the object first"):
        set_head(session, name, 0, (12.5, 30.25))
    assert session.tracks[0].head_px is None


def test_a_new_start_frame_drops_the_head_click_of_the_old_one(tracked, dish_clip):
    folder, session, store = tracked
    session.tracks[1] = track(dish_clip, "B", frame=20)  # B was started on frame 20 ...
    set_head(session, "B", 20, (12.5, 30.25))
    retrack_from(session, store, ["B"], 10, {"B": clicks_at(dish_clip, "B", 10)}, folder)  # ... and now from 10
    assert (session.tracks[1].start_frame, session.tracks[1].head_px) == (10, None)
    assert [prompt.frame for prompt in session.tracks[1].prompts] == [10]


# ---------------------------------------------------------------------------------------------
# Pending runs


def test_pending_runs_are_the_runs_track_would_start(dish_clip, tmp_path):
    tracks = [track(dish_clip, "A"), track(dish_clip, "B", mode="fine"), track(dish_clip, "C")]
    session = make_session(dish_clip, tmp_path, tracks, dish_crop=False)
    store = ResultsStore()
    runs = pending_runs(session, store)
    assert runs == plan_runs(session, store)
    assert [(plan.track_ids, plan.start_frame, plan.mode, len(plan.frames)) for plan in runs] == [
        (("A", "C"), 0, "coarse", 60), (("B",), 0, "fine", 60)]


def test_nothing_is_pending_when_every_click_is_tracked(tracked):
    _, session, store = tracked
    assert pending_runs(session, store) == []


# ---------------------------------------------------------------------------------------------
# The flags table


def test_flags_table_lists_every_flag_of_every_frame_by_track_then_frame(tracked):
    folder, session, store = tracked
    rows = flags_table(folder)
    assert all(type(name) is str and type(frame) is int and type(t_s) is float and code in FLAGS
               for name, frame, t_s, code in rows)
    assert all(t_s == frame / FPS for _, frame, t_s, _ in rows)
    keys = [("ABC".index(name), frame, FLAGS.index(code)) for name, frame, _, code in rows]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)  # by track, then frame, then the order of SPEC 9

    def frames_with(code, name):
        return [frame for row_name, frame, _, row_code in rows if (row_name, row_code) == (name, code)]

    for name in "ABC":  # no head click: the head is a guess on every frame
        assert frames_with("HEADGUESS", name) == GRID
        assert frames_with("LOWRES", name) == GRID  # a 14 px body is too small for shape numbers
        assert frames_with("LOST", name) == []
    # B and C pass each other with a gap of 0.6 px in frame 60; A swims along the wall, far from both
    assert 60 in frames_with("CONTACT", "B") and frames_with("CONTACT", "B") == frames_with("CONTACT", "C")
    assert frames_with("CONTACT", "A") == []

    # the table holds exactly the flags the export writes (`compute_flags` on the same records)
    arrays = {name: store.arrays(name) for name in "ABC"}
    world = session.world_frame()
    tracks = {a_track.id: a_track for a_track in session.tracks}
    derived = {name: derive_track(arrays[name], tracks[name], world, FPS, None, session.processing) for name in "ABC"}
    cells = compute_flags(derived, arrays, world, session.processing)
    assert rows == [(name, frame, frame / FPS, code) for name in "ABC" for frame, cell in zip(GRID, cells[name])
                    for code in cell.split(FLAG_SEPARATOR) if code]


def test_flags_table_follows_the_session_and_the_results_in_the_folder(tracked):
    folder, session, store = tracked
    set_head(session, "A", 0, [10.5, 10.5])
    end_track(session, store, "B", 50, folder)
    add_object(session)  # D: in the table of objects, without results
    session.save(folder / SESSION_JSON)
    rows = flags_table(folder)
    assert not any((name, code) == ("A", "HEADGUESS") for name, _, _, code in rows)  # A has a head click now
    assert sorted({frame for name, frame, _, _ in rows if name == "B"}) == GRID[:26]  # B's rows stop at frame 50
    assert sorted({frame for name, frame, _, _ in rows if name == "C"}) == GRID
    assert {name for name, _, _, _ in rows} == {"A", "B", "C"}


def hand_made_folder(folder):
    """A run folder with hand-made results. A: a disk 18 px across on frames 10, 12 and 16, lost on
    frame 14. A2, with a head click: the same disk, 57 px away, on frames 12 and 14. B is in the
    session without results."""
    store = ResultsStore()
    for record in (measured(10, (311.3, 230.6)), measured(12, (311.3, 230.6)), lost(14), measured(16, (311.3, 230.6))):
        store.put("A", record)
    for frame in (12, 14):
        store.put("A2", measured(frame, (368.7, 229.4)))
    store.save(folder / RESULTS_NPZ)
    session = Session(time=TimeSettings(fps_true=FPS), clip=Clip(start=10, end=22, step=2))
    session.tracks = [Track(id="B", start_frame=10), Track(id="A2", start_frame=12, head_px=[375.5, 229.4]),
                      Track(id="A", start_frame=10)]
    session.calibration.stick = {"p1_px": [0.0, 0.0], "p2_px": [100.0, 0.0], "length_mm": 3.24}
    session.save(folder / SESSION_JSON)
    return folder


def test_flags_table_on_hand_made_results(tmp_path):
    # Every visible disk is LOWRES (18 px < 20) and ORIENT (round); A has no head click: HEADGUESS on
    # every row, the lost one too, which carries LOST and no other flag. The disks are 39 px apart
    # at the nearest, more than the 15 px (2 grid cells of 7.5 px) of CONTACT. Rows are in the
    # order of the ids as text (A, its piece A2), like the tracks of results.npz, not of the session.
    seen = ["LOWRES", "ORIENT"]
    of_a = ((10, [*seen, "HEADGUESS"]), (12, [*seen, "HEADGUESS"]), (14, ["LOST", "HEADGUESS"]),
            (16, [*seen, "HEADGUESS"]))
    expected = [("A", frame, frame / FPS, code) for frame, codes in of_a for code in codes]
    expected += [("A2", frame, frame / FPS, code) for frame in (12, 14) for code in seen]
    assert flags_table(hand_made_folder(tmp_path)) == expected


def test_flags_table_is_empty_before_anything_is_tracked(tmp_path):
    Session(time=TimeSettings(fps_true=FPS)).save(tmp_path / SESSION_JSON)
    assert flags_table(tmp_path) == []


def test_flags_table_says_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        flags_table(tmp_path)  # no session.json
    folder = hand_made_folder(tmp_path)
    session = Session.load(folder / SESSION_JSON)
    session.calibration.stick = None
    session.save(folder / SESSION_JSON)
    with pytest.raises(ValueError, match="no scale yet"):
        flags_table(folder)
    session.calibration.stick = {"p1_px": [0.0, 0.0], "p2_px": [100.0, 0.0], "length_mm": 3.24}
    session.time.fps_true = None
    session.save(folder / SESSION_JSON)
    with pytest.raises(ValueError, match="fps_true"):
        flags_table(folder)
