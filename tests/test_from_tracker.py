"""`from_tracker`: a Tracker export becomes a session, a run and the files of SPEC 8 (SPEC 11, X9, X18).

Where the expected values come from: the clips are dark disks drawn at known places (the template's
`disk_video`, 240 frames per s), the exports are written with Tracker's own map (`tracker_map`:
0.05 mm per px, origin at (160, 120) px), and the frames follow from the export as last week's
`make_plan` decides. Positions are compared with where the disks were drawn, never with a file the
tool wrote before.

fps_true and the console lines are in tests/test_from_tracker_fps.py, last week's own tests and the
comparison with last week's script in tests/test_from_tracker_port.py, the command in
tests/test_from_tracker_cli.py.

Coordinates: px in Tracker's convention (SPEC 3.1: u to the right, v downward, pixel centers at
+0.5); mm in the user's axes with y up; frames are video frame numbers; t_s = frame / fps_true.
"""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import java_sci
from from_tracker_helpers import (FPS, MODEL, RUN_FILES, StopsAfter, disk_video, frame_sha256, mirrored_map,
                                  names_in, one_disk, read_track, rotated_map, run, session_json, two_disks,
                                  write_start_file)
from test_tracker_io import MM_PER_PX, tracker_map

from outline_tracker import fileio, synthetic, video
from outline_tracker.from_tracker_session import TRACK_COLORS
from outline_tracker.results import ResultsStore
from outline_tracker.synthetic_shapes import Disk, Straight


# --------------------------------------------------------------------------- the session

def test_the_session_holds_the_plan_the_fit_and_one_click_per_object(tmp_path):
    # Tracker's axes: origin at (150.25, 110.5) px, +x turned 12 degrees counterclockwise on screen
    to_mm = rotated_map(12.0, (150.25, 110.5))
    clip, export = one_disk(tmp_path, to_mm=to_mm)
    result = run(clip, export)
    assert result.run_folder == tmp_path / "clip_tracker_outline_ana"
    session = session_json(result.run_folder)

    assert session["student"] == "ana" and session["complete"] is True
    assert session["clip"] == {"start": 40, "end": 60, "step": 4}  # the Tracker track's frames
    assert session["video"]["relpath"] == "../clip_tracker.mp4"
    assert (session["video"]["width"], session["video"]["height"], session["video"]["n_frames"]) == (320, 240, 70)
    assert session["video"]["size"] == clip.stat().st_size
    assert session["video"]["sha256_first_64MiB"] == fileio.sha256_first_64mib(clip)

    calibration = session["calibration"]
    assert calibration["stick"] is None and calibration["check"] is None  # there is no stick
    assert sorted(calibration["tracker_fit"]) == ["mm_per_px", "n_points", "rms_mm"]
    assert calibration["tracker_fit"]["mm_per_px"] == pytest.approx(MM_PER_PX, rel=1e-5)
    assert calibration["tracker_fit"]["n_points"] == 6  # the six rows of the export
    assert 0 <= calibration["tracker_fit"]["rms_mm"] < 1e-5  # Tracker writes 7 digits
    assert session["axes"]["origin_px"] == pytest.approx([150.25, 110.5], abs=1e-3)
    assert session["axes"]["angle_deg"] == pytest.approx(12.0, abs=1e-3)
    assert session["circle"] is None
    assert session["processing"]["model"] == MODEL and session["processing"]["device"] == "auto"

    (track,) = session["tracks"]
    assert (track["id"], track["mode"], track["start_frame"], track["color"]) == ("A", "coarse", 40, "#FFFF00")
    assert track["head_px"] is None and track["ended_at"] is None and track["fine_window_px"] is None
    (prompt,) = track["prompts"]
    assert prompt["frame"] == 40 and prompt["labels"] == [1]
    assert prompt["points_px"] == [pytest.approx([80.5, 100.5], abs=1e-9)]  # the export's first pixelx, pixely
    assert prompt["frame_hash"] == frame_sha256(clip, 40)
    assert prompt["decoder"] == video.decoder_tag()

    (one_run,) = session["runs"]
    assert (one_run["tracks"], one_run["mode"], one_run["start_frame"]) == (["A"], "coarse", 40)
    assert one_run["frames_done"] == 6

    # the files are in those axes: x, y of the place where the disk was drawn
    got = read_track(result.files[0])
    assert list(got["frame"]) == [40, 44, 48, 52, 56, 60]
    ex, ey = to_mm(60.5 + 0.5 * got["frame"], 100.5)
    assert np.allclose(got["x"], ex, atol=0.1 * MM_PER_PX) and np.allclose(got["y"], ey, atol=0.1 * MM_PER_PX)
    assert result.plan.frames == [40, 44, 48, 52, 56, 60] and result.seconds_per_frame > 0


def test_track_colors_are_last_weeks_overlay_colors(tmp_path):
    from shrimp import segment as reference  # last week's colors are (blue, green, red)

    assert TRACK_COLORS == [f"#{red:02X}{green:02X}{blue:02X}" for blue, green, red in reference.COLORS]
    assert TRACK_COLORS[0] == "#FFFF00"  # the first track is yellow
    clip, export = two_disks(tmp_path)
    result = run(clip, export, seconds=8 / FPS, step=4, fps=FPS)
    assert [(t["id"], t["color"]) for t in session_json(result.run_folder)["tracks"]] == [("A", "#FFFF00"),
                                                                                         ("B", "#FF00FF")]


# --------------------------------------------------------------------------- where the files go

def test_the_whole_file_set_of_the_spec_is_written(tmp_path):
    clip, export = two_disks(tmp_path)
    result = run(clip, export, seconds=8 / FPS, step=4, fps=FPS, overlay=True)
    assert names_in(result.run_folder) == sorted(RUN_FILES)
    assert names_in(result.run_folder / MODEL) == ["A.csv", "B.csv"]  # nothing but <id>.csv
    assert result.overlay == result.run_folder / "overlay.mp4" and video.probe(result.overlay).n_frames == 2
    positions = pd.read_csv(result.run_folder / "positions.csv")
    assert positions.track_id.tolist() == ["A", "A", "B", "B"] and positions.frame.tolist() == [40, 44, 40, 44]
    assert set(positions["mode"]) == {"coarse"}
    # the test disk is small: its rows carry the shape flags, which are not CHECK messages (X22)
    assert set(positions["flags"]) == {"LOWRES;ORIENT;HEADGUESS"} and result.flags == []
    assert len(pd.read_csv(result.run_folder / "radial.csv")) == 0  # no fine track: the header alone


def test_no_overlay_skips_only_the_overlay(tmp_path):
    clip, export = two_disks(tmp_path)
    result = run(clip, export, seconds=8 / FPS, step=4, fps=FPS, overlay=False)
    assert names_in(result.run_folder) == sorted(set(RUN_FILES) - {"overlay.mp4"})
    assert result.overlay is None


def test_the_student_is_the_exports_home_folder_unless_given(tmp_path):
    clip, export = two_disks(tmp_path, student="ana")  # <tmp>/ana/extra/start.csv: the home is ana
    lines = []
    result = run(clip, export, lines, seconds=8 / FPS, step=4, fps=FPS)
    assert result.run_folder == tmp_path / "clip_tracker_outline_ana"
    assert session_json(result.run_folder)["student"] == "ana"
    assert any("ana" in line and "--student" in line for line in lines)  # it says which name it used
    assert any(str(result.run_folder) in line for line in lines)  # and which folder

    lines = []
    result = run(clip, export, lines, seconds=8 / FPS, step=4, fps=FPS, student="Zoë M. / B")
    assert result.run_folder == tmp_path / "clip_tracker_outline_Zoë_M.___B"  # made safe for a folder name
    assert session_json(result.run_folder)["student"] == "Zoë M. / B"  # the name itself is kept
    assert any("Zoë M. / B" in line for line in lines)


def test_names_stay_as_they_are_and_file_names_are_made_safe_as_last_week(tmp_path):
    clip, export = one_disk(tmp_path)
    export = export.rename(export.with_name("my shrimp #1.csv"))  # one point mass is named after its file
    result = run(clip, export)
    assert [path.name for path in result.files] == ["my_shrimp_1.csv"]  # last week's rule for a file name
    assert result.files[0].read_text().splitlines()[0] == ",my shrimp #1,,,,,"
    assert [track["id"] for track in session_json(result.run_folder)["tracks"]] == ["my shrimp #1"]
    assert set(pd.read_csv(result.run_folder / "positions.csv").track_id) == {"my shrimp #1"}


def test_out_is_the_run_folder_itself(tmp_path):
    clip, export = two_disks(tmp_path)
    result = run(clip, export, seconds=8 / FPS, step=4, fps=FPS, out=tmp_path / "elsewhere" / "run 1")
    assert result.run_folder == tmp_path / "elsewhere" / "run 1"
    assert names_in(result.run_folder / MODEL) == ["A.csv", "B.csv"]
    assert session_json(result.run_folder)["video"]["relpath"] == "../../clip_tracker.mp4"
    assert not (tmp_path / "clip_tracker_outline_ana").exists()


def test_a_folder_of_tracker_files_is_refused(tmp_path):
    clip, export = one_disk(tmp_path)
    before = names_in(export.parent)
    with pytest.raises(ValueError, match="folder of Tracker files"):
        run(clip, export, out=export.parent)  # last week --out was the model folder; the export's own is a mistake
    assert names_in(export.parent) == before == ["A.csv"]
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "readings.TXT").write_text("1\n")
    with pytest.raises(ValueError, match="folder of Tracker files"):
        run(clip, export, out=notes)
    assert names_in(notes) == ["readings.TXT"]


def test_a_second_run_replaces_the_first(tmp_path):
    clip, export = one_disk(tmp_path, frames=range(40, 61, 4))
    first = run(clip, export, seconds=8 / FPS)  # frames 40 and 44
    assert list(read_track(first.files[0])["frame"]) == [40, 44]
    lines = []
    second = run(clip, export, lines)  # the whole Tracker track
    assert second.run_folder == first.run_folder
    assert list(read_track(second.files[0])["frame"]) == [40, 44, 48, 52, 56, 60]
    assert len(session_json(second.run_folder)["runs"]) == 1  # a new session, not a longer one
    assert any("replace" in line for line in lines)  # and it says so


def test_a_run_made_or_corrected_in_the_app_is_not_replaced(tmp_path):
    clip, export = one_disk(tmp_path)
    first = run(clip, export)
    path = first.run_folder / "session.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["corrections"] = [{"time": "2026-10-08T10:00:00-07:00", "tracks": ["A"], "action": "end", "frame": 52,
                            "prompts": []}]
    path.write_text(json.dumps(data), encoding="utf-8")
    kept = {name: (first.run_folder / name).read_bytes() for name in ("session.json", "results.npz", "positions.csv")}
    with pytest.raises(ValueError, match="--out"):
        run(clip, export)
    assert {name: (first.run_folder / name).read_bytes() for name in kept} == kept


# --------------------------------------------------------------------------- fine tracks

def big_and_small(folder):
    """A clip of two dark disks through H.264 (crf 10): A of radius 12 px, B of radius 20 px, and a
    start file that marks both at their centers on frame 0. Returns (the ground truth, the export)."""
    objects = (synthetic.SceneObject("A", Disk(12.0), Straight((60.3, 60.5), (0.5, 0.0)), gray=40),
               synthetic.SceneObject("B", Disk(20.0), Straight((200.4, 150.2), (-0.4, 0.2)), gray=40))
    scene = synthetic.Scene((320, 240), FPS, 40, objects, mm_per_px=MM_PER_PX, background=220)
    clip = synthetic.render(scene, folder / "disks_tracker.mp4", crf=10)
    export = write_start_file(folder / "ana" / "extra" / "start.csv", {"A": (60.3, 60.5), "B": (200.4, 150.2)},
                              frame=0)
    return clip, export


def test_fine_gives_the_named_object_a_window_from_its_first_mask(tmp_path):
    clip, export = big_and_small(tmp_path)
    lines = []
    result = run(clip.path, export, lines, fine_ids=["B"], seconds=20 / FPS, step=2, fps=FPS)
    frames = list(range(0, 20, 2))
    tracks = {track["id"]: track for track in session_json(result.run_folder)["tracks"]}
    assert (tracks["A"]["mode"], tracks["B"]["mode"]) == ("coarse", "fine")
    # B's largest diameter is 40 px: W = ceil(3 x 40) = 120 px, give or take the pixels of its edge
    window = tracks["B"]["fine_window_px"]
    assert abs(window - 120) <= 4 and tracks["A"]["fine_window_px"] is None
    store = ResultsStore.load(result.run_folder / "results.npz")
    assert set(store.arrays("B").mode) == {"fine"} and set(store.arrays("A").mode) == {"coarse"}
    assert np.allclose(store.arrays("B").cell_px, window / 256)  # the model saw the window
    assert np.allclose(store.arrays("A").cell_px, 320 / 256)  # and the whole frame for A
    runs = [(one["tracks"], one["mode"]) for one in session_json(result.run_folder)["runs"]]
    assert runs == [(["A"], "coarse"), (["B"], "fine")]

    truth = clip.table.set_index(["track_id", "frame"])
    for name, file in zip(["A", "B"], result.files):
        got = read_track(file)
        assert list(got["frame"]) == frames
        want = truth.loc[name].loc[frames]
        assert np.allclose(got["pixelx"], want.u_px, atol=0.25) and np.allclose(got["pixely"], want.v_px, atol=0.25)
    positions = pd.read_csv(result.run_folder / "positions.csv")
    assert positions.groupby("track_id")["mode"].unique().map(list).to_dict() == {"A": ["coarse"], "B": ["fine"]}
    assert set(pd.read_csv(result.run_folder / "radial.csv").track_id) == {"B"}  # the shape files hold B
    with np.load(result.run_folder / "outlines.npz") as outlines:
        assert {"B__frames", "B__xy_mm", "B__xieta_mm"} <= set(outlines.files)
        assert not [key for key in outlines.files if key.startswith("A__")]
    assert result.flags == []


def test_an_unknown_fine_id_lists_the_ids_of_the_export(tmp_path):
    clip, export = two_disks(tmp_path)
    with pytest.raises(ValueError, match=r"--fine.*\bC\b.*start\.csv.*A, B"):
        run(clip, export, fine_ids=["B", "C"], seconds=8 / FPS, step=4, fps=FPS)
    assert not (tmp_path / "clip_tracker_outline_ana").exists()


# --------------------------------------------------------------------------- refusals before anything is tracked

def test_a_mirrored_calibration_is_refused_in_plain_words(tmp_path):
    video_path = tmp_path / "clip_tracker.mp4"
    disk_video(video_path, lambda f: [(60 + 0.5 * f, 60), (250 - 0.25 * f, 170), (100, 200)], n=50)
    marks = {"A": (80.5, 60.5), "B": (240.5, 170.5), "C": (100.5, 200.5)}  # three points not on one line
    export = write_start_file(tmp_path / "ana" / "extra" / "start.csv", marks, to_mm=mirrored_map)
    with pytest.raises(ValueError, match="mirrored") as refused:
        run(video_path, export, seconds=8 / FPS, step=4, fps=FPS)
    assert "start.csv" in str(refused.value) and "Traceback" not in str(refused.value)
    assert not (tmp_path / "clip_tracker_outline_ana").exists()


def test_an_export_that_starts_beyond_the_video_is_a_plain_error(tmp_path):
    clip, export = one_disk(tmp_path, frames=range(500, 521, 4), n=60)  # the video has frames 0 to 59
    with pytest.raises(ValueError, match=r"frame 500.*60 frames"):
        run(clip, export)
    assert not (tmp_path / "clip_tracker_outline_ana").exists()


def test_a_video_shorter_than_the_track_is_tracked_to_its_end(tmp_path):
    clip, export = one_disk(tmp_path, frames=range(40, 161, 4), n=100)  # the video has frames 0 to 99
    lines = []
    result = run(clip, export, lines)
    assert "  The video has 100 frames: tracking only up to frame 96." in lines  # last week's line
    assert list(read_track(result.files[0])["frame"]) == list(range(40, 97, 4))
    assert session_json(result.run_folder)["clip"] == {"start": 40, "end": 96, "step": 4}
    assert session_json(result.run_folder)["complete"] is True


def test_every_object_must_be_marked_on_the_same_first_frame(tmp_path):
    clip, _ = two_disks(tmp_path)
    export = tmp_path / "ana" / "extra" / "late.csv"
    rows = ["#multi:", ",A,,,,,B,,,,,", "t," + "frame,x,y,pixelx,pixely," * 2]
    a = [java_sci(float(v)) for v in (40, *tracker_map(80.5, 60.5), 80.5, 60.5)]
    b = [java_sci(float(v)) for v in (44, *tracker_map(239.5, 170.5), 239.5, 170.5)]  # B is first marked on frame 44
    rows.append(",".join([java_sci(0.0)] + a + [""] * 5) + ",")
    rows.append(",".join([java_sci(4 / FPS)] + [""] * 5 + b) + ",")
    export.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match=r"Every shrimp must be marked on the same first frame \(40\); B start later"):
        run(clip, export, fps=FPS)


def test_a_missing_video_or_export_is_a_plain_error(tmp_path):
    clip, export = one_disk(tmp_path)
    with pytest.raises(FileNotFoundError, match="No such video"):
        run(tmp_path / "nothing_tracker.mp4", export)
    with pytest.raises(FileNotFoundError, match="nothing.csv"):
        run(clip, tmp_path / "nothing.csv")


# --------------------------------------------------------------------------- stopping, and a model that fails

def test_ctrl_c_exports_what_was_tracked(tmp_path):
    clip, export = one_disk(tmp_path, frames=range(40, 161, 4), n=200)  # 31 frames to track
    lines = []
    result = run(clip, export, lines, segmenter=StopsAfter(20), overlay=True)  # Ctrl+C in the 21st frame
    tracked = list(range(40, 120, 4))  # the first 20 frames of the grid
    assert "  stopped at frame 20; saving what was tracked so far" in lines  # last week's line
    got = read_track(result.files[0])
    assert list(got["frame"]) == tracked
    ex, ey = tracker_map(60.5 + 0.5 * got["frame"], 100.5)
    assert np.allclose(got["x"], ex, atol=0.1 * MM_PER_PX) and np.allclose(got["y"], ey, atol=0.1 * MM_PER_PX)
    assert pd.read_csv(result.run_folder / "positions.csv").frame.tolist() == tracked
    assert names_in(result.run_folder) == sorted(RUN_FILES)  # every file, the overlay too
    assert video.probe(result.overlay).n_frames == 20
    session = session_json(result.run_folder)
    assert session["complete"] is False and session["runs"][0]["frames_done"] == 20
    assert session["clip"] == {"start": 40, "end": 160, "step": 4}  # what was asked for
    assert "results: partial" in (result.run_folder / "run.log").read_text(encoding="utf-8")
    assert result.flags == []


def test_a_model_that_fails_is_an_error_after_the_export(tmp_path):
    clip, export = one_disk(tmp_path)
    lines = []
    with pytest.raises(RuntimeError, match="clip_tracker_outline_ana"):
        run(clip, export, lines, segmenter=StopsAfter(3, error=RuntimeError))
    saved = tmp_path / "clip_tracker_outline_ana"
    assert list(read_track(saved / MODEL / "A.csv")["frame"]) == [40, 44, 48]  # the frames before the error
    assert session_json(saved)["complete"] is False
    assert any("stopped by the test" in line for line in lines)  # the model's own message reaches the console
