"""The overlay video (SPEC 8.8 P0 content, 13.2; review focus 1 and 2): `draw_overlay_frame`, the
pure drawing, and `write_overlay`, which decodes the clip again and draws what results.npz holds.

Expected values: where an outline, a dot or an id must be comes from the records the test made of
the synthetic ground truth (tests/overlay_helpers.py), scaled to the overlay; a frame without
tracks must be last week's overlay frame (`shrimp.segment._overlay`, the reference copy), which
fixes the size, the resizing and the stamp, and it must follow the rule itself: the frame resized,
and nothing but the stamp drawn on it (`stamp_box`). Colors are asserted on drawn arrays only, never
on a decoded video: compression changes them. Positions are px, pixel centers at +0.5; colors are RGB.
"""

import errno
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest
from helpers import SMALL
from overlay_helpers import (FPS_TRUE, RGB, Drawn, decoded_frames, ffmpeg_report, fraction_marked,
                             lost_record, make_run, marks, near, record, scaled)
from results_helpers import names, names_number
from shrimp import segment as reference

from outline_tracker import fileio, synthetic, video
from outline_tracker.overlay import OverlayItem, draw_overlay_frame, write_overlay
from outline_tracker.results import ResultsStore
from outline_tracker.session import Session
from outline_tracker.synthetic import GroundTruth

# The scenes of the session's clips (tests/helpers.py), for the tests that need no video file.
DISH = GroundTruth(synthetic.dish_scene(size=SMALL, n_frames=120))  # 320 x 240 px: the overlay is 3 times larger
SHAPES = GroundTruth(synthetic.shapes_scene())                      # 640 x 480 px: 1.5 times
RUN_FILES = ["results.npz", "session.json"]  # what `make_run` writes


def item(truth, track_id, frame, name=None):
    """(the item to draw for the track on that frame, the record it was made of)."""
    row = record(truth, track_id, frame)
    return OverlayItem(name or track_id, RGB[track_id], row.outline_px, (row.u, row.v)), row


@pytest.fixture
def drawn(monkeypatch):
    """What `write_overlay` draws, frame by frame (see `overlay_helpers.Drawn`)."""
    return Drawn(monkeypatch)


# ---------------------------------------------------------------------------------------------
# The drawing


@pytest.mark.parametrize("truth, frame, width, shape", [
    (SHAPES, 7, 960, (720, 960, 3)), (SHAPES, 30, 640, (480, 640, 3)),
    (DISH, 20, 960, (720, 960, 3)), (DISH, 100, 480, (360, 480, 3)),
], ids=["shapes", "shapes-640", "dish", "dish-480"])
def test_each_track_is_drawn_as_its_outline_and_a_dot_in_its_color(truth, frame, width, shape):
    rgb = synthetic.render_frame(truth.scene, frame)
    made = {obj.track_id: item(truth, obj.track_id, frame) for obj in truth.scene.objects}
    image = draw_overlay_frame(rgb, [one for one, _ in made.values()], frame, frame / FPS_TRUE, width)
    undrawn = draw_overlay_frame(rgb, [], frame, frame / FPS_TRUE, width)
    assert image.dtype == np.uint8 and image.shape == undrawn.shape == shape
    for track_id, (_, row) in made.items():
        outline = scaled(row.outline_px, truth.scene.size, image)
        assert fraction_marked(outline, marks(image, undrawn, RGB[track_id])) >= 0.9, track_id
        # A filled dot of radius 2 px covers the whole pixel that holds the centroid.
        (x, y), = scaled((row.u, row.v), truth.scene.size, image)
        assert tuple(image[int(y), int(x)]) == RGB[track_id], track_id


def test_the_id_is_written_next_to_the_dot_in_the_tracks_color():
    frame = 7
    rgb = synthetic.render_frame(SHAPES.scene, frame)
    undrawn = draw_overlay_frame(rgb, [], frame, 0.0)
    named_b, row = item(SHAPES, "B", frame)
    named_w, _ = item(SHAPES, "B", frame, name="W")
    as_b, as_w = draw_overlay_frame(rgb, [named_b], frame, 0.0), draw_overlay_frame(rgb, [named_w], frame, 0.0)
    rows, cols = np.nonzero(np.any(as_b != as_w, axis=2))  # the two frames differ in the letter alone
    assert len(rows) >= 10
    (x, y), = scaled((row.u, row.v), SHAPES.scene.size, as_b)
    assert np.hypot(cols + 0.5 - x, rows + 0.5 - y).max() <= 30.0  # overlay px from the centroid
    lettered = marks(as_b, undrawn, RGB["B"]) | marks(as_w, undrawn, RGB["B"])
    assert lettered[rows, cols].all()


def stamp_box(frame: int, t_s: float) -> tuple[int, int, int, int]:
    """The rectangle of an overlay frame that holds its stamp: (left, top, right, bottom) in whole px
    of the overlay, counted from its top-left corner, right and bottom not included.

    The stamp is the text `t = 1.234 s   frame 296`, written with `cv2.putText` from (10, 22), the
    left end of the line the letters stand on, in OpenCV's Hershey Simplex font at scale 0.6 with a
    line 2 px thick. `cv2.getTextSize` gives the box of such a text: how far it reaches to the right
    of that point, how far above it, and how far below. 3 px are added on every side for the thick
    line and its soft edge, and the rectangle ends at the frame's top."""
    (width, height), below = cv2.getTextSize(f"t = {t_s:.3f} s   frame {frame}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    return 10 - 3, max(22 - height - 3, 0), 10 + width + 3, 22 + below + 3


@pytest.mark.parametrize("size, shape, frame, t_s", [
    ((320, 240), (720, 960, 3), 0, 0.0),
    ((1920, 1080), (540, 960, 3), 1438, 6.0016694),
    ((400, 262), (628, 960, 3), 77, 0.3213689),  # 960 / 400 * 262 = 628.8: the nearest even height is 628
])
def test_a_frame_without_tracks_is_last_weeks_overlay_frame(size, shape, frame, t_s):
    # Last week's frame: resized to 960 px wide with an even height, stamped "t = ... s   frame ...".
    rgb = np.random.default_rng(3).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    image = draw_overlay_frame(rgb, [], frame, t_s)
    assert image.dtype == np.uint8 and image.shape == shape
    # The rule, without last week's script: the frame resized to the overlay's size with OpenCV's
    # bilinear interpolation, and nothing drawn on it but the stamp. Outside the stamp's rectangle
    # the two are the same picture; inside it the stamp has changed pixels.
    resized = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    left, top, right, bottom = stamp_box(frame, t_s)
    stamp = np.zeros(shape[:2], bool)
    stamp[top:bottom, left:right] = True
    assert stamp.sum() < 0.02 * stamp.size  # a small corner of the frame: nearly all of it is compared
    assert np.array_equal(image[~stamp], resized[~stamp])
    assert (image[stamp] != resized[stamp]).any()
    assert np.array_equal(image, reference._overlay(rgb, [], [], [], frame, t_s))


def test_the_stamp_is_white_text_in_the_top_left_corner_and_changes_with_frame_and_time():
    black = np.zeros((240, 320, 3), np.uint8)
    image = draw_overlay_frame(black, [], 12, 0.05)
    rows, cols = np.nonzero(image.any(axis=2))
    assert len(rows) > 100 and rows.max() < 40 and cols.max() < 400  # of 720 x 960 px
    assert (image[rows, cols] == 255).all(axis=1).any() and (np.ptp(image, axis=2) == 0).all()  # white, gray edges
    assert not np.array_equal(image, draw_overlay_frame(black, [], 13, 0.05))
    assert not np.array_equal(image, draw_overlay_frame(black, [], 12, 0.051))


def test_a_lost_track_draws_nothing():
    frame = 20
    rgb = synthetic.render_frame(DISH.scene, frame)
    lost = lost_record(DISH, frame)
    assert not lost.visible and np.isnan(lost.outline_px).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # NaN must not reach a conversion to whole pixels
        image = draw_overlay_frame(rgb, [OverlayItem("A", RGB["A"], lost.outline_px, (lost.u, lost.v))], frame, 0.1)
    assert np.array_equal(image, draw_overlay_frame(rgb, [], frame, 0.1))


def test_drawing_returns_a_new_array_and_leaves_the_frame_as_it_was():
    frame = 30
    rgb = synthetic.render_frame(SHAPES.scene, frame)
    before = rgb.copy()
    one, _ = item(SHAPES, "A", frame)
    image = draw_overlay_frame(rgb, [one], frame, 0.125, width=640)  # the frame's own width: no resizing
    assert np.array_equal(rgb, before) and not np.shares_memory(image, rgb)
    assert not np.array_equal(image, rgb)


@pytest.mark.parametrize("rgb, width, word", [
    (np.zeros((240, 320, 3), np.float32), 960, "uint8"),
    (np.zeros((240, 320), np.uint8), 960, "RGB"),
    (np.zeros((240, 320, 3), np.uint8), 961, "even"),  # yuv420p needs an even width
    (np.zeros((240, 320, 3), np.uint8), 0, "even"),
])
def test_drawing_refuses_an_image_or_a_width_it_cannot_use(rgb, width, word):
    with pytest.raises(ValueError, match=word):
        draw_overlay_frame(rgb, [], 0, 0.0, width)


# ---------------------------------------------------------------------------------------------
# The file


def test_overlay_has_one_frame_per_tracked_frame_also_in_an_odd_folder(clip_in_odd_folder, drawn):
    truth = clip_in_odd_folder  # the dish clip in a folder with a space and non-ASCII letters
    run = truth.path.parent / "dish_tracker_outline_équipe 2"
    frames = list(range(10, 50, 2))
    # A is lost on frame 24; C starts at frame 30, as a new piece does.
    make_run(run, {"A": [lost_record(truth, f) if f == 24 else record(truth, "A", f) for f in frames],
                   "C": [record(truth, "C", f) for f in frames if f >= 30]})
    messages = []
    path, n_frames = write_overlay(run, truth.path, log=messages.append)
    assert path == run / "overlay.mp4" and n_frames == 20 and messages == []
    assert names(run) == ["overlay.mp4", *RUN_FILES]

    count, stream = ffmpeg_report(path)
    assert count == 20
    assert "h264" in stream and "yuv420p" in stream and "960x720" in stream and "30 fps" in stream
    assert [frame.shape for frame in decoded_frames(path)] == [(720, 960, 3)] * 20

    # Every tracked frame was drawn once, in order, on that frame of the clip, with t = frame / fps_true.
    assert [frame for frame, _ in drawn.calls] == frames
    assert [t_s for _, t_s in drawn.calls] == pytest.approx([frame / FPS_TRUE for frame in frames], abs=1e-12)
    clip = decoded_frames(truth.path)
    assert all(np.array_equal(drawn.frames[frame], clip[frame]) for frame in frames)

    def outline(track_id, frame):
        return scaled(record(truth, track_id, frame).outline_px, truth.scene.size, drawn.images[frame][0])

    for frame in (12, 24, 40):
        image, undrawn = drawn.images[frame]
        for track_id, shown in (("A", frame != 24), ("C", frame >= 30)):
            if shown:
                assert fraction_marked(outline(track_id, frame), marks(image, undrawn, RGB[track_id])) >= 0.9
            else:  # lost, or not tracked on this frame: nothing of the track is on the frame
                around = near(outline(track_id, frame), image.shape[:2], 3.0)
                assert around.sum() > 100 and np.array_equal(image[around], undrawn[around])


def test_overlay_shows_a_correction_made_in_the_results_store(tmp_path, dish_clip, drawn):
    truth, run, k = dish_clip, tmp_path / "run", 20
    frames = list(range(10, 30, 2))
    make_run(run, {"A": [record(truth, "A", f) for f in frames], "C": [record(truth, "C", f) for f in frames]})
    write_overlay(run, truth.path)
    before = dict(drawn.images)

    # What "Re-track from here" does to the store: from frame k on, track A holds another object (B).
    store = ResultsStore.load(run / "results.npz")
    store.replace_from("A", k)
    for frame in (f for f in frames if f >= k):
        store.put("A", record(truth, "B", frame))
    store.save(run / "results.npz")
    path, n_frames = write_overlay(run, truth.path)
    assert path == run / "overlay.mp4" and n_frames == len(frames) == ffmpeg_report(path)[0]

    for frame in frames:
        image, undrawn = drawn.images[frame]
        old = scaled(record(truth, "A", frame).outline_px, truth.scene.size, image)
        new = scaled(record(truth, "B", frame).outline_px, truth.scene.size, image)
        assert np.hypot(*(old[:, None, :] - new[None, :, :]).transpose(2, 0, 1)).min() >= 60.0  # overlay px apart
        assert fraction_marked(old, marks(*before[frame], RGB["A"])) >= 0.9  # it was there before the correction
        if frame < k:
            assert np.array_equal(image, before[frame][0])
        else:
            assert fraction_marked(new, marks(image, undrawn, RGB["A"])) >= 0.9
            around = near(old, image.shape[:2], 3.0)
            assert around.sum() > 100 and np.array_equal(image[around], undrawn[around])


def test_overlay_goes_where_out_path_says_and_only_to_an_mp4(tmp_path, dish_clip):
    run = tmp_path / "run"
    make_run(run, {"A": [record(dish_clip, "A", f) for f in range(0, 8, 2)]})
    elsewhere = tmp_path / "slides" / "shown.mp4"  # the folder does not exist yet
    assert write_overlay(run, dish_clip.path, out_path=elsewhere, width=480) == (elsewhere, 4)
    count, stream = ffmpeg_report(elsewhere)
    assert count == 4 and "h264" in stream and "yuv420p" in stream and "480x360" in stream
    with pytest.raises(ValueError, match="mp4"):
        write_overlay(run, dish_clip.path, out_path=tmp_path / "shown.avi")
    assert names(run) == RUN_FILES and names(tmp_path / "slides") == ["shown.mp4"]


def test_the_video_is_finished_under_a_temporary_mp4_name_and_then_renamed(tmp_path, dish_clip, monkeypatch):
    run = tmp_path / "run"
    make_run(run, {"A": [record(dish_clip, "A", f) for f in range(0, 12, 2)]})
    renames, real_replace = [], fileio.os.replace

    def replace(src, dst):
        renames.append((Path(src), Path(dst), ffmpeg_report(src)[0]))  # the frames it holds before it has its name
        real_replace(src, dst)

    monkeypatch.setattr(fileio.os, "replace", replace)
    path, n_frames = write_overlay(run, dish_clip.path)
    (temporary, target, count), = renames
    assert target == path == run / "overlay.mp4" and count == n_frames == 6
    assert temporary.parent == run and temporary.name.endswith(".mp4") and temporary != target
    assert names(run) == ["overlay.mp4", *RUN_FILES]  # no temporary file is left


def test_a_locked_overlay_is_left_alone_and_the_new_one_is_written_next_to_it(tmp_path, dish_clip, monkeypatch):
    run = tmp_path / "run"
    make_run(run, {"A": [record(dish_clip, "A", f) for f in range(0, 12, 2)]})
    target = run / "overlay.mp4"
    target.write_bytes(b"the overlay of the last export, open in a player")
    real_replace = fileio.os.replace

    def replace(src, dst):  # what Windows does while another program holds overlay.mp4 open
        if Path(dst) == target:
            raise PermissionError(errno.EACCES, "The file is being used by another process", str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(fileio.os, "replace", replace)
    monkeypatch.setattr(fileio.time, "sleep", lambda seconds: None)
    path, n_frames = write_overlay(run, dish_clip.path)
    assert path == run / "overlay.new.mp4" and n_frames == 6 == ffmpeg_report(path)[0]
    assert target.read_bytes() == b"the overlay of the last export, open in a player"
    assert names(run) == ["overlay.mp4", "overlay.new.mp4", *RUN_FILES]


# ---------------------------------------------------------------------------------------------
# A clip that ends early (review focus 2), and what cannot be drawn at all


def test_a_clip_that_ends_before_the_last_tracked_frame_gives_an_overlay_up_to_its_end(tmp_path, dish_clip, drawn):
    run = tmp_path / "run"
    frames = list(range(100, 140, 2))  # the clip has frames 0 to 119: ten of these twenty exist
    make_run(run, {"A": [record(dish_clip, "A", min(f, 119), stored_as=f) for f in frames]})
    messages = []
    path, n_frames = write_overlay(run, dish_clip.path, log=messages.append)
    assert path == run / "overlay.mp4" and n_frames == 10
    assert ffmpeg_report(path)[0] == 10 and len(decoded_frames(path)) == 10
    assert [frame for frame, _ in drawn.calls] == frames[:10]
    (message,) = messages  # it says so: the last frame drawn, and how many of the tracked frames
    assert names_number(message, 118) and names_number(message, 10) and names_number(message, 20)
    write_overlay(run, dish_clip.path)  # without a callback it is silent, and works the same


def test_a_file_cut_short_gives_an_overlay_of_the_frames_that_still_decode(tmp_path, dish_clip, drawn):
    # A file that reports more frames than it holds: the index is whole, the frame data are cut.
    data = dish_clip.path.read_bytes()
    cut = tmp_path / "cut_tracker.mp4"
    cut.write_bytes(data[:int(0.6 * len(data))])
    decodable = len(decoded_frames(cut))
    assert video.probe(cut).n_frames == 120 and 10 < decodable < 110  # it reports 120 frames and holds fewer
    run = tmp_path / "run"
    frames = list(range(0, 120, 6))
    make_run(run, {"A": [record(dish_clip, "A", f) for f in frames]})
    expected = [f for f in frames if f < decodable]
    messages = []
    path, n_frames = write_overlay(run, cut, log=messages.append)
    assert n_frames == len(expected) == ffmpeg_report(path)[0]
    assert [frame for frame, _ in drawn.calls] == expected
    assert len(messages) == 1 and names_number(messages[0], expected[-1])


@pytest.mark.parametrize("case, error", [("notes", OSError), ("missing", FileNotFoundError), ("too short", OSError)])
def test_a_video_that_gives_no_tracked_frame_raises_and_leaves_the_old_overlay_whole(tmp_path, dish_clip, case, error):
    run = tmp_path / "run"
    first = 200 if case == "too short" else 0  # the clip ends at frame 119
    make_run(run, {"A": [record(dish_clip, "A", 0, stored_as=first + 2 * i) for i in range(3)]})
    (run / "overlay.mp4").write_bytes(b"the overlay of the last export")
    notes = tmp_path / "notes.mp4"
    notes.write_text("this is text, not a video\n")
    path = {"notes": notes, "missing": tmp_path / "missing.mp4", "too short": dish_clip.path}[case]
    with pytest.raises(error, match=path.name):
        write_overlay(run, path)
    assert (run / "overlay.mp4").read_bytes() == b"the overlay of the last export"
    assert names(run) == ["overlay.mp4", *RUN_FILES]


@pytest.mark.parametrize("change, word", [
    (lambda session: setattr(session.time, "fps_true", None), "fps_true"),
    (lambda session: setattr(session.tracks[0], "color", "yellow"), "yellow"),
    (lambda session: setattr(session.tracks[0], "color", "#FFFF0"), "#FFFF0"),
    (lambda session: session.tracks.clear(), "track"),  # no track of the session has results
], ids=["no fps_true", "a color name", "a short color", "no tracks"])
def test_a_session_that_cannot_be_drawn_is_refused_before_anything_is_written(tmp_path, dish_clip, change, word):
    run = tmp_path / "run"
    make_run(run, {"A": [record(dish_clip, "A", f) for f in range(0, 8, 2)]})
    session = Session.load(run / "session.json")
    change(session)
    session.save(run / "session.json")
    with pytest.raises(ValueError, match=word):
        write_overlay(run, dish_clip.path)
    assert names(run) == RUN_FILES


def test_the_colors_come_from_the_session_as_rgb_hex_in_either_case(tmp_path, dish_clip, drawn):
    run = tmp_path / "run"
    colors = {"A": "#0080ff", "B": "#FF00FF", "C": "#FFFF00"}  # lower case too; A and C in each other's color
    shown = {"A": (0, 128, 255), "B": (255, 0, 255), "C": (255, 255, 0)}  # red, green, blue
    make_run(run, {track_id: [record(dish_clip, track_id, 20)] for track_id in "ABC"}, colors=colors)
    assert write_overlay(run, dish_clip.path) == (run / "overlay.mp4", 1)
    image, undrawn = drawn.images[20]
    for track_id, color in shown.items():
        row = record(dish_clip, track_id, 20)
        outline = scaled(row.outline_px, dish_clip.scene.size, image)
        assert fraction_marked(outline, marks(image, undrawn, color)) >= 0.9, track_id
        (x, y), = scaled((row.u, row.v), dish_clip.scene.size, image)
        assert tuple(image[int(y), int(x)]) == color, track_id
