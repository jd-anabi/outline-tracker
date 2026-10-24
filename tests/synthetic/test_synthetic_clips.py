"""The synthetic clips encoded to video files (SPEC 13.2), and last week's selftest clip: the number of frames that
decodes, H.264 with B-frames, every frame different, the LED box, a disk clip through the encoder, folders with a
space and accents, what `render` refuses, and the selftest clip with its Tracker export. The ground truth, the
frames drawn in memory and the ready-made scenes are in tests/synthetic/test_synthetic.py.

Expected values come from the scenes' stated sizes, paths and colors, from the encoder's settings (H.264, yuv420p, a
keyframe at least every 24 frames) and from counting frames by decoding; the selftest clip's centers are worked out
from its recipe in the test. None is taken from the module's own output. Coordinates: px in Tracker's convention
(SPEC 3.1): u to the right, v downward, pixel (column c, row r) has its center at (c + 0.5, r + 0.5).
"""

import hashlib
import re
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from helpers import ODD_FOLDER
from synthetic_helpers import one_object_scene

from outline_tracker.synthetic import clips as synthetic
from outline_tracker.video import reader as video
from outline_tracker.measure.mask import mask_center
from outline_tracker.synthetic.shapes import Disk, Straight


def rgb_frames(path):
    """Yield the decoded frames of a clip as RGB arrays, one at a time, in decoding order."""
    for _, bgr in video.iter_frames(path, gray=False):
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def frame_types(path):
    """The picture type (I, P or B) of every frame in presentation order, and ffmpeg's stream line."""
    done = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path), "-vf", "showinfo",
                           "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert done.returncode == 0, done.stderr
    stream = next(line for line in done.stderr.splitlines() if "Video:" in line)
    return "".join(re.findall(r"type:([IPB])", done.stderr)), stream


# ---------------------------------------------------------------------------------------------
# Encoded clips (session fixtures of tests/conftest.py)

CLIPS = ["dish_clip", "closeup_clip", "disk_clip", "shapes_clip"]


@pytest.mark.parametrize("fixture", CLIPS)
def test_rendered_clip_decodes_with_exactly_n_frames(fixture, request):
    truth = request.getfixturevalue(fixture)
    scene = truth.scene
    info = video.probe(truth.path)
    assert (info.width, info.height) == scene.size
    assert info.n_frames == scene.n_frames
    assert info.fps_container == pytest.approx(240.0)
    shapes = [frame.shape for frame in rgb_frames(truth.path)]
    assert len(shapes) == scene.n_frames  # counted by decoding, not read from the header
    assert set(shapes) == {(scene.size[1], scene.size[0], 3)}


@pytest.mark.parametrize("fixture", CLIPS)
def test_rendered_clip_is_h264_yuv420p_gop_24_with_b_frames(fixture, request):
    truth = request.getfixturevalue(fixture)
    types, stream = frame_types(truth.path)
    assert "h264" in stream and "yuv420p" in stream
    assert len(types) == truth.scene.n_frames
    keyframes = [k for k, kind in enumerate(types) if kind == "I"]
    assert set(range(0, truth.scene.n_frames, 24)) <= set(keyframes)
    assert max(np.diff(keyframes)) <= 24
    assert "B" in types and "P" in types


@pytest.mark.parametrize("fixture", CLIPS)
def test_every_decoded_frame_differs_from_every_other(fixture, request):
    truth = request.getfixturevalue(fixture)
    digests = [hashlib.sha256(frame.tobytes()).digest() for frame in rgb_frames(truth.path)]
    assert all(a != b for a, b in zip(digests, digests[1:]))  # from its neighbors (A09 needs this)
    assert len(set(digests)) == truth.scene.n_frames


@pytest.mark.parametrize("fixture", ["dish_clip", "closeup_clip"])
def test_decoded_frame_k_shows_rendered_frame_k(fixture, request):
    # B-frames are stored out of order: every decoded frame must still be nearer to the drawn frame
    # of its own number than to the drawn frames before and after it. The levels themselves change a
    # little (measured: the conversion to yuv420p and back lowers every gray level by about 2).
    truth = request.getfixturevalue(fixture)
    scene = truth.scene
    drawn = [synthetic.render_frame(scene, frame).astype(np.int16) for frame in range(scene.n_frames)]
    for frame, decoded in enumerate(rgb_frames(truth.path)):
        error = {k: np.abs(decoded - drawn[k]).mean() for k in (frame - 1, frame, frame + 1) if 0 <= k < scene.n_frames}
        assert min(error, key=error.get) == frame
        assert error[frame] < 4.0


def test_led_box_mean_jumps_at_the_stated_frame_and_not_before(dish_clip):
    led = dish_clip.scene.led
    u0, v0, u1, v1 = led.box_px
    gray_of = lambda rgb: 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]  # noqa: E731
    off, on = gray_of(led.off_rgb), gray_of(led.on_rgb)
    assert on - off > 80
    means = np.array([frame[v0:v1, u0:u1].reshape(-1, 3).mean(axis=0) for frame in rgb_frames(dish_clip.path)])
    gray = means @ np.array([0.299, 0.587, 0.114])
    assert 0 < led.onset_frame < dish_clip.scene.n_frames - 1
    assert np.abs(gray[:led.onset_frame] - off).max() < 8
    assert np.abs(gray[led.onset_frame:] - on).max() < 8
    assert gray[led.onset_frame] - gray[led.onset_frame - 1] > 0.9 * (on - off)
    assert np.abs(means[:led.onset_frame] - led.off_rgb).max() < 10  # R, G and B each (channel order)
    assert np.abs(means[led.onset_frame:] - led.on_rgb).max() < 10


def test_disk_clip_through_h264_keeps_thresholded_centers_within_a_quarter_pixel(disk_clip):
    # The condition measured for `ThresholdFake` (SPEC 13.2): dark disks of radius 12 px at crf 10.
    scene = disk_clip.scene
    assert all(isinstance(obj.shape, Disk) and obj.shape.radius_px == 12.0 for obj in scene.objects)
    worst = 0.0
    for frame, rgb in enumerate(rgb_frames(disk_clip.path)):
        dark = rgb[:, :, 0] < 128
        for obj in scene.objects:
            u, v, _ = obj.path.pose(frame)
            near = np.zeros_like(dark)
            near[int(v) - 20:int(v) + 21, int(u) - 20:int(u) + 21] = True
            got_u, got_v, area = mask_center(dark & near)
            assert area > 300  # pi 12^2 = 452 px
            worst = max(worst, float(np.hypot(got_u - u, got_v - v)))
    assert worst < 0.25


def test_render_writes_into_a_folder_with_a_space_and_non_ascii_characters(tmp_path):
    folder = tmp_path / ODD_FOLDER
    folder.mkdir()
    scene = one_object_scene(Disk(8.0), Straight((30.0, 30.0), (1.0, 0.5)), size=(96, 64), n_frames=12)
    truth = synthetic.render(scene, folder / "clip_tracker.mp4")
    assert truth.path == folder / "clip_tracker.mp4" and truth.scene == scene
    assert video.probe(truth.path).n_frames == 12
    assert sum(1 for _ in rgb_frames(truth.path)) == 12


def test_clip_in_odd_folder_opens(clip_in_odd_folder, dish_clip, tmp_path):
    path = clip_in_odd_folder.path
    assert path.parent == tmp_path / "vidéo test ü" and path != dish_clip.path
    assert clip_in_odd_folder.scene == dish_clip.scene
    assert video.probe(path).n_frames == 120
    assert sum(1 for _ in rgb_frames(path)) == 120
    assert clip_in_odd_folder.table.equals(dish_clip.table)


def test_render_refuses_what_it_cannot_encode(tmp_path):
    disk, path = Disk(8.0), Straight((30.0, 30.0), (1.0, 0.5))
    with pytest.raises(ValueError, match="even"):  # yuv420p needs an even width and height
        synthetic.render(one_object_scene(disk, path, size=(97, 64)), tmp_path / "odd.mp4")
    with pytest.raises(FileNotFoundError, match="missing"):
        synthetic.render(one_object_scene(disk, path, size=(96, 64)), tmp_path / "missing" / "clip.mp4")
    for skip_before in [(0,), (2, 4)]:  # a gap before frame 0 is no gap; the scene has frames 0 to 3
        with pytest.raises(ValueError, match="skip_before"):
            synthetic.render(one_object_scene(disk, path, size=(96, 64)), tmp_path / "gap.mp4", skip_before=skip_before)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------------------------
# Last week's selftest clip and its Tracker export


def test_the_selftest_clip_is_40_frames_of_1080p_and_an_export_of_the_true_centers_on_every_second_frame(tmp_path):
    # The recipe of `selftest_clip`: 40 frames of 1920 x 1080 px at 240 frames per s, 0.0324 mm per px.
    # The ellipse starts at the pixel in column 700 and row 500 and moves (0.6, 0.2) px per frame, so in
    # Tracker's convention (+0.5) its center is at (700.5 + 0.6 f, 500.5 + 0.2 f) px in frame f. The
    # export has every 2nd frame, with x and y in mm from the frame's center, (960, 540) px, y up.
    made = synthetic.selftest_clip(tmp_path / "clip")
    assert made["video"] == tmp_path / "clip" / "selftest_tracker.mp4" and made["video"].is_file()
    assert made["export"] == tmp_path / "clip" / "selftest.csv" and made["export"].is_file()
    frames = np.arange(0, 40, 2)
    true_u, true_v = 700.5 + 0.6 * frames, 500.5 + 0.2 * frames
    assert made["frames"] == list(range(0, 40, 2))
    assert np.allclose(made["pixelx"], true_u) and np.allclose(made["pixely"], true_v)

    clip = cv2.VideoCapture(str(made["video"]))  # counted by decoding, with OpenCV alone
    try:
        for frame in range(40):
            ok, image = clip.read()
            assert ok, f"frame {frame} is missing"
            assert image.shape == (1080, 1920, 3)
        assert not clip.read()[0]  # 40 frames, no more
    finally:
        clip.release()

    # The export, read here as text: the name line, the column names, one row per marked frame. A value
    # in the file is the true one to half of its last decimal: 7 decimals for t, 6 for mm, 3 for px.
    lines = made["export"].read_text(encoding="ascii").splitlines()
    assert lines[:2] == [",selftest,,,,,", "t,frame,x,y,pixelx,pixely"]
    t, marked, x, y, pixelx, pixely = np.loadtxt(lines[2:], delimiter=",", unpack=True)
    assert marked.tolist() == list(range(0, 40, 2))
    assert pixelx == pytest.approx(true_u, rel=0, abs=0.5e-3) and pixely == pytest.approx(true_v, rel=0, abs=0.5e-3)
    assert t == pytest.approx(frames / 240.0, rel=0, abs=0.5e-7)
    assert x == pytest.approx(0.0324 * (true_u - 960.0), rel=0, abs=0.5e-6)
    assert y == pytest.approx(-0.0324 * (true_v - 540.0), rel=0, abs=0.5e-6)
