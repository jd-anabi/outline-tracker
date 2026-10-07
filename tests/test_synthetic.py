"""Synthetic clips with ground truth (SPEC 13.2; decision X16: shapes are implicit functions).

Expected values come from geometry written out here (exact inequalities, analytic outlines, the
scenes' stated sizes), never from the module's own output. Coordinates: px in Tracker's convention
(SPEC 3.1): u to the right, v downward, pixel (column c, row r) has its center at (c + 0.5, r + 0.5).
Angles are in rad, counterclockwise on screen from the image's rightward direction (y up). The body
frame of a shape has xi toward the head and eta 90 degrees counterclockwise from it.
"""

import hashlib
import re
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from helpers import ODD_FOLDER, SMALL
from scipy.spatial import ConvexHull

from outline_tracker import synthetic, video
from outline_tracker.measure import mask_center
from outline_tracker.synthetic import GroundTruth, Led, Scene, SceneObject
from outline_tracker.synthetic_shapes import Arc, Disk, Ellipse, Shrimp, Straight

CLOSEUP_SHRIMP = Shrimp(a_px=23.5, b_px=10.0, antenna_length_px=30.0, antenna_width_px=3.0, attach_px=14.0)


def pixel_centers(width, height):
    """(u, v) of every pixel center of a frame, as two arrays indexed [row, column]."""
    rows, cols = np.mgrid[0:height, 0:width]
    return cols + 0.5, rows + 0.5


def one_object_scene(shape, path, size=(120, 90), n_frames=4, **more):
    return Scene(size=size, fps=240.0, n_frames=n_frames, objects=(SceneObject("A", shape, path),),
                 mm_per_px=0.01, **more)


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
# Ground truth: the mask is d > 0 at pixel centers, the logits are d


def test_disk_mask_is_the_set_of_pixel_centers_inside_the_circle():
    truth = GroundTruth(one_object_scene(Disk(12.0), Straight((50.3, 40.7), (0.5, 0.13))))
    u, v = pixel_centers(120, 90)
    for frame in (0, 3):
        center = (50.3 + 0.5 * frame, 40.7 + 0.13 * frame)
        radius = np.hypot(u - center[0], v - center[1])
        mask, logits = truth.mask("A", frame), truth.logits("A", frame)
        assert mask.dtype == bool and mask.shape == (90, 120)
        assert np.array_equal(mask, radius < 12.0)
        assert logits.dtype == np.float32 and logits.shape == (90, 120)
        assert logits == pytest.approx(12.0 - radius, abs=1e-4)  # the signed distance, in the whole frame
        assert np.array_equal(logits > 0, mask)


def test_ellipse_mask_turns_counterclockwise_on_screen():
    theta = np.radians(30.0)
    truth = GroundTruth(one_object_scene(Ellipse(40.0, 15.0), Straight((60.3, 45.6), (0.0, 0.0), heading_rad=theta)))
    u, v = pixel_centers(120, 90)
    x, y = u - 60.3, -(v - 45.6)  # y up
    along, across = x * np.cos(theta) + y * np.sin(theta), -x * np.sin(theta) + y * np.cos(theta)
    mask = truth.mask("A", 0)
    assert np.array_equal(mask, (along / 40.0) ** 2 + (across / 15.0) ** 2 < 1)
    assert mask[45 - 17, 60 + 30] and not mask[45 + 17, 60 + 30]  # the head end is up and to the right


def test_mask_and_logits_reject_unknown_tracks_and_frames():
    truth = GroundTruth(one_object_scene(Disk(12.0), Straight((50.3, 40.7), (0.5, 0.13))))
    with pytest.raises(KeyError, match="Z"):
        truth.mask("Z", 0)
    for frame in (-1, 4):
        with pytest.raises(IndexError, match="frame"):
            truth.logits("A", frame)


def test_an_object_outside_the_frame_has_an_empty_mask_and_a_nan_row():
    truth = GroundTruth(one_object_scene(Disk(12.0), Straight((100.0, 40.0), (20.0, 0.0))))
    assert truth.mask("A", 0).any() and not truth.mask("A", 3).any()  # at u = 160 of a 120 px frame
    row = truth.table.iloc[3]
    assert row["area_px"] == 0 and np.isnan(row["u_px"]) and np.isnan(row["v_px"])


SCENES = {
    "dish": lambda: synthetic.dish_scene(size=SMALL, n_frames=120),
    "closeup": lambda: synthetic.closeup_scene(size=SMALL, n_frames=120),
    "disks": lambda: synthetic.disk_scene(),
    "shapes": lambda: synthetic.shapes_scene(),
}


@pytest.mark.parametrize("name", SCENES)
def test_table_centroid_is_mask_center_of_the_ground_truth_mask(name):
    scene = SCENES[name]()
    truth = GroundTruth(scene)
    table = truth.table
    assert list(table.columns) == ["track_id", "frame", "u_px", "v_px", "area_px", "heading_rad", "solidity"]
    ids = [obj.track_id for obj in scene.objects]
    assert list(table["track_id"]) == [i for i in ids for _ in range(scene.n_frames)]  # sorted by track, frame
    assert list(table["frame"]) == list(range(scene.n_frames)) * len(ids)
    for obj in scene.objects:
        rows = table[table["track_id"] == obj.track_id].set_index("frame")
        for frame in (0, 1, scene.n_frames // 2, scene.n_frames - 1):
            mask = truth.mask(obj.track_id, frame)
            u, v, area = mask_center(mask)
            assert area > 0
            assert (rows.loc[frame, "u_px"], rows.loc[frame, "v_px"]) == pytest.approx((u, v), abs=1e-9)
            assert rows.loc[frame, "area_px"] == area
            assert rows.loc[frame, "heading_rad"] == pytest.approx(obj.path.pose(frame)[2])
            assert np.array_equal(truth.logits(obj.track_id, frame) > 0, mask)  # nothing lies outside the window


# ---------------------------------------------------------------------------------------------
# True solidity: polygon area / hull area of the analytic outline


def exact_shrimp_solidity(shrimp, beta):
    """Area of the body-plus-antennae union / area of its convex hull, from exact geometry.

    The area counts the points of a 0.1 px grid that satisfy the ellipse inequality or lie within
    half a width of an antenna's center line. The hull is the hull of the ellipse outline and of
    the four end circles of the two antennae (a thick segment is the hull of its two end disks).
    """
    a, b, length, width, attach = (shrimp.a_px, shrimp.b_px, shrimp.antenna_length_px, shrimp.antenna_width_px,
                                   shrimp.attach_px)
    step = 0.1
    axis = np.arange(-(attach + length + width), attach + length + width, step) + step / 2
    xi, eta = np.meshgrid(axis, axis)
    inside = (xi / a) ** 2 + (eta / b) ** 2 < 1
    angle = np.linspace(0.0, 2 * np.pi, 721)
    outline = [np.column_stack([a * np.cos(angle), b * np.sin(angle)])]
    for side in (1.0, -1.0):
        along = np.array([np.cos(beta), side * np.sin(beta)])
        px, py = xi - attach, eta
        s = np.clip(px * along[0] + py * along[1], 0.0, length)
        inside |= np.hypot(px - s * along[0], py - s * along[1]) < width / 2
        for end in (0.0, length):
            center = np.array([attach, 0.0]) + end * along
            outline.append(center + width / 2 * np.column_stack([np.cos(angle), np.sin(angle)]))
    return inside.sum() * step ** 2 / ConvexHull(np.vstack(outline)).volume


@pytest.mark.parametrize("quarter_beats", [0, 1, 3])  # beta = 45, 75 and 15 degrees
def test_true_solidity_of_the_shrimp_matches_exact_geometry(quarter_beats):
    t_s = quarter_beats / (4 * 9.0)
    expected = exact_shrimp_solidity(CLOSEUP_SHRIMP, CLOSEUP_SHRIMP.beta_rad(t_s))
    assert 0.4 < expected < 0.8
    assert synthetic.true_solidity(CLOSEUP_SHRIMP, t_s) == pytest.approx(expected, abs=0.003)


def test_true_solidity_of_convex_shapes_is_one():
    assert synthetic.true_solidity(Disk(50.0)) == pytest.approx(1.0, abs=1e-3)
    assert synthetic.true_solidity(Ellipse(40.0, 15.0)) == pytest.approx(1.0, abs=1e-3)
    assert synthetic.true_solidity(Ellipse(7.25, 3.09)) == pytest.approx(1.0, abs=2e-3)  # dish scale


def test_table_solidity_is_the_true_solidity_of_each_frames_shape():
    scene = synthetic.closeup_scene(size=SMALL, n_frames=61)
    table = GroundTruth(scene).table.set_index(["track_id", "frame"])
    shrimp = scene.objects[0].shape
    for frame, beta_deg in [(0, 45.0), (20, 15.0), (60, 75.0)]:  # 240 fps and 9 Hz: sin(2 pi 9 n / 240) = 0, -1, 1
        expected = exact_shrimp_solidity(shrimp, np.radians(beta_deg))
        assert table.loc[("A", frame), "solidity"] == pytest.approx(expected, abs=0.003)
    assert (table.loc["B", "solidity"] > 0.998).all()  # the plain body is convex


def test_true_solidity_of_the_closeup_shrimp_beats_at_9_hz_not_18():
    scene = synthetic.closeup_scene(size=SMALL, n_frames=480)  # 2 s at 240 fps: 0.5 Hz per bin
    assert scene.fps == 240.0
    table = GroundTruth(scene).table
    solidity = table.loc[table["track_id"] == "A", "solidity"].to_numpy()
    assert len(solidity) == 480 and np.isfinite(solidity).all()
    amplitude = np.abs(np.fft.rfft(solidity - solidity.mean()))
    frequency = np.fft.rfftfreq(480, 1 / 240.0)
    assert abs(frequency[np.argmax(amplitude)] - 9.0) <= 0.5
    assert amplitude[frequency == 18.0][0] < amplitude[frequency == 9.0][0]
    assert np.ptp(solidity) > 0.1  # a signal, not rounding noise


# ---------------------------------------------------------------------------------------------
# Frames: rendered from the same function, coverage = clip(d + 1/2, 0, 1)


def test_frame_is_rendered_with_coverage_from_the_signed_distance():
    # A disk of radius 10 centered on the center of pixel (column 50, row 40).
    scene = Scene(size=(120, 90), fps=240.0, n_frames=2, mm_per_px=0.01, background=220,
                  objects=(SceneObject("A", Disk(10.0), Straight((50.5, 40.5), (0.0, 0.0)), gray=40),))
    frame = synthetic.render_frame(scene, 0)
    assert frame.dtype == np.uint8 and frame.shape == (90, 120, 3)
    assert tuple(frame[5, 5]) == (220, 220, 220)   # background
    assert tuple(frame[40, 50]) == (40, 40, 40)    # inside
    assert tuple(frame[40, 59]) == (40, 40, 40)    # d = 1: fully covered
    assert tuple(frame[40, 60]) == (130, 130, 130)  # d = 0: half covered, (220 + 40) / 2
    assert tuple(frame[40, 61]) == (220, 220, 220)  # d = -1: not covered
    assert tuple(frame[30, 50]) == (130, 130, 130)  # the same above the center (v = 30.5)


def test_led_box_switches_on_at_its_onset_frame():
    led = Led(box_px=(8, 6, 40, 28), onset_frame=3, off_rgb=(150, 140, 140), on_rgb=(255, 240, 180))
    scene = one_object_scene(Disk(8.0), Straight((80.0, 60.0), (0.5, 0.0)), n_frames=6, led=led)
    for frame in range(6):
        image = synthetic.render_frame(scene, frame)
        expected = (255, 240, 180) if frame >= 3 else (150, 140, 140)
        assert (image[6:28, 8:40] == expected).all()  # the slice [v0:v1, u0:u1]
        outside = image.copy()
        outside[6:28, 8:40] = scene.background
        assert (outside[:32, :44] == scene.background).all()  # nothing drawn around the box


def test_dish_is_drawn_lighter_inside_than_outside():
    scene = one_object_scene(Disk(8.0), Straight((60.0, 45.0), (0.5, 0.0)), dish=(60.0, 45.0, 40.0))
    image = synthetic.render_frame(scene, 0)
    assert tuple(image[45, 60 + 30]) == (scene.background,) * 3  # 30 px from the center: inside
    assert tuple(image[45, 60 + 45]) == (scene.outside,) * 3     # 45 px from the center: outside
    assert scene.outside < scene.background and scene.outside > 128  # not "dark" for a threshold at 128


# ---------------------------------------------------------------------------------------------
# Encoded clips (session fixtures of tests/helpers.py)

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
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------------------------
# Ready-made scenes


def inside_circle(mask, circle):
    """Whether every pixel center of a mask lies inside a circle (u, v, radius) in px."""
    rows, cols = np.nonzero(mask)
    return bool((np.hypot(cols + 0.5 - circle[0], rows + 0.5 - circle[1]) < circle[2]).all())


def touches_border(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


@pytest.mark.parametrize("size, n_frames", [((1920, 1080), 480), (SMALL, 120)])
def test_dish_scene_has_what_spec_13_2_lists(size, n_frames):
    scene = synthetic.dish_scene() if size == (1920, 1080) else synthetic.dish_scene(size=size, n_frames=n_frames)
    assert (scene.size, scene.n_frames, scene.fps, scene.mm_per_px) == (size, n_frames, 240.0, 0.0324)
    truth = GroundTruth(scene)
    by_id = {obj.track_id: obj for obj in scene.objects}
    assert sorted(by_id) == ["A", "B", "C"]
    frames = range(0, n_frames, max(n_frames // 12, 1))
    # the dish circle lies inside the frame
    cu, cv, radius = scene.dish
    assert radius < cu < size[0] - radius and radius < cv < size[1] - radius
    # A: a shrimp-like object, body about 14 px (0.47 mm at 32.4 um/px), near the wall, on a curved path
    shrimp = by_id["A"].shape
    assert isinstance(shrimp, Shrimp) and 13.0 < 2 * shrimp.a_px < 16.0
    assert (shrimp.beat_hz, shrimp.beta0_deg, shrimp.beat_deg) == (9.0, 45.0, 30.0)
    assert isinstance(by_id["A"].path, Arc)
    for frame in frames:
        u, v, _ = by_id["A"].path.pose(frame)
        assert 0 < radius - np.hypot(u - cu, v - cv) < 15.0  # the body center is under 15 px from the wall
        assert inside_circle(truth.mask("A", frame), scene.dish)
    # B and C: straight paths with known speed, passing within 1 px without overlapping
    contact = scene.contact
    assert contact.track_ids == ("B", "C") and 0 < contact.frame < n_frames - 1
    body_b, body_c = by_id["B"].shape, by_id["C"].shape
    assert isinstance(body_b, Ellipse) and body_b == body_c and 13.0 < 2 * body_b.a_px < 16.0
    assert by_id["B"].path.speed_px_per_frame == pytest.approx(5.0 / 0.0324 / 240.0)  # 5 mm/s
    ub, vb, heading_b = by_id["B"].path.pose(contact.frame)
    uc, vc, heading_c = by_id["C"].path.pose(contact.frame)
    assert (heading_b, abs(heading_c)) == pytest.approx((0.0, np.pi))  # side by side, opposite ways
    assert ub == pytest.approx(uc)
    gap = abs(vc - vb) - 2 * body_b.b_px  # two parallel ellipses side by side
    assert 0 < gap < 1 and contact.gap_px == pytest.approx(gap)
    mask_b, mask_c = truth.mask("B", contact.frame), truth.mask("C", contact.frame)
    assert not (mask_b & mask_c).any()
    assert (cv2.dilate(mask_b.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & mask_c).any()  # <= 2 px
    for frame in (0, n_frames - 1):  # far apart at both ends: the contact is an event
        assert abs(by_id["B"].path.pose(frame)[0] - by_id["C"].path.pose(frame)[0]) > 4 * body_b.a_px
    for track in "BC":
        assert all(inside_circle(truth.mask(track, frame), scene.dish) for frame in frames)
    # the LED box: inside the frame, outside the dish, switching on inside the clip
    u0, v0, u1, v1 = scene.led.box_px
    assert 0 <= u0 < u1 <= size[0] and 0 <= v0 < v1 <= size[1]
    assert min(np.hypot(u - cu, v - cv) for u in (u0, u1) for v in (v0, v1)) > radius
    assert 0 < scene.led.onset_frame < n_frames - 1


@pytest.mark.parametrize("size, n_frames", [((1920, 1080), 480), (SMALL, 120), (SMALL, 480)])
def test_closeup_scene_has_the_spec_shrimp(size, n_frames):
    scene = synthetic.closeup_scene() if size == (1920, 1080) else synthetic.closeup_scene(size=size, n_frames=n_frames)
    assert (scene.size, scene.n_frames, scene.fps, scene.mm_per_px) == (size, n_frames, 240.0, 0.010)
    assert [obj.track_id for obj in scene.objects] == ["A", "B"]
    shrimp, body = scene.objects[0].shape, scene.objects[1].shape
    assert (2 * shrimp.a_px, 2 * shrimp.b_px) == pytest.approx((47.0, 20.0))  # body 0.47 x 0.20 mm at 10 um/px
    assert (shrimp.antenna_length_px, shrimp.antenna_width_px) == pytest.approx((30.0, 3.0))
    assert 0 < shrimp.attach_px < shrimp.a_px  # attached near the front, on the axis
    assert (shrimp.beat_hz, shrimp.beta0_deg, shrimp.beat_deg) == (9.0, 45.0, 30.0)
    assert body == Ellipse(shrimp.a_px, shrimp.b_px)  # the same body without antennae
    assert scene.objects[0].path.speed_px_per_frame == pytest.approx(5.0 / 0.010 / 240.0)  # 5 mm/s
    truth = GroundTruth(scene)
    for frame in range(0, n_frames, max(n_frames // 16, 1)):
        mask_a, mask_b = truth.mask("A", frame), truth.mask("B", frame)
        assert mask_a.any() and mask_b.any()
        assert not touches_border(mask_a) and not touches_border(mask_b)
        assert not (cv2.dilate(mask_a.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool) & mask_b).any()


def test_disk_scene_is_dark_disks_of_radius_12_on_a_light_background():
    scene = synthetic.disk_scene()
    assert (scene.size, scene.n_frames, scene.fps) == (SMALL, 120, 240.0)
    assert scene.dish is None and scene.led is None
    assert len(scene.objects) >= 2
    assert all(obj.shape == Disk(12.0) and obj.gray < 128 for obj in scene.objects)
    assert scene.background > 128
    truth = GroundTruth(scene)
    for frame in range(0, 120, 7):
        masks = [truth.mask(obj.track_id, frame) for obj in scene.objects]
        assert not any(touches_border(mask) for mask in masks)
        centers = np.array([obj.path.pose(frame)[:2] for obj in scene.objects])
        apart = np.hypot(*(centers[:, None, :] - centers[None, :, :]).transpose(2, 0, 1))
        assert apart[np.triu_indices(len(centers), 1)].min() > 2 * 12.0 + 20.0  # never merged, never confused
    assert any(obj.path.pose(1)[0] % 1 not in (0.0, 0.5) for obj in scene.objects)  # sub-pixel positions


def test_shapes_scene_has_the_ellipse_and_the_disk_inside_a_dish():
    scene = synthetic.shapes_scene()
    assert (scene.size, scene.n_frames, scene.fps) == ((640, 480), 60, 240.0)
    by_id = {obj.track_id: obj for obj in scene.objects}
    assert by_id["A"].shape == Ellipse(40.0, 15.0) and by_id["B"].shape == Disk(50.0)
    truth = GroundTruth(scene)
    for frame in range(0, 60, 6):
        for track in "AB":
            mask = truth.mask(track, frame)
            assert mask.any() and inside_circle(mask, scene.dish) and not touches_border(mask)
    for track in "AB":  # sub-pixel positions: the centers do not sit on the pixel grid
        assert len({round(by_id[track].path.pose(frame)[0] % 1, 2) for frame in range(60)}) > 10
    # the ellipse moves and turns slowly: under 1 degree per frame, more than 10 degrees in all
    path = by_id["A"].path
    turn = abs(path.pose(1)[2] - path.pose(0)[2])
    assert 0 < np.degrees(turn) < 1.0 and np.degrees(turn) * 59 > 10.0
    assert 0.2 < path.speed_px_per_frame < 2.0
    assert 0 < by_id["B"].path.speed_px_per_frame < 2.0
    with pytest.raises(ValueError, match="640 x 480"):
        synthetic.shapes_scene(size=SMALL)
