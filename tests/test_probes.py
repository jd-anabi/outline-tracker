"""Brightness probes (SPEC 4.6, 8.7; 13.2: the onset frame is exact; review focus 1, 2 and 3).

Where the expected values come from:
- the LED's box, its two colors and its onset frame are the synthetic scene's (`dish_clip.scene.led`);
- a box's pixels are the array slice worked out here from the rule "the pixel in column c and row r
  has its center at (c + 0.5, r + 0.5), and belongs to the box if that center lies inside", and the
  means are numpy means over that slice of frames decoded here with OpenCV alone;
- t_s = frame / fps_true, gray = 0.299 r + 0.587 g + 0.114 b.

Coordinates: boxes are (u0, v0, u1, v1) in image px, u to the right, v down (SPEC 3.1); frames are
the video's own frame numbers, counted from 0. The dish clip is 320 x 240 px and 120 frames long;
its LED is the slice [8:28, 8:40] and switches on at frame 41.
"""

import cv2
import numpy as np
import pytest

from outline_tracker import probes, schema, video

COLUMNS = ["frame", "t_s", "probe", "r", "g", "b", "gray"]  # SPEC 8.7
WEIGHTS = np.array([0.299, 0.587, 0.114])  # SPEC 4.6
FPS_TRUE = 239.6  # not the 240 written in the file: t_s comes from the argument, never from the file
N = 120  # frames in the dish clip
LAST = N - 1
ONSET = 41  # the dish clip's LED: 120 // 3 + 1
CORNER_BOX = (36, 24, 44, 33)  # 8 x 9 px over the LED's lower right corner (the LED ends at u = 40, v = 28)
DISH_BOX = (150, 110, 170, 130)  # inside the dish, away from the LED
NAN = float("nan")


def decode(path):
    """Every frame that decodes, in file order, as RGB uint8 arrays [row, column, 3] (OpenCV's BGR reversed)."""
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                return frames
            frames.append(bgr[:, :, ::-1])
    finally:
        capture.release()


@pytest.fixture(scope="module")
def frames(dish_clip):
    """The 120 frames of the dish clip, decoded here and not by the package."""
    decoded = decode(dish_clip.path)
    assert len(decoded) == N
    return decoded


def slice_means(frames, numbers, rows, cols):
    """Mean R, G, B over frames[k][rows, cols] for each k in `numbers`: a float array [len(numbers), 3]."""
    return np.array([frames[k][rows, cols].reshape(-1, 3).mean(axis=0) for k in numbers])


def rgb_of(table):
    return table[["r", "g", "b"]].to_numpy()


def gray_of(rgb):
    return float(WEIGHTS @ np.asarray(rgb, dtype=float))


# ---------------------------------------------------------------------------------------------
# The LED of the synthetic scene


def test_led_onset_is_exactly_the_scenes_onset_frame(dish_clip):
    led = dish_clip.scene.led
    assert led.onset_frame == ONSET
    messages = []
    table = probes.measure_probes(dish_clip.path, {"LED1": led.box_px}, 0, LAST, FPS_TRUE, log=messages.append)
    midpoint = (gray_of(led.off_rgb) + gray_of(led.on_rgb)) / 2
    on = table["gray"] > midpoint
    assert int(table.loc[on, "frame"].iloc[0]) == ONSET
    assert on.tolist() == [frame >= ONSET for frame in range(N)]  # off on every frame before, on from then on
    assert messages == []  # every frame asked for was there: nothing to say


def test_onset_frame_is_the_videos_frame_number_when_the_pass_starts_later(dish_clip):
    led = dish_clip.scene.led
    table = probes.measure_probes(dish_clip.path, {"LED1": led.box_px}, 30, 60, FPS_TRUE)
    midpoint = (gray_of(led.off_rgb) + gray_of(led.on_rgb)) / 2
    assert table["frame"].tolist() == list(range(30, 61))
    assert int(table.loc[table["gray"] > midpoint, "frame"].iloc[0]) == ONSET


def test_onset_is_found_in_a_folder_with_a_space_and_non_ascii_characters(clip_in_odd_folder):
    led = clip_in_odd_folder.scene.led
    table = probes.measure_probes(clip_in_odd_folder.path, {"LED1": led.box_px}, 0, LAST, FPS_TRUE)
    midpoint = (gray_of(led.off_rgb) + gray_of(led.on_rgb)) / 2
    assert int(table.loc[table["gray"] > midpoint, "frame"].iloc[0]) == ONSET


def test_led_colors_are_the_scenes_red_green_and_blue(dish_clip):
    # The LED's colors differ per channel (on: 255, 240, 180), so an exchanged channel order would
    # show. 4 px inside the box, away from the color bleeding at its edges, the decoded H.264 frames
    # are within 3.3 levels of the drawn colors (measured on the clip); 5 is allowed.
    led = dish_clip.scene.led
    u0, v0, u1, v1 = led.box_px
    inner = (u0 + 4, v0 + 4, u1 - 4, v1 - 4)
    table = probes.measure_probes(dish_clip.path, {"LED1": inner}, 0, LAST, FPS_TRUE)
    assert rgb_of(table)[:ONSET] == pytest.approx(np.tile(led.off_rgb, (ONSET, 1)), abs=5)
    assert rgb_of(table)[ONSET:] == pytest.approx(np.tile(led.on_rgb, (N - ONSET, 1)), abs=5)
    assert table["gray"].to_numpy()[:ONSET] == pytest.approx(gray_of(led.off_rgb), abs=5)
    assert table["gray"].to_numpy()[ONSET:] == pytest.approx(gray_of(led.on_rgb), abs=5)


# ---------------------------------------------------------------------------------------------
# Which pixels a box holds


def test_integer_corners_select_the_array_slice(dish_clip, frames):
    u0, v0, u1, v1 = CORNER_BOX
    table = probes.measure_probes(dish_clip.path, {"corner": CORNER_BOX}, 0, LAST, FPS_TRUE)
    expected = slice_means(frames, range(N), slice(v0, v1), slice(u0, u1))
    assert rgb_of(table) == pytest.approx(expected, abs=1e-9)
    assert table["gray"].to_numpy() == pytest.approx(expected @ WEIGHTS, abs=1e-9)
    # What makes this a test of the slice: the box lies over the LED's corner, so a slice one pixel
    # off in u or in v, or with u and v exchanged, has other means (by more than one level).
    others = [(slice(v0, v1), slice(u0 + 1, u1 + 1)), (slice(v0 + 1, v1 + 1), slice(u0, u1)),
              (slice(u0, u1), slice(v0, v1))]
    for rows, cols in others:
        assert np.abs(slice_means(frames, range(N), rows, cols) - expected).max() > 1.0


@pytest.mark.parametrize("box, rows, cols", [
    # centers 36.5 ... 43.5 and 25.5 ... 31.5 lie inside; 24.5 and 32.5 lie outside
    ((36.4, 24.6, 43.6, 32.4), slice(25, 32), slice(36, 44)),
    # a center on the low edge belongs to the box, a center on the high edge does not, so two
    # boxes that share an edge share no pixel, and a 7 x 8 px box holds 7 x 8 pixels
    ((36.5, 24.5, 43.5, 32.5), slice(24, 32), slice(36, 43)),
    # two opposite corners in any order (two clicks, SPEC 4.6)
    ((44, 33, 36, 24), slice(24, 33), slice(36, 44)),
    ((36, 33, 44, 24), slice(24, 33), slice(36, 44)),
    # partly outside the frame: the pixels inside it
    ((-10, -5.5, 12, 11), slice(0, 11), slice(0, 12)),
    ((150, 200, 400, 300), slice(200, 240), slice(150, 320)),
])
def test_a_box_holds_the_pixels_whose_centers_lie_inside_it(dish_clip, frames, box, rows, cols):
    numbers = range(36, 48)  # frames before and after the LED's onset
    table = probes.measure_probes(dish_clip.path, {"box": box}, numbers[0], numbers[-1], FPS_TRUE)
    expected = slice_means(frames, numbers, rows, cols)
    assert rgb_of(table) == pytest.approx(expected, abs=1e-9)
    assert table["gray"].to_numpy() == pytest.approx(expected @ WEIGHTS, abs=1e-9)


@pytest.mark.parametrize("box", [
    (10, 10, 10, 20),  # no width
    (10, 10, 20, 10),  # no height
    (10.6, 10.6, 11.4, 11.4),  # between the pixel centers 10.5 and 11.5
    (320, 10, 330, 20),  # right of the frame, which is 320 px wide
    (10, 240, 20, 250),  # below the frame, which is 240 px high
    (-20, -20, 0, 0),  # above and left of it
])
def test_an_empty_box_raises_value_error(dish_clip, box):
    with pytest.raises(ValueError, match="LED1"):
        probes.measure_probes(dish_clip.path, {"dish": DISH_BOX, "LED1": box}, 0, 10, FPS_TRUE)


# ---------------------------------------------------------------------------------------------
# The table


def test_two_boxes_give_two_rows_per_frame(dish_clip, frames):
    led = dish_clip.scene.led.box_px
    numbers = range(38, 45)
    table = probes.measure_probes(dish_clip.path, {"LED1": led, "DISH": DISH_BOX}, 38, 44, FPS_TRUE)
    assert len(table) == 2 * len(numbers)
    assert table["frame"].tolist() == [frame for frame in numbers for _ in range(2)]
    assert table["probe"].tolist() == ["LED1", "DISH"] * len(numbers)  # in the order given, not by name
    for name, (u0, v0, u1, v1) in (("LED1", led), ("DISH", DISH_BOX)):
        rows = table[table["probe"] == name]
        assert rgb_of(rows) == pytest.approx(slice_means(frames, numbers, slice(v0, v1), slice(u0, u1)), abs=1e-9)


def test_table_has_the_schemas_columns_and_every_frame_from_start_to_end(dish_clip):
    assert [column.name for column in schema.PROBES] == COLUMNS
    table = probes.measure_probes(dish_clip.path, {"LED1": dish_clip.scene.led.box_px}, 7, 31, FPS_TRUE)
    assert list(table.columns) == COLUMNS
    assert table["frame"].tolist() == list(range(7, 32))  # step 1, both ends included
    assert table["frame"].dtype.kind == "i"
    assert table["t_s"].to_numpy() == pytest.approx(np.arange(7, 32) / FPS_TRUE, abs=1e-12)
    assert set(table["probe"]) == {"LED1"}
    assert [table[name].dtype for name in ("t_s", "r", "g", "b", "gray")] == [np.float64] * 5
    assert table.index.tolist() == list(range(25))
    # The rows go into probes.csv as they are (the command writes the file).
    lines = schema.csv_text(schema.PROBES, table.to_dict("records")).splitlines()
    assert lines[0] == "frame,t_s,probe,r,g,b,gray" and len(lines) == 1 + 25
    assert lines[1].startswith(f"7,{7 / FPS_TRUE:.7f},LED1,")


def test_one_frame_gives_one_row(dish_clip):
    table = probes.measure_probes(dish_clip.path, {"LED1": dish_clip.scene.led.box_px}, 50, 50, FPS_TRUE)
    assert table["frame"].tolist() == [50]


def test_the_video_is_decoded_once_from_its_start_whatever_the_number_of_boxes(dish_clip, monkeypatch):
    calls = []
    decoder = video.iter_rgb_frames

    def counting(path, frame_numbers):
        calls.append(list(frame_numbers))
        return decoder(path, frame_numbers)

    monkeypatch.setattr(video, "iter_rgb_frames", counting)
    boxes = {"LED1": dish_clip.scene.led.box_px, "DISH": DISH_BOX, "corner": CORNER_BOX}
    probes.measure_probes(dish_clip.path, boxes, 20, 29, FPS_TRUE)
    assert calls == [list(range(20, 30))]  # one pass of the sequential decoder (SPEC 3.5), frames 20 to 29


# ---------------------------------------------------------------------------------------------
# A video that ends before `end` (review focus 2)


def test_a_clip_shorter_than_end_stops_at_its_last_frame_and_says_so(dish_clip):
    boxes = {"LED1": dish_clip.scene.led.box_px, "DISH": DISH_BOX}
    messages = []
    table = probes.measure_probes(dish_clip.path, boxes, 100, 500, FPS_TRUE, log=messages.append)
    assert table["frame"].tolist() == [frame for frame in range(100, N) for _ in range(2)]
    assert len(messages) == 1
    assert f"frame {LAST}" in messages[0] and "500" in messages[0]
    quiet = probes.measure_probes(dish_clip.path, boxes, 100, 500, FPS_TRUE)  # no callback: the same rows
    assert quiet.equals(table)
    probes.measure_probes(dish_clip.path, boxes, 100, N, FPS_TRUE, log=messages.append)  # one frame too many
    assert len(messages) == 2 and f"frame {N} could not be read" in messages[1]


def test_a_file_that_reports_more_frames_than_it_holds_stops_at_the_last_decodable_frame(dish_clip, tmp_path):
    # The first 80% of the file's bytes: the header still announces 120 frames, the last ones are gone.
    cut = tmp_path / "cut_tracker.mp4"
    data = dish_clip.path.read_bytes()
    cut.write_bytes(data[: len(data) * 8 // 10])
    decodable = len(decode(cut))
    assert video.probe(cut).n_frames == N and 0 < decodable < N  # the premise of this test
    messages = []
    table = probes.measure_probes(cut, {"LED1": dish_clip.scene.led.box_px}, 0, LAST, FPS_TRUE, log=messages.append)
    assert table["frame"].tolist() == list(range(decodable))
    assert len(messages) == 1
    assert f"frame {decodable - 1}" in messages[0] and str(N) in messages[0]


def test_a_start_past_the_last_frame_gives_an_empty_table_and_says_so(dish_clip):
    boxes = {"LED1": dish_clip.scene.led.box_px}
    messages = []
    table = probes.measure_probes(dish_clip.path, boxes, 200, 300, FPS_TRUE, log=messages.append)
    assert len(table) == 0 and list(table.columns) == COLUMNS
    assert table.dtypes.equals(probes.measure_probes(dish_clip.path, boxes, 0, 1, FPS_TRUE).dtypes)
    assert len(messages) == 1 and "200" in messages[0]


# ---------------------------------------------------------------------------------------------
# Nonsense (review focus 3) and a missing file


@pytest.mark.parametrize("changes, named", [
    ({"start": -1}, "start"),
    ({"start": 50, "end": 49}, "end"),
    ({"fps_true": 0.0}, "fps_true"),
    ({"fps_true": -240.0}, "fps_true"),
    ({"fps_true": NAN}, "fps_true"),
    ({"fps_true": float("inf")}, "fps_true"),
    ({"boxes": {}}, "probe"),
    ({"boxes": {"": DISH_BOX}}, "name"),
    ({"boxes": {"LED1": (10, 10, 20)}}, "LED1"),  # three numbers
    ({"boxes": {"LED1": (10, NAN, 20, 20)}}, "LED1"),
    ({"boxes": {"LED1": (10, 10, float("inf"), 20)}}, "LED1"),
    ({"boxes": {"LED1": (10, "left", 20, 20)}}, "LED1"),
    ({"boxes": {"LED1": 10}}, "LED1"),
])
def test_nonsense_raises_value_error_naming_what_is_wrong(dish_clip, changes, named):
    arguments = {"boxes": {"LED1": DISH_BOX}, "start": 0, "end": 10, "fps_true": FPS_TRUE} | changes
    with pytest.raises(ValueError, match=named):
        probes.measure_probes(dish_clip.path, **arguments)


def test_a_missing_video_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        probes.measure_probes(tmp_path / "missing.mp4", {"LED1": DISH_BOX}, 0, 10, FPS_TRUE)
