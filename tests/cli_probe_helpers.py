"""Shared by the tests of the `probe` command (tests/test_cli_probe.py: a video and `--rect`;
tests/test_cli_probe_session.py: a session file, and run.log).

Where the expected values come from: the synthetic dish clip (320 x 240 px, 120 frames; its LED is the
array slice [8:28, 8:40] and switches on at frame 41 = 120 // 3 + 1), frames decoded here with OpenCV
alone, and t_s = frame / fps_true. Rectangles are (u0, v0, u1, v1) in image px, u to the right, v down
(SPEC 3.1): whole-number corners select the slice [v0:v1, u0:u1].
"""

import cv2
import numpy as np
import pandas as pd

from outline_tracker import cli

COLUMNS = ["frame", "t_s", "probe", "r", "g", "b", "gray"]  # SPEC 8.7
WEIGHTS = np.array([0.299, 0.587, 0.114])  # SPEC 4.6
FPS = 239.6  # fps_true of the tests; the file itself says 240, which must never be used
N = 120  # frames in the dish clip
ONSET = 41  # the dish clip's LED switches on at this frame
LED_BOX = (8, 8, 40, 28)  # the scene's LED (asserted in the tests that rely on it)
WALL_BOX = (150, 110, 170, 130)  # inside the dish, away from the LED
LED = "LED1:8,8,40,28"
WALL = "WALL:150,110,170,130"
UNKNOWN_FPS = "ERROR: Unknown fps_true: give it with --fps (e.g. --fps 239.6), or fill in data/manifest.csv.\n"


def write_manifest(folder, fps, video_file="dish.MOV"):
    """<folder>/data/manifest.csv with one row for the dish clip's original and `fps` in its fps_true cell."""
    path = folder / "data" / "manifest.csv"
    path.parent.mkdir(parents=True)
    path.write_text(f"video_file,group,fps_true,stick_mm\n{video_file},B,{fps},30\n", encoding="utf-8")
    return path


def probe(*args) -> int:
    """Run `outline-tracker probe ARGS...` in this process and return its exit code."""
    return cli.main(["probe", *(str(arg) for arg in args)])


def table(folder) -> pd.DataFrame:
    """The probes.csv of a folder."""
    return pd.read_csv(folder / "probes.csv")


def error_line(capsys) -> str:
    """The one `ERROR: ...` line the command printed on stderr (asserted to be exactly one line)."""
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ") and err.endswith("\n") and err.count("\n") == 1, err
    return err


def decode(path) -> list[np.ndarray]:
    """Every frame of a video, in file order, as RGB uint8 arrays [row, column, 3], with OpenCV alone."""
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


def slice_means(frames, numbers, box) -> np.ndarray:
    """Mean R, G, B inside the whole-number box (u0, v0, u1, v1) on each frame of `numbers`: [n, 3]."""
    u0, v0, u1, v1 = box
    return np.array([frames[k][v0:v1, u0:u1].reshape(-1, 3).mean(axis=0) for k in numbers])


def assert_rows(rows: pd.DataFrame, frames, numbers, box, fps_true=FPS) -> None:
    """The rows of one probe hold `numbers` as frames, t_s = frame / fps_true and the means inside `box`
    of the decoded `frames`, as written with 7 decimals (t_s) and 3 decimals (the means)."""
    numbers = list(numbers)
    assert rows["frame"].tolist() == numbers
    np.testing.assert_allclose(rows["t_s"], np.array(numbers) / fps_true, rtol=0, atol=0.6e-7)
    means = slice_means(frames, numbers, box)
    np.testing.assert_allclose(rows[["r", "g", "b"]].to_numpy(), means, rtol=0, atol=0.6e-3)
    np.testing.assert_allclose(rows["gray"], means @ WEIGHTS, rtol=0, atol=0.6e-3)


def onset_frame(rows: pd.DataFrame, led) -> int:
    """The first frame whose gray lies above the midpoint between the LED's off and on colors."""
    midpoint = (WEIGHTS @ np.array(led.off_rgb, float) + WEIGHTS @ np.array(led.on_rgb, float)) / 2
    return int(rows.loc[rows["gray"] > midpoint, "frame"].iloc[0])
