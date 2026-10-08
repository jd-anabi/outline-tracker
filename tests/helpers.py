"""Helpers and fixtures shared by the new tests (registered in the root conftest.py).

New test helpers go here, not into tests/conftest.py, which holds the helpers that the ported tests
import by name.
"""

import shutil
from dataclasses import replace

import pytest

# ---------------------------------------------------------------------------------------------
# Synthetic clips with ground truth (outline_tracker/synthetic.py). Each is rendered once per test
# session; a test must not write next to these files (use `clip_in_odd_folder` for that).

SMALL = (320, 240)  # frame size of the fast tests' clips: (width, height) in px
ODD_FOLDER = "vidéo test ü"  # a space and non-ASCII characters (review focus 1)


@pytest.fixture(scope="session")
def dish_clip(tmp_path_factory):
    """`dish_scene` at 320 x 240 px, 120 frames (0.5 s), H.264: the GroundTruth with its `path`."""
    from outline_tracker import synthetic

    scene = synthetic.dish_scene(size=SMALL, n_frames=120)
    return synthetic.render(scene, tmp_path_factory.mktemp("dish") / "dish_tracker.mp4")


@pytest.fixture(scope="session")
def closeup_clip(tmp_path_factory):
    """`closeup_scene` at 320 x 240 px, 120 frames (0.5 s), H.264: the GroundTruth with its `path`."""
    from outline_tracker import synthetic

    scene = synthetic.closeup_scene(size=SMALL, n_frames=120)
    return synthetic.render(scene, tmp_path_factory.mktemp("closeup") / "closeup_tracker.mp4")


@pytest.fixture(scope="session")
def disk_clip(tmp_path_factory):
    """`disk_scene` (320 x 240 px, 120 frames) encoded with crf 10, as `ThresholdFake` needs."""
    from outline_tracker import synthetic

    return synthetic.render(synthetic.disk_scene(), tmp_path_factory.mktemp("disks") / "disks_tracker.mp4", crf=10)


@pytest.fixture(scope="session")
def shapes_clip(tmp_path_factory):
    """`shapes_scene` (640 x 480 px, 60 frames), H.264: the GroundTruth with its `path`."""
    from outline_tracker import synthetic

    return synthetic.render(synthetic.shapes_scene(), tmp_path_factory.mktemp("shapes") / "shapes_tracker.mp4")


@pytest.fixture
def clip_in_odd_folder(tmp_path, dish_clip):
    """The dish clip inside a fresh folder named with a space and non-ASCII characters.

    Returns the GroundTruth with `path` = `<tmp_path>/vidéo test ü/dish_tracker.mp4`. The folder is
    this test's own, so run folders and exports may be written next to the clip.
    """
    folder = tmp_path / ODD_FOLDER
    folder.mkdir()
    path = folder / dish_clip.path.name
    shutil.copyfile(dish_clip.path, path)
    return replace(dish_clip, path=path)


# ---------------------------------------------------------------------------------------------
# A clip whose timestamps have gaps (tests/test_frame_source.py), as a phone video with dropped
# frames has: there, plain OpenCV seeking delivers a neighbor of the frame asked for.

# Frame numbers before which one frame duration is skipped (72 twice: two durations). Three gaps.
GAPS_BEFORE = (30, 72, 72, 101)


@pytest.fixture(scope="session")
def gapped_clip(tmp_path_factory):
    """The frames of `disk_clip` (320 x 240 px, 120 frames, crf 10, B-frames) with three gaps in the
    timestamps, before frames 30, 72 and 101: frame k is shown at (k + gaps so far) / 240 s."""
    from outline_tracker import synthetic

    path = tmp_path_factory.mktemp("gapped") / "gapped_tracker.mp4"
    return synthetic.render(synthetic.disk_scene(), path, crf=10, skip_before=GAPS_BEFORE)


def frame_times(skip_before=(), n_frames=120, fps=240.0):
    """The time stamped on every frame of a clip written by `synthetic.render`, in s, frame 0 at 0:
    one frame duration (1 / fps) each, plus one more for every entry of `skip_before` up to the frame."""
    import numpy as np

    frames = np.arange(n_frames)
    return (frames + sum((frames >= before).astype(int) for before in skip_before)) / fps


# ---------------------------------------------------------------------------------------------
# Segmenter results (tests/test_fakes.py, tests/test_fakes_threshold.py)


def click(obj_id, u=1.0, v=1.0):
    """A prompt with one positive point at (u, v), px in the pixel frame of the image given with it."""
    from outline_tracker.segmenter.base import ObjectPrompt

    return ObjectPrompt(obj_id, [(u, v)], [1])


def mask_in_image(result, shape):
    """A `MaskResult`'s mask put back where it was cut out: a bool array of `shape` = (rows, columns),
    the size of the image the segmenter was given; the result's offset is (column, row) in px of it."""
    import numpy as np

    mask = np.zeros(shape, bool)
    (col0, row0), (rows, cols) = result.offset, result.mask.shape
    mask[row0:row0 + rows, col0:col0 + cols] = result.mask
    return mask


def assert_empty_result(result):
    """The `MaskResult` of an object that was not found: nothing in it, as segmenter/base.py describes."""
    import numpy as np

    assert result.mask.shape == (0, 0) and result.mask.dtype == bool
    assert result.logits.shape == (0, 0) and result.logits.dtype == np.float32
    assert result.offset == (0, 0) and result.score is None


# ---------------------------------------------------------------------------------------------
# The main window (tests/gui/)


@pytest.fixture(scope="session", autouse=True)
def settings_out_of_the_users_own(tmp_path_factory):
    """What the window remembers between sittings (QSettings, an INI file in the user's scope) goes
    to a temporary folder for the whole test run, so that no test reads or writes the settings of
    the person who runs the tests. A test that needs a folder of its own sets one after this."""
    from PySide6.QtCore import QSettings

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path_factory.mktemp("settings")))


def stand_in_segmenter(model, device):
    """What a GUI test gives the window in place of the real model's factory: a `ThresholdFake`, whatever
    the model key and the device are (images and prompts in px of the image given, segmenter/base.py).
    No torch is loaded."""
    from outline_tracker.segmenter.fake import ThresholdFake

    return ThresholdFake()


@pytest.fixture
def window(qtbot):
    """A `MainWindow` made with `stand_in_segmenter`, not shown yet. It is closed after the test,
    whatever happened in it, and the video it had open is released (Windows cannot delete a file
    that is open): every GUI test gets its window from here."""
    from outline_tracker.gui.main_window import MainWindow

    made = MainWindow(segmenter_factory=stand_in_segmenter)
    qtbot.addWidget(made)
    yield made
    made.close()
    made.controller.close()
