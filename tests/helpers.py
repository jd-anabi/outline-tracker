"""Helpers shared by the tests of every folder: plain functions and values, imported by name.

Fixtures are not here: the ones for every test are in tests/conftest.py, where pytest finds them by
name. What tests import by name is here or in a helper module beside the tests, never in a
conftest.py (tests/test_repo_rules.py says why).
"""

# ---------------------------------------------------------------------------------------------
# Synthetic clips with ground truth (outline_tracker/synthetic.py): the clip fixtures of
# tests/conftest.py are made with these.

SMALL = (320, 240)  # frame size of the fast tests' clips: (width, height) in px
ODD_FOLDER = "vidéo test ü"  # a space and non-ASCII characters (review focus 1)

# ---------------------------------------------------------------------------------------------
# A clip whose timestamps have gaps (tests/test_frame_source.py), as a phone video with dropped
# frames has: there, plain OpenCV seeking delivers a neighbor of the frame asked for.

# Frame numbers before which one frame duration is skipped (72 twice: two durations). Three gaps.
GAPS_BEFORE = (30, 72, 72, 101)


def frame_times(skip_before=(), n_frames=120, fps=240.0):
    """The time stamped on every frame of a clip written by `synthetic.render`, in s, frame 0 at 0:
    one frame duration (1 / fps) each, plus one more for every entry of `skip_before` up to the frame."""
    import numpy as np

    frames = np.arange(n_frames)
    return (frames + sum((frames >= before).astype(int) for before in skip_before)) / fps


# ---------------------------------------------------------------------------------------------
# Tracker's export files (tests/test_tracker_io.py, tests/from_tracker_helpers.py)


def java_sci(v):
    """A number the way Tracker writes it with Number Format "Full Precision" (Java's 0.000000E0)."""
    if v == 0:
        return "0.000000E0"
    mantissa, exponent = f"{v:.6E}".split("E")
    return f"{mantissa}E{int(exponent)}"


# ---------------------------------------------------------------------------------------------
# What a command printed, what a folder holds, a text to search in


def error_line(capsys) -> str:
    """The one `ERROR: ...` line the command printed on stderr (asserted to be exactly one line)."""
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ") and err.endswith("\n") and err.count("\n") == 1, err
    return err


def _names(folder) -> list[str]:
    """The names of the files and folders in `folder`, sorted."""
    return sorted(p.name for p in folder.iterdir())


def normalized(text: str) -> str:
    """Whitespace collapsed to single spaces, so a wrapped paragraph can be searched."""
    return " ".join(text.split())


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


def stand_in_segmenter(model, device):
    """What a GUI test gives the window in place of the real model's factory: a `ThresholdFake`, whatever
    the model key and the device are (images and prompts in px of the image given, segmenter/base.py).
    No torch is loaded."""
    from outline_tracker.segmenter.fake import ThresholdFake

    return ThresholdFake()


def new_window(qtbot):
    """Make a `MainWindow` with `stand_in_segmenter`, not shown yet, and close it afterwards: a
    generator for a fixture to `yield from`, which hands out the window once. Run on after the test,
    it closes the window and then the window's controller, which releases the video. The `window`
    fixture of tests/conftest.py is this, and so is a second window that a test asks for."""
    from outline_tracker.gui.main_window import MainWindow

    made = MainWindow(segmenter_factory=stand_in_segmenter)
    qtbot.addWidget(made)
    yield made
    made.close()
    made.controller.close()
