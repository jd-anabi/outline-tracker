"""Fixtures for the tests of every folder: synthetic clips with ground truth, the settings kept out
of the user's own, the main window. pytest finds a fixture by its name; nothing imports this file
(tests/test_repo_rules.py says why). The plain helpers and values that tests import by name are in
tests/helpers.py and in the helper modules beside it.
"""

import shutil
from dataclasses import replace

import pytest

# pytest rewrites the asserts of test modules, conftest.py files and plugins, and helpers.py was a
# plugin while it held these fixtures. Asked for here, before its first import, so that a failing
# assert in it (`assert_empty_result`) still shows the values it compared.
pytest.register_assert_rewrite("helpers")

from helpers import GAPS_BEFORE, ODD_FOLDER, SMALL, new_window  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Synthetic clips with ground truth (outline_tracker/synthetic.py). Each is rendered once per test
# session; a test must not write next to these files (use `clip_in_odd_folder` for that).


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


@pytest.fixture(scope="session")
def gapped_clip(tmp_path_factory):
    """The frames of `disk_clip` (320 x 240 px, 120 frames, crf 10, B-frames) with three gaps in the
    timestamps, before frames 30, 72 and 101: frame k is shown at (k + gaps so far) / 240 s."""
    from outline_tracker import synthetic

    path = tmp_path_factory.mktemp("gapped") / "gapped_tracker.mp4"
    return synthetic.render(synthetic.disk_scene(), path, crf=10, skip_before=GAPS_BEFORE)


# ---------------------------------------------------------------------------------------------
# The main window (tests/gui/)


@pytest.fixture(scope="session", autouse=True)
def settings_out_of_the_users_own(tmp_path_factory):
    """What the window remembers between sittings (QSettings, an INI file in the user's scope) goes
    to a temporary folder for the whole test run, so that no test reads or writes the settings of
    the person who runs the tests. A test that needs a folder of its own sets one after this."""
    from PySide6.QtCore import QSettings

    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path_factory.mktemp("settings")))


@pytest.fixture
def window(qtbot):
    """A `MainWindow` made with `stand_in_segmenter`, not shown yet. It is closed after the test,
    whatever happened in it, and the video it had open is released (Windows cannot delete a file
    that is open): every GUI test gets its window from here."""
    yield from new_window(qtbot)
