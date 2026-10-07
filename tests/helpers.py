"""Helpers and fixtures shared by the new tests (registered in the root conftest.py).

New test helpers go here, not into tests/conftest.py, which is the template's file, unchanged.
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
