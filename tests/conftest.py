"""Fixtures for the tests of every folder: synthetic clips with ground truth, the frames of two of them
from the sequential decode and those of the dish clip from OpenCV alone, a tracked run folder, a test's
own copy of the run folder its module tracked, a file that another program holds open, the settings
kept out of the user's own, the main window. pytest finds a fixture by its name; nothing imports this
file (tests/test_repo_rules.py says why). The plain helpers and values that tests import by name are in
tests/helpers.py and in the helper modules beside it.
"""

import errno
import math
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

# pytest rewrites the asserts of test modules, conftest.py files and plugins, and helpers.py was a
# plugin while it held these fixtures. Asked for here, before its first import, so that a failing
# assert in it (`assert_empty_result`) still shows the values it compared.
pytest.register_assert_rewrite("helpers")

from frame_source_helpers import CLIPS, N  # noqa: E402
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
# The frames of two clips from the sequential decode, for the tests of the time stamps
# (tests/test_frame_source_times.py, tests/test_frame_source_jumps.py)


@pytest.fixture(scope="module")
def sequential_frames(disk_clip, gapped_clip):
    """Every frame of both clips from the sequential decode: {fixture name: [RGB frame 0, 1, ...]}.
    Decoded once for each test module that asks for it."""
    from outline_tracker import video

    clips = {"disk_clip": disk_clip, "gapped_clip": gapped_clip}
    return {name: [rgb for _, rgb in video.iter_rgb_frames(clip.path, range(N))] for name, clip in clips.items()}


@pytest.fixture(params=CLIPS)
def clip_with_frames(request, sequential_frames):
    """(fixture name, path of the clip, its frames from the sequential decode) for each of the two clips."""
    return request.param, request.getfixturevalue(request.param).path, sequential_frames[request.param]


# ---------------------------------------------------------------------------------------------
# The frames of the dish clip as OpenCV alone decodes them, for the tests of the `probe` command
# (tests/test_cli_probe.py, tests/test_cli_probe_session.py)


@pytest.fixture(scope="module")
def frames(dish_clip):
    """The 120 frames of the dish clip, decoded by `decode` of tests/cli_probe_helpers.py and not by the
    package. Decoded once for each test module that asks for it. tests/test_probes.py has a `frames` of
    its own, which overrides this one there."""
    from cli_probe_helpers import LED_BOX, N, ONSET, decode

    decoded = decode(dish_clip.path)
    assert len(decoded) == N and dish_clip.scene.led.box_px == LED_BOX and dish_clip.scene.led.onset_frame == ONSET
    return decoded


# ---------------------------------------------------------------------------------------------
# A tracked run folder, for the tests that change a run after it was tracked (tests/test_corrections.py,
# tests/test_corrections_gaps.py, tests/test_tracking_edit.py)


@pytest.fixture(scope="module")
def tracked_dish_folder(dish_clip, tmp_path_factory):
    """A run folder in which A, B and C of the dish clip are tracked on every grid frame. Made once for
    each test module that asks for it; a test changes a copy of it (`tracked_dish_run`), never this folder."""
    from outline_tracker.segmenter.fake import ExactFake
    from tracking_helpers import abc_session, run

    folder = tmp_path_factory.mktemp("tracked")
    assert run(dish_clip, abc_session(dish_clip, folder), folder, ExactFake(dish_clip))[0] == "complete"
    return folder


@pytest.fixture
def tracked_dish_run(tracked_dish_folder, tmp_path):
    """This test's own copy of that run, in a folder named with a space and non-ASCII characters:
    (run folder, session, store)."""
    from outline_tracker.results import ResultsStore
    from outline_tracker.schema import RESULTS_NPZ, SESSION_JSON
    from outline_tracker.session import Session

    folder = tmp_path / ODD_FOLDER
    shutil.copytree(tracked_dish_folder, folder)
    return folder, Session.load(folder / SESSION_JSON), ResultsStore.load(folder / RESULTS_NPZ)


# ---------------------------------------------------------------------------------------------
# A test's own copy of the run folder that its module tracked, for the tests of the export
# (tests/test_export.py, tests/test_export_tracker_folder.py, tests/test_cli_export.py,
# tests/test_cli_export_edits.py)


@pytest.fixture
def run_folder(tracked, tmp_path):
    """This test's own copy of the tracked run folder (its video is found through the absolute path).

    `tracked` is not a fixture of this file: each module that asks for `run_folder` has its own, the
    run folder it tracks once. tests/test_export_corrections.py and tests/test_export_log.py have a
    `run_folder` of their own, which overrides this one there.
    """
    return shutil.copytree(tracked, tmp_path / "run")


# ---------------------------------------------------------------------------------------------
# A file that another program holds open (Windows: a CSV open in a spreadsheet program, a file being
# synced). The package renames a finished file onto its target with `os.replace`, in
# outline_tracker/fileio.py only; a locked target is an `os.replace` that raises.


class LockedFiles:
    """What `lock_file` returns: what was tried while the files were locked, and the way to free them.

    attempts: the destination of every rename (`os.replace`) since the call, as a Path, in order,
    whether it was refused or not. sleeps: every wait asked of `time.sleep` since the call, in s, in
    order (recorded with waits="skip" only).
    """

    def __init__(self, targets, times, error, rename):
        self.attempts: list[Path] = []
        self.sleeps: list[float] = []
        self._error, self._rename = error, rename
        self._left: dict[Path | str, float] = {}  # refusals left: by Path for a file, by name for a bare name
        for target in targets:
            path = Path(target)
            self._left[path.name if len(path.parts) == 1 else path] = times

    def release(self) -> None:
        """The other program closes the files: from now on no rename is refused."""
        self._left.clear()

    def _replace(self, src, dst, **kwargs):
        """In place of `os.replace`: refuse a locked destination, rename onto any other."""
        path = Path(dst)
        self.attempts.append(path)
        for locked in (path, path.name):
            if self._left.get(locked, 0) > 0:
                self._left[locked] -= 1
                if self._error is not None:
                    raise self._error
                raise PermissionError(errno.EACCES, "The file is being used by another process", str(dst))
        return self._rename(src, dst, **kwargs)


@pytest.fixture
def lock_file(monkeypatch):
    """`lock_file(*targets, times=math.inf, error=None, waits="skip")` lets files be open in another
    program: a rename onto one of them is refused, and the file stays as it was.

    targets: a path locks that file; a bare file name locks every file of that name, in any folder.
    times: how many renames onto each target are refused before one goes through (default: all).
    error: the exception to raise for a refused rename (default: PermissionError, as Windows raises
    for a file in use, which is what `fileio` takes for a lock). waits: "skip" makes `time.sleep`
    return at once, so that the tries of `fileio` (about 5 s) take no time; "none" takes the waits
    out of `fileio` (`RETRY_DELAYS_S = ()`), which then tries once and goes on to the fallback.

    Returns a `LockedFiles`. The files are free again after its `release()`, after the test, and
    after `monkeypatch.undo()` in the test: the patches are made with the test's own `monkeypatch`.
    `os.replace` is patched for the whole process, so a rename in another thread is refused as well.
    """
    from outline_tracker import fileio

    def lock(*targets, times=math.inf, error=None, waits="skip"):
        if waits not in ("skip", "none"):
            raise ValueError(f'waits is "skip" or "none", not {waits!r}')
        held = LockedFiles(targets, times, error, os.replace)
        monkeypatch.setattr(os, "replace", held._replace)
        if waits == "skip":
            monkeypatch.setattr(fileio.time, "sleep", held.sleeps.append)
        else:
            monkeypatch.setattr(fileio, "RETRY_DELAYS_S", ())
        return held

    return lock


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
