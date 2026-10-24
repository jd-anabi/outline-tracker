"""Tests of outline_tracker/core/session.py: the video's path in session.json, and finding the video again.

Expected values come from hashlib (the video's identity: its size and the SHA-256 of its first bytes),
from paths worked by hand, and from the files that each test makes. Nothing is copied from the output
of the code under test.

The rest of the tests of the session are in tests/core/test_session.py (session.json <-> dataclasses,
the optional and unknown keys, the world frame) and tests/core/test_session_file.py (the file on disk,
the version). What the three share is in tests/support/session_json_helpers.py.
"""

import hashlib
import json
import ntpath
import os
import posixpath
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from session_json_helpers import VIDEO_NAME

from outline_tracker.core import session as session_module
from outline_tracker.core.fileio import relative_path
from outline_tracker.core.session import (
    Clip,
    Session,
    TimeSettings,
    VideoNotFoundError,
    VideoRef,
    WrongVideoError,
)


# --------------------------------------------------------------------------- relative paths

def test_relative_path_from_the_run_folder():
    run = "/data/videos/clip_outline_ana"
    assert relative_path("/data/videos/clip.mp4", run, posixpath) == "../clip.mp4"
    assert relative_path("/data/videos/clip_outline_ana/clip.mp4", run, posixpath) == "clip.mp4"
    assert relative_path("/media/day 1/clip.mp4", run, posixpath) == "../../../media/day 1/clip.mp4"


def test_relative_path_on_windows_is_stored_with_forward_slashes():
    run = "C:\\data\\videos\\clip_outline_ana"
    assert relative_path("C:\\data\\videos\\clip.mp4", run, ntpath) == "../clip.mp4"
    assert relative_path("C:\\data\\videos\\day 1\\clip.mp4", "C:\\data\\runs\\ana", ntpath) == "../../videos/day 1/clip.mp4"
    assert relative_path("c:\\DATA\\videos\\clip.mp4", run, ntpath) == "../clip.mp4"  # one drive, whatever the case


def test_a_video_on_another_windows_drive_has_no_relative_path():
    run = "C:\\outline\\runs\\clip_outline_ana"
    assert relative_path("G:\\My Drive\\videos\\clip.mp4", run, ntpath) is None
    assert relative_path("\\\\server\\share\\videos\\clip.mp4", run, ntpath) is None  # a network share


def test_a_video_without_a_relative_path_stores_null_and_is_found_by_its_absolute_path(tmp_path, monkeypatch):
    video = tmp_path / "videos" / VIDEO_NAME
    _video(video)
    run = tmp_path / "runs" / "clip_outline_ana"
    monkeypatch.setattr(session_module, "relative_path", lambda *args, **kwargs: None)  # as on another drive
    s = Session(video=VideoRef.from_file(video, run, width=1920, height=1080, n_frames=2400, fps_container=240.0))
    assert s.video.relpath is None
    s.save(run / "session.json")
    stored = json.loads((run / "session.json").read_text(encoding="utf-8"))["video"]
    assert stored["relpath"] is None  # null in the file
    assert stored["abspath"] == str(video)
    assert Session.load(run / "session.json").locate_video(run) == video


# --------------------------------------------------------------------------- finding the video again

def _video(path: Path, seed: int = 0, size: int = 5000) -> bytes:
    """A stand-in for a video file: identity is its size and its first bytes, never its content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(seed).bytes(size)
    path.write_bytes(data)
    return data


def _saved_run(root: Path) -> tuple[Path, Path, bytes]:
    """`root/videos/<video>` and the default run folder next to it (SPEC 8.1), with a saved session."""
    video = root / "videos" / VIDEO_NAME
    data = _video(video)
    run = video.parent / (video.stem + "_outline_ana")
    s = Session(student="ana", clip=Clip(start=0, end=2399, step=2), time=TimeSettings(fps_true=239.6, source="typed"),
                video=VideoRef.from_file(video, run, width=1920, height=1080, n_frames=2400, fps_container=240.0))
    assert s.save(run / "session.json") == run / "session.json"
    return video, run, data


def test_video_is_stored_relative_to_the_run_folder_with_its_identity(tmp_path):
    video, run, data = _saved_run(tmp_path / "a")
    v = Session.load(run / "session.json").video
    assert v.relpath == "../" + VIDEO_NAME  # relative to the run folder, forward slash
    assert v.abspath == str(video)
    assert v.size == 5000
    assert v.sha256_first_64mib == hashlib.sha256(data).hexdigest()
    assert (v.width, v.height, v.n_frames, v.fps_container) == (1920, 1080, 2400, 240.0)
    stored = json.loads((run / "session.json").read_text(encoding="utf-8"))["video"]
    assert stored["sha256_first_64MiB"] == hashlib.sha256(data).hexdigest()  # the key as SPEC 8.10 spells it
    assert Session.load(run / "session.json").locate_video(run) == video


def test_video_path_given_relative_to_the_current_folder(tmp_path, monkeypatch):
    video, run, _ = _saved_run(tmp_path / "a")
    monkeypatch.chdir(video.parent)
    ref = VideoRef.from_file(VIDEO_NAME, run.name, width=1920, height=1080, n_frames=2400, fps_container=240.0)
    assert ref.relpath == "../" + VIDEO_NAME
    assert Path(ref.abspath) == Path.cwd() / VIDEO_NAME
    found = Session.load(Path(run.name) / "session.json").locate_video(run.name)
    assert found.is_absolute() and found == Path.cwd() / VIDEO_NAME


def test_the_whole_folder_can_be_moved(tmp_path):
    video, run, _ = _saved_run(tmp_path / "a")
    shutil.move(tmp_path / "a", tmp_path / "moved here")  # video and run folder together, e.g. to another disk
    new_run = tmp_path / "moved here" / "videos" / run.name
    s = Session.load(new_run / "session.json")
    assert not Path(s.video.abspath).exists()  # the absolute path is stale
    assert s.locate_video(new_run) == tmp_path / "moved here" / "videos" / VIDEO_NAME


def test_a_run_folder_moved_alone_finds_the_video_by_its_absolute_path(tmp_path):
    video, run, _ = _saved_run(tmp_path / "a")
    new_run = tmp_path / "elsewhere" / "deep" / run.name
    new_run.parent.mkdir(parents=True)
    shutil.move(run, new_run)
    s = Session.load(new_run / "session.json")
    assert not (new_run / s.video.relpath).exists()  # the relative path leads nowhere now
    assert s.locate_video(new_run) == video


def test_a_moved_video_is_reported_missing_then_accepted_where_the_user_points(tmp_path):
    video, run, _ = _saved_run(tmp_path / "a")
    new_video = tmp_path / "drive" / "renamed.mp4"
    new_video.parent.mkdir()
    shutil.move(video, new_video)
    s = Session.load(run / "session.json")
    with pytest.raises(VideoNotFoundError) as err:
        s.locate_video(run)
    assert isinstance(err.value, FileNotFoundError)
    assert VIDEO_NAME in str(err.value)  # which video to look for
    assert str(video) in str(err.value)  # and where it was looked for

    s.video.check(new_video)  # the file the user picked is the session's video: no error
    s.video.point_to(new_video, run)
    assert s.video.relpath == "../../../drive/renamed.mp4"
    assert s.video.abspath == str(new_video)
    s.save(run / "session.json")
    assert Session.load(run / "session.json").locate_video(run) == new_video


def test_a_different_file_in_the_videos_place_is_refused(tmp_path):
    video, run, data = _saved_run(tmp_path / "a")
    s = Session.load(run / "session.json")

    _video(video, seed=1)  # the same size, other content (say, converted again)
    with pytest.raises(WrongVideoError) as err:
        s.locate_video(run)
    assert isinstance(err.value, ValueError)
    assert VIDEO_NAME in str(err.value)
    assert "content" in str(err.value)

    video.write_bytes(data[:4000])  # another size (say, a download that stopped)
    with pytest.raises(WrongVideoError) as err:
        s.locate_video(run)
    assert VIDEO_NAME in str(err.value)
    sizes = re.sub(r"[,. ]", "", str(err.value).replace(str(tmp_path), ""))
    assert "4000" in sizes and "5000" in sizes  # both sizes, in bytes, however the digits are grouped

    video.write_bytes(data)  # the right file again
    assert s.locate_video(run) == video


def test_check_refuses_another_file_and_a_missing_one(tmp_path):
    video, run, data = _saved_run(tmp_path / "a")
    s = Session.load(run / "session.json")
    other = tmp_path / "other.mp4"
    _video(other, seed=2)
    with pytest.raises(WrongVideoError, match="other.mp4"):
        s.video.check(other)
    with pytest.raises(VideoNotFoundError, match="nothing.mp4"):
        s.video.check(tmp_path / "nothing.mp4")
    copy_of_it = tmp_path / "copy.mp4"
    copy_of_it.write_bytes(data)
    s.video.check(copy_of_it)  # a copy of the same file is the same video
    s.video.check(str(copy_of_it))


def test_a_wrong_file_at_the_relative_path_does_not_hide_the_right_one(tmp_path):
    video, run, _ = _saved_run(tmp_path / "a")
    new_run = tmp_path / "b" / run.name
    new_run.parent.mkdir()
    shutil.move(run, new_run)
    _video(new_run.parent / VIDEO_NAME, seed=5)  # another video of the same name next to the moved run folder
    assert Session.load(new_run / "session.json").locate_video(new_run) == video


def test_a_session_without_a_video_is_reported_missing(tmp_path):
    with pytest.raises(VideoNotFoundError):
        Session().locate_video(tmp_path)


def test_a_path_from_another_kind_of_computer_is_reported_missing_by_name(tmp_path):
    # a run folder made on Windows and opened on a Mac, or the other way round, without its video
    foreign = "/Volumes/lab/videos/clip.mp4" if os.name == "nt" else "G:\\My Drive\\videos\\clip.mp4"
    s = Session(video=VideoRef(relpath=None, abspath=foreign, size=5000, sha256_first_64mib="ab" * 32))
    with pytest.raises(VideoNotFoundError) as err:
        s.locate_video(tmp_path)
    assert "The video clip.mp4 " in str(err.value)  # the file's name, whichever slashes the path has
    assert "videos" in str(err.value)  # and where it was
