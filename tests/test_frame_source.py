"""Exact random access (SPEC 3.5, 13.6; decisions X4 and X8): `FrameSource.get(k)` is the k-th frame
of the sequential decode, bit for bit, also where plain OpenCV seeking is not.

Two clips, both H.264 with B-frames (stored out of presentation order): `disk_clip`, evenly timed,
and `gapped_clip`, the same frames with three gaps in its timestamps. Frame numbers count from 0 in
decoding order from the start of the file; times are s of file time, 1/240 s per frame. Expected
timestamps come from how the clips are written (`GAPS_BEFORE`, `frame_times`), expected frames
from the sequential decode `video.iter_rgb_frames`: decoded frames are compared with decoded
frames, never with drawn ones (the codec changes the levels). What video.py itself adds is tested
in tests/test_video_frames.py.
"""

import hashlib
import shutil
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from helpers import GAPS_BEFORE, frame_times

from outline_tracker import frame_source, video
from outline_tracker.frame_source import FrameSource

CLIPS = ["disk_clip", "gapped_clip"]
SKIPS = {"disk_clip": (), "gapped_clip": GAPS_BEFORE}
N = 120  # frames in each clip
SHAPE = (240, 320, 3)  # rows, columns, RGB


def packet_times(path):
    """Presentation time in s of every video packet, in the order the file stores them.

    Read here with the bundled ffmpeg (no decoding), not through the code under test.
    """
    done = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
                           "-map", "0:v:0", "-c", "copy", "-f", "framecrc", "-"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    numerator, denominator = next(line for line in lines if line.startswith("#tb")).split(":")[1].split("/")
    stamps = [int(line.split(",")[2]) for line in lines if line and not line.startswith("#")]
    return np.array(stamps) * int(numerator) / int(denominator)


def frames_to_test(n=N, seed=13):
    """20 seeded random frame numbers plus 0, 1, 23, 24, 25, n - 2, n - 1: 27 different ones, shuffled."""
    fixed = [0, 1, 23, 24, 25, n - 2, n - 1]
    rng = np.random.default_rng(seed)
    others = [k for k in range(n) if k not in fixed]
    frames = fixed + [int(k) for k in rng.choice(others, 20, replace=False)]
    rng.shuffle(frames)
    return frames


def naive_seek_and_read(path, k, shift=0):
    """What plain OpenCV gives for frame k when the seek is `shift` frames off: RGB, or None."""
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(k + shift, 0))
        ok, bgr = capture.read()
    finally:
        capture.release()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None


def shift_seeks(source, shift, monkeypatch):
    """Make every seek of `source` ask for a frame `shift` frames away; returns the list of its calls."""
    calls, seek = [], source._seek

    def shifted(frame):
        calls.append(frame)
        seek(max(frame + shift, 0))

    monkeypatch.setattr(source, "_seek", shifted)
    return calls


def same(frame, expected):
    return frame is not None and frame.dtype == np.uint8 and np.array_equal(frame, expected)


@pytest.fixture(scope="session")
def sequential(disk_clip, gapped_clip):
    """Every frame of both clips from the sequential decode: {fixture name: [RGB frame 0, 1, ...]}."""
    clips = {"disk_clip": disk_clip, "gapped_clip": gapped_clip}
    return {name: [rgb for _, rgb in video.iter_rgb_frames(clip.path, range(N + 5))] for name, clip in clips.items()}


@pytest.fixture(params=CLIPS)
def clip(request):
    """(fixture name, path of the clip) for each of the two clips."""
    return request.param, request.getfixturevalue(request.param).path


# ---------------------------------------------------------------------------------------------
# The fixtures are what the tests need: B-frames in both, three timestamp gaps in one


@pytest.mark.parametrize("name", CLIPS)
def test_clips_store_their_packets_out_of_timestamp_order(name, request, sequential):
    stored = packet_times(request.getfixturevalue(name).path)
    assert len(stored) == N
    assert (np.diff(stored) < 0).any()  # B-frames: a later packet is shown earlier
    assert len(sequential[name]) == N  # counted by decoding (5 more were asked for)
    assert {frame.shape for frame in sequential[name]} == {SHAPE}
    assert len({hashlib.sha256(frame.tobytes()).digest() for frame in sequential[name]}) == N  # all different


def test_gapped_clip_has_three_timestamp_gaps_and_the_even_clip_none(disk_clip, gapped_clip):
    even = np.diff(np.sort(packet_times(disk_clip.path))) * 240.0  # steps in frame durations
    gapped = np.diff(np.sort(packet_times(gapped_clip.path))) * 240.0
    assert even == pytest.approx(1.0, abs=1e-6)
    assert list(np.flatnonzero(gapped > 1.5) + 1) == [30, 72, 101]  # the frames after each gap
    assert gapped[[29, 71, 100]] == pytest.approx([2.0, 3.0, 2.0], abs=1e-6)
    assert np.delete(gapped, [29, 71, 100]) == pytest.approx(1.0, abs=1e-6)


def test_gapped_clip_shows_the_same_frames_in_the_same_order(sequential):
    # The gaps are in the timestamps only: no frame is dropped, doubled or moved.
    even = [frame.astype(np.int16) for frame in sequential["disk_clip"]]
    for k, frame in enumerate(sequential["gapped_clip"]):
        error = {j: np.abs(frame - even[j]).mean() for j in (k - 1, k, k + 1) if 0 <= j < N}
        assert min(error, key=error.get) == k


# ---------------------------------------------------------------------------------------------
# FrameSource.get(k): bit-identical to frame k of the sequential decode


def test_get_is_bit_identical_to_the_sequential_decode(clip, sequential):
    name, path = clip
    with FrameSource(path) as source:
        for k in frames_to_test():
            frame = source.get(k)
            assert frame.shape == SHAPE
            assert same(frame, sequential[name][k]), f"frame {k} of {name} differs"


def test_jumps_use_the_verified_seek_and_never_restart_from_frame_0(clip):
    # Exactness is the test above, on every platform. This one says that the fast way is in use:
    # if it fails, frames are still exact, but every jump back decodes from the start (slow).
    name, path = clip
    with FrameSource(path) as source:
        assert source.timestamps_s == pytest.approx(frame_times(SKIPS[name]), abs=1e-6)  # the table is in use
        for k in frames_to_test():
            source.get(k)
        assert source.stats["seek"] > 0, "no jump used the verified seek"
        assert source.stats["from_zero"] == 0, "a jump fell back to decoding from frame 0"


def test_a_clip_in_a_folder_with_a_space_and_non_ascii_characters(clip_in_odd_folder):
    path = clip_in_odd_folder.path
    wanted = [100, 3, 50, 119, 0]
    sequential_frames = dict(video.iter_rgb_frames(path, sorted(wanted)))
    with FrameSource(path) as source:
        assert source.timestamps_s is not None and len(source.timestamps_s) == N  # ffmpeg read the odd path too
        for k in wanted:
            assert same(source.get(k), sequential_frames[k]), f"frame {k} differs"


@pytest.mark.parametrize("table", ["missing", "repeated times", "another length"])
def test_get_without_a_usable_timestamp_table_decodes_from_the_start(table, clip, sequential, monkeypatch):
    name, path = clip
    tables = {"missing": None, "repeated times": np.zeros(N), "another length": np.arange(N - 1) / 240.0}
    monkeypatch.setattr(video, "frame_timestamps", lambda path: tables[table])
    with FrameSource(path) as source:
        assert source.timestamps_s is None
        for k in frames_to_test():
            assert same(source.get(k), sequential[name][k]), f"frame {k} of {name} differs"
        assert source.stats["seek"] == 0
        assert source.stats["from_zero"] > 0


@pytest.mark.parametrize("shift", [-2, -1, 1, 2, 6])
def test_a_seek_that_lands_on_another_frame_never_gives_a_wrong_frame(shift, clip, sequential, monkeypatch):
    # Regression guard by fault injection: every seek of FrameSource goes through `_seek`; here it
    # lands `shift` frames away from the frame asked for. FrameSource identifies the frame that
    # arrived and still returns frame k; a plain seek-and-read with the same fault does not.
    name, path = clip
    frames = frames_to_test()
    with FrameSource(path) as source:
        calls = shift_seeks(source, shift, monkeypatch)
        for k in frames:
            assert same(source.get(k), sequential[name][k]), f"frame {k} of {name} differs (seek off by {shift})"
        assert calls, "no seek was made, so nothing was injected"
    naive_wrong = [k for k in frames if not same(naive_seek_and_read(path, k, shift), sequential[name][k])]
    assert naive_wrong, "the injected fault would go unnoticed"


def test_how_often_plain_seeking_is_wrong_is_printed(disk_clip, gapped_clip, sequential):
    # Not asserted: this is OpenCV's behavior (measured on the Mac: exact on the evenly timed clip,
    # one frame early or late on the gapped one). Shown with `pytest -s`; the number goes into the docs.
    frames = frames_to_test()
    for name, truth in (("disk_clip", disk_clip), ("gapped_clip", gapped_clip)):
        wrong = [k for k in frames if not same(naive_seek_and_read(truth.path, k), sequential[name][k])]
        print(f"plain cv2 seeking on {name}: {len(wrong)} of {len(frames)} frames wrong {sorted(wrong)}")
    assert len(frames) == 27 and len(set(frames)) == 27


def test_get_outside_the_clip_is_an_index_error_and_the_source_goes_on(clip, sequential, monkeypatch):
    name, path = clip
    for usable_table in (True, False):
        if not usable_table:
            monkeypatch.setattr(video, "frame_timestamps", lambda path: None)
        with FrameSource(path) as source:
            for k in (-1, N, N + 400):
                with pytest.raises(IndexError, match=str(k)):
                    source.get(k)
                assert same(source.get(N - 1), sequential[name][N - 1])
                assert same(source.get(7), sequential[name][7])


def test_the_end_of_the_video_is_searched_for_at_most_once(disk_clip, monkeypatch):
    with FrameSource(disk_clip.path) as source:  # the table lists 120 frames: no decoding needed
        for k in (N, N + 400):
            with pytest.raises(IndexError):
                source.get(k)
        assert source.stats["seek"] == 0 and source.stats["from_zero"] == 0
    monkeypatch.setattr(video, "frame_timestamps", lambda path: None)
    with FrameSource(disk_clip.path) as source:  # no table: only decoding to the end shows it
        with pytest.raises(IndexError):
            source.get(N)
        restarts = source.stats["from_zero"]
        for k in (N, N + 1, N + 400):
            with pytest.raises(IndexError):
                source.get(k)
        assert source.stats["from_zero"] == restarts


def test_info_is_what_the_file_says(gapped_clip):
    with FrameSource(gapped_clip.path) as source:
        assert source.info == video.probe(gapped_clip.path)
        assert (source.info.width, source.info.height, source.info.n_frames) == (320, 240, N)


def test_a_file_that_cannot_be_opened_is_an_error_and_stays_unlocked(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        FrameSource(tmp_path / "missing.mp4")
    text = tmp_path / "notes.mp4"
    text.write_text("this is text, not a video\n")
    with pytest.raises(IOError, match="notes.mp4"):
        FrameSource(text)
    text.unlink()  # on Windows this fails while a capture holds the file


def test_close_releases_the_file_and_get_then_raises(gapped_clip, sequential, tmp_path):
    copy = tmp_path / "copy_tracker.mp4"
    shutil.copyfile(gapped_clip.path, copy)
    with FrameSource(copy) as source:
        assert same(source.get(90), sequential["gapped_clip"][90])  # a seek
        assert same(source.get(3), sequential["gapped_clip"][3])
    with pytest.raises(ValueError, match="closed"):
        source.get(3)
    source.close()  # a second close is harmless
    copy.unlink()  # on Windows this fails while a capture holds the file
    assert not copy.exists()


# ---------------------------------------------------------------------------------------------
# The cache: at most 64 frames or 400 MB, least recently used out first


def test_cache_keeps_the_most_recently_used_frames(disk_clip, sequential):
    with FrameSource(disk_clip.path, cache_frames=3) as source:
        first = [source.get(k) for k in (0, 1, 2)]
        assert source.cached_frames == [0, 1, 2]
        assert source.get(0) is first[0] and source.stats["cache"] == 1  # served from the cache
        assert source.cached_frames == [1, 2, 0]
        source.get(3)
        assert source.cached_frames == [2, 0, 3]  # frame 1 was the least recently used
        again = source.get(1)
        assert again is not first[1] and same(again, sequential["disk_clip"][1])  # decoded again, the same
        assert not again.flags.writeable  # a caller cannot change a cached frame
        assert source.stats["cache"] == 1


def test_cache_holds_64_frames_unless_told_otherwise(disk_clip):
    with FrameSource(disk_clip.path) as source:
        for k in range(N):
            source.get(k)
            assert len(source.cached_frames) <= 64
        assert source.cached_frames == list(range(N - 64, N))


def test_cache_never_holds_more_than_400_mb(disk_clip, monkeypatch):
    assert frame_source.CACHE_BYTES == 400_000_000
    assert 400_000_000 // (1080 * 1920 * 3) == 64  # 64 frames of 1080p fit (398 MB); 16 of 4K
    frame_bytes = 240 * 320 * 3
    monkeypatch.setattr(frame_source, "CACHE_BYTES", 2 * frame_bytes + frame_bytes // 2)  # room for 2.5 frames
    with FrameSource(disk_clip.path, cache_frames=64) as source:
        for k in range(10):
            source.get(k)
            assert sum(source.get(j).nbytes for j in source.cached_frames) <= frame_source.CACHE_BYTES
        assert source.cached_frames == [8, 9]


# ---------------------------------------------------------------------------------------------
# The frame hash (decision X8) of frames that came through FrameSource


def test_frame_hash_tells_decoded_frames_apart(gapped_clip, sequential):
    frames = sequential["gapped_clip"]
    with FrameSource(gapped_clip.path) as source:
        for k in (0, 71, 72, 73, N - 1):
            assert video.frame_hash(source.get(k)) == video.frame_hash(frames[k])
            assert video.frame_hash(source.get(k)) != video.frame_hash(frames[k - 1])


# ---------------------------------------------------------------------------------------------
# check_seek: what `outline-tracker check VIDEO --seek` reports


@pytest.mark.parametrize("name", CLIPS)
def test_check_seek_compares_20_random_frames_with_the_sequential_decode(name, request):
    result = frame_source.check_seek(request.getfixturevalue(name).path)
    assert (result.n_frames, result.tested, result.exact) == (N, 20, 20)
    assert result.gaps == len(set(SKIPS[name]))  # 0, and 3


def test_check_seek_without_a_table_reports_no_gap_count(gapped_clip, monkeypatch):
    monkeypatch.setattr(video, "frame_timestamps", lambda path: None)
    result = frame_source.check_seek(gapped_clip.path)
    assert (result.tested, result.exact, result.gaps) == (20, 20, None)


def test_check_seek_tests_every_frame_of_a_clip_shorter_than_20(tmp_path):
    from outline_tracker import synthetic

    scene = synthetic.disk_scene(n_frames=12)
    truth = synthetic.render(scene, tmp_path / "short_tracker.mp4", crf=10)
    result = frame_source.check_seek(truth.path)
    assert (result.n_frames, result.tested, result.exact, result.gaps) == (12, 12, 12, 0)


def test_check_seek_counts_a_wrong_frame_as_not_exact(gapped_clip, monkeypatch):
    # A random access that is one frame early (as plain seeking is on a clip with gaps) must show.
    get = FrameSource.get
    monkeypatch.setattr(FrameSource, "get", lambda self, k: get(self, max(k - 1, 0)))
    result = frame_source.check_seek(gapped_clip.path)
    assert result.tested == 20 and result.exact <= 1  # only frame 0, if it was drawn, still matches
