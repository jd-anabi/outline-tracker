"""The time-stamp side of exact random access (SPEC 3.5, 13.6; decision X4).

`FrameSource` learns which frame a seek delivered from that frame's time stamp and a table of all
frame times. tests/test_frame_source.py makes the seek wrong; here the evidence is made wrong: a
table that is not this file's, a decoder that reports other times. `get(k)` must still be frame k
of the sequential decode, bit for bit, by decoding from frame 0 wherever the two do not agree.

The clips, the 27 frame numbers and the helpers are those of tests/test_frame_source.py. Frame
numbers count from 0; times are s or ms of file time, as each name says; one frame step is 1/240 s,
the shortest step between two frames of both clips. Expected frames come from the sequential decode
`video.iter_rgb_frames`. The faults of the first two parts are injected: none of them was seen on a
real file. The last part is a real file: a copy cut without re-encoding, which hides its first frames.
"""

import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from helpers import frame_times
from test_frame_source import CLIPS, N, SKIPS, frames_to_test, same

from outline_tracker import video
from outline_tracker.frame_source import FrameSource

STEP_S = 1.0 / 240.0  # one frame step in s
STEP_MS = 1000.0 * STEP_S  # and in ms
LATER = 50  # the faults that begin inside the clip begin at this frame
FAR = N - 3  # read first: more than 64 frames ahead of where a source is after opening, so a jump


@pytest.fixture(scope="module")
def sequential(disk_clip, gapped_clip):
    """Every frame of both clips from the sequential decode: {fixture name: [RGB frame 0, 1, ...]}."""
    clips = {"disk_clip": disk_clip, "gapped_clip": gapped_clip}
    return {name: [rgb for _, rgb in video.iter_rgb_frames(clip.path, range(N))] for name, clip in clips.items()}


@pytest.fixture(params=CLIPS)
def clip(request, sequential):
    """(fixture name, path of the clip, its frames from the sequential decode) for each of the two clips."""
    return request.param, request.getfixturevalue(request.param).path, sequential[request.param]


def change_the_table(monkeypatch, change):
    """Make `video.frame_timestamps` return `change(times_s)` of the file's real table."""
    real = video.frame_timestamps
    monkeypatch.setattr(video, "frame_timestamps", lambda path: change(real(path)))


def change_the_decoder_times(monkeypatch, change):
    """Make every FrameSource read each time stamp of the decoder as `change(real_ms)`; returns the
    list of the real readings, so that a test can see that the fault was reached."""
    read, seen = FrameSource._time_ms, []

    def changed(self):
        seen.append(read(self))
        return change(seen[-1])

    monkeypatch.setattr(FrameSource, "_time_ms", changed)
    return seen


def time_of_frame_0_ms(path):
    """What OpenCV reports as the time of frame 0, in ms (read here, not through FrameSource)."""
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.grab()
        return capture.get(cv2.CAP_PROP_POS_MSEC)
    finally:
        capture.release()


def every_tested_frame_is_exact(source, frames):
    """Read frame 117 (a jump, if the table is in use) and then the 27 frames in their shuffled
    order, which begins with steps forward; `frames` is the clip's sequential decode."""
    for k in [FAR, *frames_to_test()]:
        assert same(source.get(k), frames[k]), f"frame {k} differs"
    return True


# ---------------------------------------------------------------------------------------------
# A table that is not the decoder's


TABLES = {
    "scaled": lambda times_s: times_s * 1.37,
    "scaled and offset": lambda times_s: times_s * 1.37 + 0.0003,
    "offset by 7% of a step": lambda times_s: times_s + 0.0003,
    "one step late after frame 0": lambda times_s: times_s + STEP_S * (np.arange(N) >= 1),
}


@pytest.mark.parametrize("fault", list(TABLES))
def test_a_table_that_is_not_the_decoders_is_not_used(fault, clip, monkeypatch):
    # Each table has the right length and increases, so only the decoder's own times can show that
    # it is wrong. Frames 0 and 1 show it when the file is opened.
    _, path, frames = clip
    change_the_table(monkeypatch, TABLES[fault])
    with FrameSource(path) as source:
        assert every_tested_frame_is_exact(source, frames)
        assert source.timestamps_s is None and source.stats["seek"] == 0  # the table was never in use
        assert source.stats["from_zero"] > 0


@pytest.mark.parametrize("fault_steps", [0.3, 1.0])
def test_a_table_that_fits_only_the_first_frames_is_given_up(fault_steps, clip, monkeypatch):
    # From frame 50 on the table is 0.3 or 1 frame step late; frames 0 and 1 cannot show that.
    # Frame 60 is read 11 frames after frame 49, so it is reached by counting, and the count is
    # the truth: the table disagrees with it and is given up. (A jump instead would not be placed
    # with this table either. A time 0.3 of a step off is no entry's time, see the next test; a
    # whole step off is the time the table gives the frame before, which no single time stamp can
    # tell from the truth, and there the look at the far end of the file stops the jump:
    # tests/test_frame_source_jumps.py.)
    _, path, frames = clip
    change_the_table(monkeypatch, lambda times_s: times_s + fault_steps * STEP_S * (np.arange(N) >= LATER))
    with FrameSource(path) as source:
        for k in (20, LATER - 1):
            assert same(source.get(k), frames[k]), f"frame {k} differs"
        assert source.timestamps_s is not None and source.stats["from_zero"] == 0  # nothing wrong so far
        assert same(source.get(LATER + 10), frames[LATER + 10])
        assert source.timestamps_s is None, "the table was kept after the count showed it wrong"
        assert source.stats["from_zero"] == 1
        assert every_tested_frame_is_exact(source, frames)
        assert source.stats["seek"] == 0


def test_a_table_that_stops_fitting_is_given_up_at_a_jump_too(clip, monkeypatch):
    # The same table, 0.3 of a step late from frame 50 on, met by jumps in the shuffled order: a
    # time 0.3 of a step away from every entry is no entry's time.
    _, path, frames = clip
    change_the_table(monkeypatch, lambda times_s: times_s + 0.3 * STEP_S * (np.arange(N) >= LATER))
    with FrameSource(path) as source:
        assert source.timestamps_s is not None
        assert every_tested_frame_is_exact(source, frames)
        assert source.timestamps_s is None
        assert source.stats["from_zero"] > 0


# ---------------------------------------------------------------------------------------------
# A decoder that reports other times


def test_decoder_times_that_start_elsewhere_are_placed_all_the_same(clip, monkeypatch):
    # Not a fault: every time stamp is 1234.5 ms later (a file whose first frame is not at 0). Only
    # the time since frame 0 counts, so the verified seek stays in use.
    _, path, frames = clip
    seen = change_the_decoder_times(monkeypatch, lambda real_ms: real_ms + 1234.5)
    with FrameSource(path) as source:
        assert every_tested_frame_is_exact(source, frames)
        assert seen, "no time stamp was read, so nothing was changed"
        assert source.timestamps_s is not None
        assert source.stats["seek"] > 0
        assert source.stats["from_zero"] == 0


@pytest.mark.parametrize("fault_steps", [-2, -1, 1, 2])
def test_a_wrong_time_for_frame_0_never_gives_a_wrong_frame(fault_steps, clip, monkeypatch):
    # The time of frame 0 is the reading every other time is measured from. Here the decoder
    # reports it 1 or 2 frame steps early or late (as one that takes the first frame's time from
    # the order in which the frames are stored would), and every other frame's time correctly.
    _, path, frames = clip
    first_ms = time_of_frame_0_ms(path)
    seen = change_the_decoder_times(
        monkeypatch, lambda real_ms: real_ms + fault_steps * STEP_MS if real_ms == first_ms else real_ms)
    with FrameSource(path) as source:
        assert first_ms in seen, "frame 0's time was not read, so nothing was injected"
        assert every_tested_frame_is_exact(source, frames)
        assert source.timestamps_s is None and source.stats["seek"] == 0  # the table was never in use
        assert source.stats["from_zero"] > 0


@pytest.mark.parametrize("fault_steps", [-1.0, -0.3, 0.3, 1.0, 2.0])
def test_a_wrong_time_after_a_seek_never_gives_a_wrong_frame(fault_steps, clip, monkeypatch):
    # The twin of the seek that lands elsewhere: the seek is right, and the time stamp read after
    # it is off. By 0.3 of a step it is no frame's time; by 1 or 2 steps it is another frame's.
    # Either way frame k then comes from frame 0 by counting, and the table, which is right, stays.
    _, path, frames = clip
    seek, read = FrameSource._seek, FrameSource._time_ms
    state = {"just sought": False, "wrong readings": 0}

    def seek_and_remember(self, frame):
        state["just sought"] = True
        seek(self, frame)

    def read_wrong_once_after_a_seek(self):
        real_ms = read(self)
        if not state["just sought"]:
            return real_ms
        state["just sought"] = False
        state["wrong readings"] += 1
        return real_ms + fault_steps * STEP_MS

    with FrameSource(path) as source:
        monkeypatch.setattr(FrameSource, "_seek", seek_and_remember)
        monkeypatch.setattr(FrameSource, "_time_ms", read_wrong_once_after_a_seek)
        assert every_tested_frame_is_exact(source, frames)
        assert state["wrong readings"] > 0, "no time stamp was read after a seek, so nothing was injected"
        assert source.stats["from_zero"] > 0
        assert source.timestamps_s is not None


def test_a_time_that_no_frame_has_is_not_given_to_the_nearest_frame(clip, monkeypatch):
    # From frame 50 on the decoder's times are 0.3 of a frame step late: nearer to the right entry
    # of the table than to any other, and still not that entry's time.
    name, path, frames = clip
    fault_from_ms = time_of_frame_0_ms(path) + 1000.0 * frame_times(SKIPS[name])[LATER] - 0.5 * STEP_MS
    seen = change_the_decoder_times(
        monkeypatch, lambda real_ms: real_ms + 0.3 * STEP_MS if real_ms > fault_from_ms else real_ms)
    with FrameSource(path) as source:
        assert source.timestamps_s is not None
        assert every_tested_frame_is_exact(source, frames)
        assert max(seen) > fault_from_ms, "no time from frame 50 on was read, so nothing was injected"
        assert source.timestamps_s is None  # the decoder's times are not the table's: given up
        assert source.stats["from_zero"] > 0


# ---------------------------------------------------------------------------------------------
# A file that hides its first frames (a real file, nothing injected)


@pytest.fixture(scope="module")
def cut_clip(gapped_clip, tmp_path_factory):
    """`gapped_clip` cut at 0.11 s without re-encoding: (path, [RGB frame 0, 1, ...] of the sequential decode).

    Such a copy starts at the keyframe before the cut (frame 24 of `gapped_clip`) and tells players
    not to show what lies before 0.11 s. Measured on the Mac: the file reports 96 frames and lists
    96 packets, the decoder delivers 93 (frame 0 of the copy is frame 27 of `gapped_clip`).
    """
    path = tmp_path_factory.mktemp("cut") / "cut_tracker.mp4"
    done = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", "0.11",
                           "-i", str(gapped_clip.path), "-c", "copy", str(path)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert done.returncode == 0, done.stderr
    return path, [rgb for _, rgb in video.iter_rgb_frames(path, range(N))]


def test_a_file_that_hides_its_first_frames_is_read_exactly(cut_clip):
    # With the hidden frames in its table, entry k was not frame k, and FrameSource as first
    # committed returned wrong frames from this file (measured on the Mac: 3 of these 94 reads, 14
    # of 150 random ones). How many frames a cut hides is ffmpeg's and the decoder's doing, so it is
    # printed (`pytest -s`), not asserted. What must hold everywhere: the table has one entry per
    # frame of the sequential decode, and every frame is exact, a jump first.
    path, frames = cut_clip
    n, reported = len(frames), video.probe(path).n_frames
    print(f"cut clip: the file reports {reported} frames, the sequential decode has {n}")
    assert 60 < n <= reported
    assert len(video.frame_timestamps(path)) == n
    with FrameSource(path) as source:
        for k in [n - 3, *(int(k) for k in np.random.default_rng(13).permutation(n))]:
            assert same(source.get(k), frames[k]), f"frame {k} differs"
