"""What a jump of `FrameSource` rests on (SPEC 3.5, 13.6; decision X4): faults on the time-stamp
side that agree with themselves, read in any order.

tests/test_frame_source_times.py makes a table or a time stamp wrong so that one comparison shows
it. Here the wrong times fit each other: a table that is whole frame steps late from some frame on,
a decoder whose times are whole frame steps off after the file was opened or after a seek. Each
time read is then some entry's time, and the frame reached from a misplaced one carries the time
expected for frame k. Two things show such a fault: the count from frame 0, and the last frame of
the file, which is the frame after which none comes, whatever its number. `get(k)` must be frame k
of the sequential decode, bit for bit, in whatever order the frames are read (`read_orders`): which
checks a read passes through depends on whether it is a step forward or a jump.

The clips, the 27 frame numbers and the helpers are those of tests/frame_source_helpers.py. Frame numbers
count from 0; times are s or ms of file time, as each name says; one frame step is 1/240 s. Expected
frames come from the sequential decode `video.iter_rgb_frames`. Every fault here is injected: none
was seen on a real file.
"""

import numpy as np
import pytest
from frame_source_helpers import (FAR, LATER, N, STEP_MS, STEP_S, change_the_table, every_tested_frame_is_exact,
                                  frames_to_test, same, shift_seeks)

from outline_tracker.frame_source import SEEK_BACK, FrameSource


def read_orders():
    """The 27 frames in three orders, {name: [frame numbers]}: shuffled (it begins with steps
    forward), the same after a jump to frame 117, and shuffled backwards (it begins with a jump)."""
    frames = frames_to_test()
    return {"steps first": frames, "a jump first": [FAR, *frames], "backwards": frames[::-1]}


def without_one_frame(times_s, first=80, last=99):
    """A table (s) that has no entry for frame `first`, numbers the frames after it up to `last` one
    too low, and fits everywhere else: one entry half a frame step after frame `last` is added (same
    length, increasing; both clips step by 1/240 s from frame `last` to the next)."""
    return np.sort(np.append(np.delete(times_s, first), times_s[last] + 0.5 * STEP_S))


# ---------------------------------------------------------------------------------------------
# Faults that agree with themselves, in any order of reading


@pytest.mark.parametrize("fault_steps", [1.0, 2.0])
def test_a_table_that_is_late_from_some_frame_to_the_end_is_safe_in_any_read_order(fault_steps, clip_with_frames,
                                                                                   monkeypatch):
    # From frame 50 on every entry is one or two whole frame steps late. A time read there is another
    # entry's time, so no time stamp alone shows the fault, and a jump placed with this table would
    # deliver a neighbor of the frame asked for. The last frame of the file shows it: it does not
    # carry the table's last time. No jump is trusted before that was looked at.
    _, path, frames = clip_with_frames
    change_the_table(monkeypatch, lambda times_s: times_s + fault_steps * STEP_S * (np.arange(N) >= LATER))
    for order_name, order in read_orders().items():
        with FrameSource(path) as source:
            for k in order:
                assert same(source.get(k), frames[k]), f"frame {k} differs ({order_name})"
            assert source.stats["from_zero"] > 0, order_name
            assert source.timestamps_s is None, order_name  # a count went past frame 50 and showed it wrong


@pytest.mark.parametrize("fault_steps", [-2, 1])
def test_times_that_change_after_the_file_was_opened_never_give_a_wrong_frame(fault_steps, clip_with_frames,
                                                                              monkeypatch):
    # What the opening of the file established (the time of frame 0, from which every other time is
    # measured, and that frame 1 comes one table step later) is 2 frame steps early or 1 late for
    # every time read afterwards. Each such time is another frame's time. A count shows it at the
    # first frame stepped to; a jump shows it at the far end of the file, before any frame is placed.
    _, path, frames = clip_with_frames
    read = FrameSource._time_ms
    for order_name, order in read_orders().items():
        with FrameSource(path) as source, monkeypatch.context() as patch:
            patch.setattr(FrameSource, "_time_ms", lambda self: read(self) + fault_steps * STEP_MS)
            for k in order:
                assert same(source.get(k), frames[k]), f"frame {k} differs ({order_name})"
            assert source.stats["from_zero"] > 0, order_name
            assert source.timestamps_s is None, order_name


@pytest.mark.parametrize("fault_steps", [-1, 2])
def test_times_that_shift_after_every_seek_never_give_a_wrong_frame(fault_steps, clip_with_frames, monkeypatch):
    # After a seek the decoder reports every time 1 frame step early or 2 late, until the file is
    # opened again. The times read after one seek agree with each other: the frame that arrived is
    # misplaced by whole frames, and the frame reached from it carries the time expected for frame
    # k. Counted from frame 0 nothing is wrong, so the table, which is right, stays; but no jump is
    # trusted, because the last frame of the file, reached by a seek, is not where the table ends.
    _, path, frames = clip_with_frames
    opened, seek, read = FrameSource._open, FrameSource._seek, FrameSource._time_ms
    state = {"sought": False, "shifted readings": 0}

    def open_and_forget(self):
        state["sought"] = False
        return opened(self)

    def seek_and_remember(self, frame):
        state["sought"] = True
        seek(self, frame)

    def read_shifted_after_a_seek(self):
        state["shifted readings"] += state["sought"]
        return read(self) + (fault_steps * STEP_MS if state["sought"] else 0.0)

    monkeypatch.setattr(FrameSource, "_open", open_and_forget)
    monkeypatch.setattr(FrameSource, "_seek", seek_and_remember)
    monkeypatch.setattr(FrameSource, "_time_ms", read_shifted_after_a_seek)
    for order_name, order in read_orders().items():
        state["shifted readings"] = 0
        with FrameSource(path) as source:
            for k in order:
                assert same(source.get(k), frames[k]), f"frame {k} differs ({order_name})"
            assert state["shifted readings"] > 0, "no time was read after a seek, so nothing was injected"
            assert source.stats["from_zero"] > 0, order_name
            assert source.timestamps_s is not None, order_name


@pytest.mark.parametrize("fault_steps", [-1.0, 0.3, 2.0])
def test_a_wrong_time_after_a_later_seek_never_gives_a_wrong_frame(fault_steps, clip_with_frames, monkeypatch):
    # As test_a_wrong_time_after_a_seek_never_gives_a_wrong_frame in the other file, but the first
    # jump goes well, so jumps are trusted when the fault begins: the one wrong reading after each
    # seek then meets the placing of the frame that arrived and the comparison at every frame
    # stepped to, not the look at the far end.
    _, path, frames = clip_with_frames
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
        assert same(source.get(FAR), frames[FAR])
        assert source.stats["seek"] > 0 and source.stats["from_zero"] == 0  # a jump, and it was trusted
        monkeypatch.setattr(FrameSource, "_seek", seek_and_remember)
        monkeypatch.setattr(FrameSource, "_time_ms", read_wrong_once_after_a_seek)
        for k in frames_to_test()[::-1]:
            assert same(source.get(k), frames[k]), f"frame {k} differs"
        assert state["wrong readings"] > 0, "no time stamp was read after a seek, so nothing was injected"
        assert source.stats["from_zero"] > 0
        assert source.timestamps_s is not None


# ---------------------------------------------------------------------------------------------
# The look at the far end, the count on the way, and the cache


def test_the_far_end_of_the_table_is_looked_at_once_before_the_first_jump(disk_clip, sequential_frames, monkeypatch):
    # On a healthy file the look at the far end costs one seek, at the first jump, and none later.
    # It asks for the frame `SEEK_BACK` before the last one (115), as a jump to the last frame would.
    frames = sequential_frames["disk_clip"]
    with FrameSource(disk_clip.path) as source:
        asked = shift_seeks(source, 0, monkeypatch)  # every seek as it is, and the list of them
        for k in (20, 40):  # steps forward: no seek, and the far end is left alone
            assert same(source.get(k), frames[k])
        assert asked == []
        for k in (3, 100, 30, 110, 10, 90):  # six jumps (back, or more than 64 ahead); none asks for frame 115
            assert same(source.get(k), frames[k]), f"frame {k} differs"
        assert asked[0] == N - 1 - SEEK_BACK and asked.count(N - 1 - SEEK_BACK) == 1
        assert len(asked) >= 7 and source.stats["from_zero"] == 0


def test_a_seek_that_lands_after_the_last_frame_is_asked_again_earlier(clip_with_frames, monkeypatch):
    # A seek to one of the last frames lands 10 frames late here, which is after the end: no frame
    # comes. That is no reason to distrust the table: the seek is asked again for an earlier frame,
    # as when it lands after frame k, so the look at the far end succeeds and jumps stay fast.
    _, path, frames = clip_with_frames
    seek, late = FrameSource._seek, []

    def seek_late_near_the_end(self, frame):
        if frame >= N - 1 - SEEK_BACK:
            late.append(frame)
            frame += 10
        seek(self, frame)

    monkeypatch.setattr(FrameSource, "_seek", seek_late_near_the_end)
    with FrameSource(path) as source:
        for k in (100, 3, N - 1, 30, N - 2, 10):  # jumps, two of them to the last frames
            assert same(source.get(k), frames[k]), f"frame {k} differs"
        assert late, "no seek asked for one of the last frames, so nothing was injected"
        assert source.stats["seek"] > len(late) and source.stats["from_zero"] == 0


def test_frames_placed_with_a_table_found_wrong_later_are_not_served_again(clip_with_frames, monkeypatch):
    # The limit of the method (stated in frame_source.py): this table fits the decoder at both ends
    # of the file and at every frame compared on the way to frame 90, and still numbers frames 81
    # to 99 one too low. What the jump to frame 90 returns is therefore not asserted. What is: once
    # a count shows the table wrong (the steps from frame 70 to 85 pass frame 80), nothing that was
    # placed with it is served again, from the cache or otherwise.
    _, path, frames = clip_with_frames
    change_the_table(monkeypatch, without_one_frame)
    with FrameSource(path) as source:
        source.get(90)  # a jump, placed with the table
        assert same(source.get(70), frames[70])  # a jump to where the table is right
        assert source.timestamps_s is not None
        assert same(source.get(85), frames[85])
        assert source.timestamps_s is None, "the table was kept after the count showed it wrong"
        assert same(source.get(90), frames[90]), "a frame placed with the wrong table was served again"
        assert every_tested_frame_is_exact(source, frames)


def test_a_count_that_passes_over_wrong_entries_gives_the_table_up(clip_with_frames, monkeypatch):
    # The same table. Frames 70 and 105 are where it is right; the steps from 70 to 105 pass over
    # frames 80 to 99, whose times it has wrong. Every time read on the way is compared, not only
    # the last, so the table is given up there, before a jump into that stretch can be misplaced.
    _, path, frames = clip_with_frames
    change_the_table(monkeypatch, without_one_frame)
    with FrameSource(path) as source:
        assert same(source.get(70), frames[70])
        assert source.timestamps_s is not None
        assert same(source.get(105), frames[105])
        assert source.timestamps_s is None, "the table was kept although the count passed over wrong entries"
        assert same(source.get(90), frames[90])
        assert every_tested_frame_is_exact(source, frames)
