"""The guard at the start of a job (SPEC 3.5, 13.2; decision X8): the frame a prompt was clicked on
must be the frame the tracking decoder delivers under that number, and every run's start frame must
exist and have the session's size. Nothing is tracked when it does not hold.

The stored hashes are made as the app makes them: `video.frame_hash` of the frame that
`FrameSource`, the display's reader, returns. A wrong frame is the true hash of another frame of the
same clip. Frames are video frame numbers.
"""

import pytest
from tracking_helpers import Watched, make_session, run, track

from outline_tracker import video
from outline_tracker.frame_source import FrameSource
from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.session import Session
from outline_tracker.tracking import FrameHashMismatch

OTHER_DECODER = "opencv-4.10.0/win32/AMD64"  # the tag of a computer this test does not run on


def shown(clip, frame):
    """The hash of a frame as the display delivers it."""
    with FrameSource(clip.path) as source:
        return video.frame_hash(source.get(frame))


def stamp(prompt, clip, frame=None, decoder="here"):
    """Store in a prompt the hash of `frame` (its own frame if None) and the decoder it was made
    with (this computer's, unless another tag or None is given)."""
    prompt.frame_hash = shown(clip, prompt.frame if frame is None else frame)
    prompt.decoder = video.decoder_tag() if decoder == "here" else decoder


def two_runs(clip, tmp_path):
    """A and B clicked on frame 0, C on frame 60; every prompt stamped with its own frame's hash."""
    session = make_session(clip, tmp_path, [track(clip, "A"), track(clip, "B"), track(clip, "C", frame=60)])
    for a_track in session.tracks:
        stamp(a_track.prompts[0], clip)
    return session


def assert_nothing_happened(segmenter, folder):
    assert segmenter.made == 0 and segmenter.calls == []  # no model was loaded, no frame was tracked
    assert list(folder.iterdir()) == []


def test_prompts_made_on_the_displayed_frames_pass(dish_clip, tmp_path):
    assert video.decoder_tag() != OTHER_DECODER
    assert len({shown(dish_clip, frame) for frame in (0, 2, 60, 62)}) == 4  # the frames used below all differ
    session = two_runs(dish_clip, tmp_path)
    status, _ = run(dish_clip, session, tmp_path, ExactFake(dish_clip))
    assert status == "complete"
    assert ResultsStore.load(tmp_path / "results.npz").track_ids == ["A", "B", "C"]


def test_a_prompt_made_on_another_frame_aborts_before_any_tracking(dish_clip, tmp_path):
    session = two_runs(dish_clip, tmp_path)
    stamp(session.tracks[0].prompts[0], dish_clip, frame=2)  # A was clicked on what is frame 2 here
    segmenter = Watched(ExactFake(dish_clip))
    with pytest.raises(FrameHashMismatch, match=r"[Ff]rame 0\b.*\bA\b"):
        run(dish_clip, session, tmp_path, segmenter)
    assert_nothing_happened(segmenter, tmp_path)


def test_a_wrong_start_frame_of_a_later_run_aborts_before_the_first_run(dish_clip, tmp_path):
    session = two_runs(dish_clip, tmp_path)
    stamp(session.tracks[2].prompts[0], dish_clip, frame=62)  # C, which starts on frame 60
    segmenter = Watched(ExactFake(dish_clip))
    with pytest.raises(FrameHashMismatch, match=r"[Ff]rame 60\b.*\bC\b"):
        run(dish_clip, session, tmp_path, segmenter)
    assert_nothing_happened(segmenter, tmp_path)


def test_a_hash_without_a_decoder_tag_is_compared_as_the_spec_says(dish_clip, tmp_path):
    session = two_runs(dish_clip, tmp_path)
    stamp(session.tracks[1].prompts[0], dish_clip, frame=2, decoder=None)
    segmenter = Watched(ExactFake(dish_clip))
    with pytest.raises(FrameHashMismatch, match=r"\bB\b"):
        run(dish_clip, session, tmp_path, segmenter)
    assert_nothing_happened(segmenter, tmp_path)


def test_hashes_made_by_another_decoder_are_not_compared_but_logged(dish_clip, tmp_path):
    session = two_runs(dish_clip, tmp_path)
    for a_track in session.tracks:  # the folder came from another computer, whose decoder gave other bytes
        stamp(a_track.prompts[0], dish_clip, frame=2, decoder=OTHER_DECODER)
    stored = session.tracks[0].prompts[0].frame_hash
    status, seen = run(dish_clip, session, tmp_path, ExactFake(dish_clip))
    assert status == "complete"
    assert sum(OTHER_DECODER in line for line in seen.log) == 1  # said once, not per prompt
    # nothing is computed anew here: the session keeps the hash and the tag it came with
    kept = Session.load(tmp_path / "session.json").tracks[0].prompts[0]
    assert (kept.frame_hash, kept.decoder) == (stored, OTHER_DECODER)


def test_what_can_be_compared_still_is_when_other_prompts_came_from_another_decoder(dish_clip, tmp_path):
    session = two_runs(dish_clip, tmp_path)
    stamp(session.tracks[0].prompts[0], dish_clip, frame=2, decoder=OTHER_DECODER)  # A: cannot be compared
    stamp(session.tracks[1].prompts[0], dish_clip, frame=2)  # B: made here, and wrong
    segmenter = Watched(ExactFake(dish_clip))
    with pytest.raises(FrameHashMismatch, match=r"\bB\b"):
        run(dish_clip, session, tmp_path, segmenter)
    assert_nothing_happened(segmenter, tmp_path)


def test_prompts_without_a_hash_are_tracked(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "B")])
    assert session.tracks[0].prompts[0].frame_hash is None  # as from-tracker makes them: nobody saw a frame
    status, _ = run(dish_clip, session, tmp_path, ExactFake(dish_clip))
    assert status == "complete"
    assert ResultsStore.load(tmp_path / "results.npz").arrays("B").frames.tolist() == list(range(0, 120, 2))


def test_a_start_frame_the_video_does_not_have_is_an_error_before_any_tracking(dish_clip, tmp_path):
    # the clip is said to reach frame 400, and C to start on frame 300; the video ends with frame 119
    tracks = [track(dish_clip, "A"), track(dish_clip, "C", frame=300)]
    session = make_session(dish_clip, tmp_path, tracks, end=400)
    segmenter = Watched(ExactFake(dish_clip))
    with pytest.raises(ValueError, match=r"(?s)\bC\b.*\b300\b|\b300\b.*\bC\b"):
        run(dish_clip, session, tmp_path, segmenter)
    assert_nothing_happened(segmenter, tmp_path)


def test_a_video_of_another_frame_size_than_the_session_says_is_refused(dish_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "B")])
    session.video.width, session.video.height = 640, 480  # the clip is 320 x 240 px
    segmenter = Watched(ExactFake(dish_clip))
    with pytest.raises(ValueError, match=r"320 x 240.*640 x 480|640 x 480.*320 x 240"):
        run(dish_clip, session, tmp_path, segmenter)
    assert_nothing_happened(segmenter, tmp_path)
