"""The guard at the start of a job (SPEC 3.5, 13.2; decision X8): the frame a prompt was clicked on
must be the frame the tracking decoder delivers under that number, and every run's start frame must
exist and have the session's size. Nothing is tracked when it does not hold.

The stored hashes are made as the app makes them: `video.frame_hash` of the frame that
`FrameSource`, the display's reader, returns. A wrong frame is the true hash of another frame of the
same clip. A hash made by another decoder is replaced by this computer's (`decoded`: the frame read
in order from the start, as tracking reads it). Frames are video frame numbers.
"""

import copy
import re
from dataclasses import replace

import pytest
from tracking_helpers import Recorder, Watched, make_session, run, track

from outline_tracker import video
from outline_tracker.frame_source import FrameSource
from outline_tracker.results import ResultsStore
from outline_tracker.segmenter.fake import ExactFake
from outline_tracker.session import Prompt, Session, Track
from outline_tracker.tracking import FrameHashMismatch, FrameHashUpdate, Job, run_job

OTHER_DECODER = "opencv-4.10.0/win32/AMD64"  # the tag of a computer this test does not run on


def shown(clip, frame):
    """The hash of a frame as the display delivers it."""
    with FrameSource(clip.path) as source:
        return video.frame_hash(source.get(frame))


def decoded(clip, frame):
    """The hash of a frame as tracking reads it on this computer: decoded in order from the start."""
    ((_, rgb),) = video.iter_rgb_frames(clip.path, [frame])
    return video.frame_hash(rgb)


def stamps(session):
    """(frame hash, decoder tag) of the first prompt of every track, by track id."""
    return {a_track.id: (a_track.prompts[0].frame_hash, a_track.prompts[0].decoder) for a_track in session.tracks}


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


@pytest.mark.xfail(strict=True, reason="Asserts the opposite of decision X8 (docs/PLAN.md, section 4): a hash made "
                   "under another decoder tag is recomputed here and stored, not kept. Found in the review of A15; "
                   "the test below it, test_hashes_made_by_another_decoder_are_made_anew_here_and_logged, holds "
                   "what X8 asks. For J or the controller: delete this test, or amend X8.")
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


# ---------------------------------------------------------------------------------------------
# Decision X8: a hash made under another decoder tag cannot be compared; the hash of the frame as
# this computer decodes it is stored in its place, under this computer's tag, and that is logged.


def test_hashes_made_by_another_decoder_are_made_anew_here_and_logged(dish_clip, tmp_path):
    session = two_runs(dish_clip, tmp_path)
    for a_track in session.tracks:  # the folder came from another computer, whose decoder gave other bytes
        stamp(a_track.prompts[0], dish_clip, frame=2, decoder=OTHER_DECODER)
    status, seen = run(dish_clip, session, tmp_path, ExactFake(dish_clip))
    assert status == "complete"
    said = [line for line in seen.log if OTHER_DECODER in line]
    assert len(said) == 1 and "A, B, C" in said[0] and video.decoder_tag() in said[0]  # once, not per prompt
    # the session now holds each start frame's hash as tracking reads it here, under this computer's tag
    here = video.decoder_tag()
    new = {"A": (decoded(dish_clip, 0), here), "B": (decoded(dish_clip, 0), here), "C": (decoded(dish_clip, 60), here)}
    assert stamps(Session.load(tmp_path / "session.json")) == new
    assert stamps(session) == new  # the default writer keeps the caller's session and the file together


def test_after_that_the_prompts_are_compared_on_this_computer(dish_clip, disk_clip, tmp_path):
    session = make_session(dish_clip, tmp_path, [track(dish_clip, "B")])
    stamp(session.tracks[0].prompts[0], dish_clip, frame=2, decoder=OTHER_DECODER)
    assert run(dish_clip, session, tmp_path, ExactFake(dish_clip))[0] == "complete"

    # results.npz does not travel with the folder (SPEC 8.13), so B is tracked again from the saved session
    (tmp_path / "results.npz").unlink()
    status, seen = run(dish_clip, Session.load(tmp_path / "session.json"), tmp_path, ExactFake(dish_clip))
    assert status == "complete"
    assert not any("decoder" in line for line in seen.log)  # the hash was compared: there is nothing to say

    # and another picture under that frame number stops the job: here, another video of the same size
    (tmp_path / "results.npz").unlink()
    assert disk_clip.scene.size == dish_clip.scene.size
    segmenter = Watched(ExactFake(disk_clip))
    with pytest.raises(FrameHashMismatch, match=r"[Ff]rame 0\b.*\bB\b"):
        run(disk_clip, Session.load(tmp_path / "session.json"), tmp_path, segmenter)
    assert segmenter.made == 0 and segmenter.calls == []


def test_only_the_hashes_of_another_decoder_are_made_anew(dish_clip, tmp_path):
    session = two_runs(dish_clip, tmp_path)
    stamp(session.tracks[0].prompts[0], dish_clip, frame=2, decoder=OTHER_DECODER)  # A: from elsewhere
    session.tracks[1].prompts[0].decoder = None  # B: a hash without a tag, compared as it is (and right)
    before = stamps(session)
    status, seen = run(dish_clip, session, tmp_path, ExactFake(dish_clip))
    assert status == "complete"
    (said,) = [line for line in seen.log if OTHER_DECODER in line]
    assert re.search(r"\bA\b", said) and not re.search(r"\b[BC]\b", said)  # A alone
    saved = stamps(Session.load(tmp_path / "session.json"))
    assert saved == {"A": (decoded(dish_clip, 0), video.decoder_tag()), "B": before["B"], "C": before["C"]}


def test_the_new_hashes_reach_the_session_with_the_first_change_of_the_job(dish_clip, tmp_path):
    session = two_runs(dish_clip, tmp_path)
    stamp(session.tracks[0].prompts[0], dish_clip, frame=2, decoder=OTHER_DECODER)  # A, of the first run
    stamp(session.tracks[2].prompts[0], dish_clip, frame=62, decoder=OTHER_DECODER)  # C, of the second run
    before = copy.deepcopy(session).to_json()
    status, seen = run(dish_clip, session, tmp_path, ExactFake(dish_clip), record_session=True)
    assert status == "complete"
    assert session.to_json() == before  # the job worked on a copy: the session's one writer makes the change

    here = video.decoder_tag()
    first, *later = seen.changes
    assert sorted((new.track_id, new.frame, new.frame_hash, new.decoder) for new in first.frame_hashes) == [
        ("A", 0, decoded(dish_clip, 0), here), ("C", 60, decoded(dish_clip, 60), here)]
    assert len(later) == 3 and all(change.frame_hashes == () for change in later)
    mine = copy.deepcopy(session)
    for change in seen.changes:
        change.apply(mine)
    assert stamps(mine) == {"A": (decoded(dish_clip, 0), here), "B": stamps(session)["B"],
                            "C": (decoded(dish_clip, 60), here)}


def test_no_hash_is_stored_anew_by_a_job_that_starts_no_run(dish_clip, tmp_path):
    session = two_runs(dish_clip, tmp_path)
    for a_track in session.tracks:
        stamp(a_track.prompts[0], dish_clip, frame=2, decoder=OTHER_DECODER)
    before, recorder = stamps(session), Recorder()
    callbacks = replace(recorder.callbacks(record_session=True), should_cancel=lambda: True)
    status = run_job(Job(session, tmp_path, dish_clip.path, lambda: ExactFake(dish_clip)), callbacks)
    assert status == "cancelled"
    assert recorder.changes == [] and list(tmp_path.iterdir()) == [] and stamps(session) == before


def test_a_new_hash_replaces_only_another_decoders_hashes_on_that_frame_of_that_track():
    here = "opencv-9.9.9/here/arm64"

    def prompt(frame, frame_hash, decoder):
        return Prompt(frame=frame, frame_hash=frame_hash, decoder=decoder, points_px=[[10.5, 20.5]], labels=[1])

    a = Track(id="A", prompts=[prompt(0, "sha256:there", OTHER_DECODER), prompt(0, None, None),
                               prompt(0, "sha256:untagged", None), prompt(0, "sha256:mine", here),
                               prompt(60, "sha256:there60", OTHER_DECODER)])
    b = Track(id="B", prompts=[prompt(0, "sha256:there", OTHER_DECODER)])
    session = Session(tracks=[a, b])
    FrameHashUpdate(track_id="A", frame=0, frame_hash="sha256:here", decoder=here).apply(session)
    # of A's clicks on frame 0, only the hash of the other decoder is replaced: one without a hash stays without,
    # an untagged one and one made here (the user clicked again meanwhile) stay as they are
    assert [(p.frame_hash, p.decoder) for p in a.prompts] == [
        ("sha256:here", here), (None, None), ("sha256:untagged", None), ("sha256:mine", here),
        ("sha256:there60", OTHER_DECODER)]
    assert stamps(session)["B"] == ("sha256:there", OTHER_DECODER)
    FrameHashUpdate(track_id="Z", frame=0, frame_hash="sha256:here", decoder=here).apply(session)  # removed meanwhile
    assert [track_.id for track_ in session.tracks] == ["A", "B"]


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
