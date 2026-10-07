"""The guard at the start of a tracking job (SPEC 3.5, decision X8): is the frame the tracking
decoder delivers under a number the frame the user clicked on?

When the user clicks on a frame, the session stores a hash of that decoded frame and the tag of
the decoder that made it (`video.frame_hash`, `video.decoder_tag`). Before anything is tracked,
`check_start_frames` decodes every run's start frame as tracking will (`video.iter_rgb_frames`,
the definition of frame numbers) and compares. A different frame under the same number would put
every click on the wrong picture, so the job stops there.

Frames are video frame numbers; frame sizes are in px. No Qt, no torch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path

from outline_tracker.session import Session
from outline_tracker.tracking_plan import RunPlan
from outline_tracker.video import decoder_tag, frame_hash, iter_rgb_frames


class FrameHashMismatch(RuntimeError):
    """The tracking decoder delivers another frame than the one a prompt was clicked on (SPEC 3.5)."""


def check_start_frames(video_path, session: Session, plans: Sequence[RunPlan], log: Callable[[str], None]) -> None:
    """Check the start frame of every planned run before anything is tracked.

    Each start frame (a video frame number) is decoded once, in order, the way tracking decodes,
    and then discarded. Raises ValueError when the video has no such frame or its frames do not
    have the size the session holds (width x height, px), and `FrameHashMismatch` when a prompt on
    that frame, of a track of the run, stores another hash than the decoded frame has.

    A prompt without a hash is not compared. A hash made by another decoder (its tag differs from
    `video.decoder_tag()` here: the folder came from another computer) is not compared either,
    since decoders may differ in the last bit; that is said once through `log`, and nothing is
    stored anew. A hash without a tag is compared as it is.
    """
    here = decoder_tag()
    size = (session.video.width, session.video.height)
    tracks = {track.id: track for track in session.tracks}
    starting: dict[int, list[str]] = {}
    for plan in plans:
        starting.setdefault(plan.start_frame, []).extend(plan.track_ids)
    name = Path(video_path).name
    elsewhere: dict[str, list[str]] = {}  # another decoder's tag -> the tracks whose hashes it made
    with closing(iter_rgb_frames(video_path, sorted(starting))) as frames:
        for frame, rgb in frames:
            found_size = (rgb.shape[1], rgb.shape[0])
            if found_size != size:
                raise ValueError(f"The frames of {name} are {found_size[0]} x {found_size[1]} px, but the session "
                                 f"was made with a video of {size[0]} x {size[1]} px. Is this the same video?")
            found = None
            for track_id in starting.pop(frame):
                for prompt in tracks[track_id].prompts:
                    if prompt.frame != frame or prompt.frame_hash is None:
                        continue
                    if prompt.decoder is not None and prompt.decoder != here:
                        elsewhere.setdefault(prompt.decoder, []).append(track_id)
                        continue
                    found = frame_hash(rgb) if found is None else found
                    if prompt.frame_hash != found:
                        raise FrameHashMismatch(
                            f"Frame {frame} of {name}, as tracking reads it, is not the frame on which track "
                            f"{track_id} was clicked (stored {prompt.frame_hash[:19]}..., found {found[:19]}...). "
                            "The video file, or the way this computer decodes it, has changed since the clicks "
                            "were made. Nothing was tracked. Check that this is the same video, then click the "
                            "object again on that frame.")
    if starting:
        frame, track_ids = min(starting.items())
        raise ValueError(f"{name} has no frame {frame}, where tracking of {', '.join(track_ids)} starts: the video "
                         "ends before it. Nothing was tracked.")
    for tag, track_ids in elsewhere.items():
        log(f"The clicks of {', '.join(dict.fromkeys(track_ids))} were made with another decoder ({tag}; here: "
            f"{here}), so their frames could not be compared with the frames tracking reads.")
