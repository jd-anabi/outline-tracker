"""The guard at the start of a tracking job (SPEC 3.5, decision X8): is the frame the tracking
decoder delivers under a number the frame the user clicked on?

When the user clicks on a frame, the session stores a hash of that decoded frame and the tag of
the decoder that made it (`video.frame_hash`, `video.decoder_tag`). Before anything is tracked,
`check_start_frames` decodes every run's start frame as tracking will (`video.iter_rgb_frames`,
the definition of frame numbers) and compares. A different frame under the same number would put
every click on the wrong picture, so the job stops there.

A hash made by another decoder (the run folder came from another computer) cannot be compared:
decoders may differ in the last bit. For such clicks the guard makes the hash anew from the frame
as this computer decodes it (`FrameHashUpdate`), so that they are compared here from then on.

Frames are video frame numbers; frame sizes are in px. No Qt, no torch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from outline_tracker.session import Prompt, Session
from outline_tracker.tracking_plan import RunPlan
from outline_tracker.video import decoder_tag, frame_hash, iter_rgb_frames


class FrameHashMismatch(RuntimeError):
    """The tracking decoder delivers another frame than the one a prompt was clicked on (SPEC 3.5)."""


@dataclass(frozen=True)
class FrameHashUpdate:
    """The hash of a start frame, made anew on this computer for clicks whose hash another decoder
    made (decision X8).

    track_id, frame: the clicks it is for, those of that track on that video frame. frame_hash:
    `video.frame_hash` of the frame as tracking decodes it here. decoder: `video.decoder_tag()` here.
    """

    track_id: str
    frame: int
    frame_hash: str
    decoder: str

    def apply(self, session: Session) -> None:
        """Store the hash and its tag in `session`: in every prompt of the track on that frame that
        holds a hash made by another decoder. A prompt without a hash, with a hash without a tag,
        or with a hash made by this decoder (the user clicked again meanwhile) stays as it is; so
        does a session that no longer has the track."""
        for track in session.tracks:
            if track.id == self.track_id:
                for prompt in track.prompts:
                    if prompt.frame == self.frame and _made_elsewhere(prompt, self.decoder):
                        prompt.frame_hash, prompt.decoder = self.frame_hash, self.decoder


def _made_elsewhere(prompt: Prompt, here: str) -> bool:
    """Does the prompt hold a frame hash made by another decoder than `here` (a decoder tag)?"""
    return prompt.frame_hash is not None and prompt.decoder is not None and prompt.decoder != here


def check_start_frames(video_path, session: Session, plans: Sequence[RunPlan],
                       log: Callable[[str], None]) -> list[FrameHashUpdate]:
    """Check the start frame of every planned run before anything is tracked.

    Each start frame (a video frame number) is decoded once, in order, the way tracking decodes,
    and then discarded. Raises ValueError when the video has no such frame or its frames do not
    have the size the session holds (width x height, px), and `FrameHashMismatch` when a prompt on
    that frame, of a track of the run, stores another hash than the decoded frame has.

    A prompt without a hash is not compared. A hash without a decoder tag is compared as it is. A
    hash made by another decoder (its tag differs from `video.decoder_tag()` here: the folder came
    from another computer) cannot be compared; the frame's hash here takes its place (X8). Those
    are returned, one `FrameHashUpdate` per track and start frame, for the session's writer to
    store, and said once per other decoder through `log`. `session` itself is not changed.
    """
    here = decoder_tag()
    size = (session.video.width, session.video.height)
    tracks = {track.id: track for track in session.tracks}
    starting: dict[int, list[str]] = {}
    for plan in plans:
        starting.setdefault(plan.start_frame, []).extend(plan.track_ids)
    name = Path(video_path).name
    elsewhere: dict[str, list[str]] = {}  # another decoder's tag -> the tracks whose hashes it made
    made_anew: list[FrameHashUpdate] = []
    with closing(iter_rgb_frames(video_path, sorted(starting))) as frames:
        for frame, rgb in frames:
            found_size = (rgb.shape[1], rgb.shape[0])
            if found_size != size:
                raise ValueError(f"The frames of {name} are {found_size[0]} x {found_size[1]} px, but the session "
                                 f"was made with a video of {size[0]} x {size[1]} px. Is this the same video?")
            found = None
            for track_id in starting.pop(frame):
                tags = []  # the other decoders that made hashes for this track's clicks on this frame
                for prompt in tracks[track_id].prompts:
                    if prompt.frame != frame or prompt.frame_hash is None:
                        continue
                    found = frame_hash(rgb) if found is None else found
                    if _made_elsewhere(prompt, here):
                        tags.append(prompt.decoder)
                    elif prompt.frame_hash != found:
                        raise FrameHashMismatch(
                            f"Frame {frame} of {name}, as tracking reads it, is not the frame on which track "
                            f"{track_id} was clicked (stored {prompt.frame_hash[:19]}..., found {found[:19]}...). "
                            "The video file, or the way this computer decodes it, has changed since the clicks "
                            "were made. Nothing was tracked. Check that this is the same video, then click the "
                            "object again on that frame.")
                if tags:
                    made_anew.append(FrameHashUpdate(track_id, frame, found, here))
                    for tag in tags:
                        elsewhere.setdefault(tag, []).append(track_id)
    if starting:
        frame, track_ids = min(starting.items())
        raise ValueError(f"{name} has no frame {frame}, where tracking of {', '.join(track_ids)} starts: the video "
                         "ends before it. Nothing was tracked.")
    for tag, track_ids in elsewhere.items():
        log(f"The clicks of {', '.join(dict.fromkeys(track_ids))} were made with another decoder ({tag}; here: "
            f"{here}), so their frames could not be compared with the frames tracking reads. Their frame hashes are "
            "made anew with this computer's decoder and stored in the session when tracking starts.")
    return made_anew
