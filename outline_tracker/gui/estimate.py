"""How long tracking will take, and how the time is worded (SPEC 6.4, 6.5, 10.1; panel 7).

The estimate before a run is SPEC 6.4's: the time one frame takes on this computer, times the
frames, times the objects. One frame with n objects in one run takes (1 + 0.5 (n - 1)) times as
long as one frame with one object: the picture is encoded once for all of them (0.5 per extra
object is last week's measured value for EdgeTAM, `selftest.EXTRA_PER_SHRIMP`). A fine object has
a run of its own (SPEC 6.3), so it costs a whole frame each time. The time one frame takes comes
from what this computer did last: a run, or the outline made after a click.

Units: times are s; "s per frame" is s per tracked frame, and where a function says "of one
object", for a frame that holds one object. Frames are counts of tracked frames. No Qt here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

EXTRA_PER_OBJECT = 0.5  # of the time of a frame, for every object of a run beyond the first


def seconds_per_object_frame(seconds_per_frame: float, objects: int) -> float:
    """The s per frame of one object, from the s that a frame with `objects` objects took (a run's
    s per frame, or the s an outline took): seconds_per_frame / (1 + 0.5 (objects - 1))."""
    return float(seconds_per_frame) / (1.0 + EXTRA_PER_OBJECT * (max(int(objects), 1) - 1))


def estimate_seconds(seconds_per_frame: float, plans: Iterable) -> float:
    """The estimated time of a job in s: for each run of `plans` (`tracking_plan.RunPlan`), the s
    per frame of one object (`seconds_per_frame`) x the run's frames x (1 + 0.5 per extra object
    of the run)."""
    return sum(float(seconds_per_frame) * len(plan.frames) * (1.0 + EXTRA_PER_OBJECT * (len(plan.track_ids) - 1))
               for plan in plans)


def frames_and_objects(plans: Iterable) -> tuple[int, int]:
    """(frames, objects) of a job: the tracked frames of all its runs, as the progress bar counts
    them, and the objects of all its runs."""
    plans = list(plans)
    return sum(len(plan.frames) for plan in plans), sum(len(plan.track_ids) for plan in plans)


def last_run_seconds(runs: Sequence) -> float | None:
    """The s per frame of one object that the newest run of `runs` (the session's `RunRecord`s,
    oldest first) measured; None when no run tracked a frame."""
    for run in reversed(runs):
        if run.frames_done > 0 and run.seconds_per_frame > 0:
            return seconds_per_object_frame(run.seconds_per_frame, len(run.tracks))
    return None


def counted(count: int, word: str) -> str:
    """ "1 frame", "60 frames": a count with its word."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def about(seconds: float) -> str:
    """An estimated time in s, in words: "about 4 min" (whole minutes, at least 1), or
    "under 1 min" for less than half a minute."""
    return "under 1 min" if seconds < 30 else f"about {max(1, round(seconds / 60))} min"


def time_left(seconds: float) -> str:
    """The time left in s, in words: "2 min 30 s", or "45 s" under a minute."""
    whole = max(0, round(seconds))
    return f"{whole} s" if whole < 60 else f"{whole // 60} min {whole % 60} s"


def progress_text(run: int, runs: int, done: int, total: int, seconds_per_frame: float, eta_s: float) -> str:
    """The line under the progress bar: "Run 1 of 2 · frame 240 of 600 · 0.42 s/frame · time left
    (ETA) 2 min 30 s". run, runs: the run that is tracked and how many the job has; done, total:
    tracked frames and frames to track over all runs; seconds_per_frame: s per tracked frame so
    far; eta_s: s left at that rate."""
    return (f"Run {run} of {runs} · frame {done} of {total} · {seconds_per_frame:.2f} s/frame · "
            f"time left (ETA) {time_left(eta_s)}")
