"""The flags table of a run folder (SPEC 9: "a flags table (track, frame, t, code) whose rows jump
to the frame"), for the GUI's review panel.

`flags_table` reads session.json and results.npz and gives one row per flag on a frame. The flags
are the ones the export writes: `derive.derive_track` and `qc.compute_flags` on the same records
with the session's calibration as it is now. Nothing here looks at a video or a model.

Units: frames are video frame numbers; t_s = frame / fps_true, in s. No Qt, no torch.
"""

from __future__ import annotations

from pathlib import Path

from outline_tracker.derive import derive_track
from outline_tracker.qc import compute_flags
from outline_tracker.results import ResultsStore
from outline_tracker.schema import FLAG_SEPARATOR, RESULTS_NPZ, SESSION_JSON
from outline_tracker.session import Session


def flags_table(run_folder) -> list[tuple[str, int, float, str]]:
    """Every flag of every tracked frame of a run folder, as rows (track_id, frame, t_s, code).

    `run_folder` holds session.json and results.npz. frame: the video frame number; t_s: its time
    frame / fps_true in s; code: one of `schema.FLAGS`. A frame with several flags has one row for
    each, in the order of SPEC 9. Rows are sorted by track (the ids as text, the order of the
    tracks in results.npz: A, A2, B), then by frame. A track of the session without results has no
    row; a folder without results.npz gives an empty list.

    Raises FileNotFoundError without a session.json, and ValueError when there are results but
    fps_true is not known yet or the session has no scale (`Session.world_frame`): JUMP is a speed
    in mm per s.
    """
    folder = Path(run_folder)
    session = Session.load(folder / SESSION_JSON)
    if not (folder / RESULTS_NPZ).is_file():
        return []
    store = ResultsStore.load(folder / RESULTS_NPZ)
    tracks = sorted((track for track in session.tracks if track.id in store.track_ids), key=lambda track: track.id)
    if not tracks:
        return []
    fps = session.time.fps_true
    if fps is None:
        raise ValueError("fps_true is not known yet: take it from the manifest or the stopwatch clip, or type it "
                         "in, before the flags can be listed.")
    world = session.world_frame()
    arrays = {track.id: store.arrays(track.id) for track in tracks}
    derived = {track.id: derive_track(arrays[track.id], track, world, fps, session.circle, session.processing)
               for track in tracks}
    cells = compute_flags(derived, arrays, world, session.processing)
    rows = []
    for track in tracks:
        one = derived[track.id]
        for frame, t_s, cell in zip(one.frame.tolist(), one.t_s.tolist(), cells[track.id], strict=True):
            rows += [(track.id, frame, t_s, code) for code in cell.split(FLAG_SEPARATOR) if code]
    return rows
