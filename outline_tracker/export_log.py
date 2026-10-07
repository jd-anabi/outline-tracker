"""The block that an export appends to run.log (SPEC 8.9), and the appending itself.

One block per export, in this order: software and machine, device, model, video, time, calibration,
runs, corrections, the QC summary, the files written. Everything is read from the session, the
stored records and the installed packages' metadata: the device, the model id and the hash of the
weights are what tracking wrote into the session's run records, so writing the log loads no model.
What a run's model saw is read from the records themselves (their grid cell is 1/256 of the
longer side of the model's image), since a setting may have changed after the run.

The text is plain ASCII apart from names the user gave (um, mm^2, deg), UTF-8, LF line ends.
run.log is appended to, never rewritten: the earlier text is kept byte for byte and the file is
replaced in one step (`fileio.atomic_write`).

Units: px in Tracker's image coordinates (pixel centers at +0.5), mm, um, s, degrees, as each line
says. Frames are video frame numbers. No Qt, no torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from outline_tracker import fileio, provenance
from outline_tracker.geometry import stick_scale, tape_check
from outline_tracker.measure import GRID_CELLS
from outline_tracker.results import TrackArrays
from outline_tracker.session import RunRecord, Session


def export_block(session: Session, arrays_by_track: Mapping[str, TrackArrays], qc_lines: Sequence[str],
                 outputs: Sequence[str], when: datetime) -> list[str]:
    """The lines of one export's block of run.log (no line ends).

    session: the session as exported. arrays_by_track: the exported tracks' records (image px),
    from which each run's crop is read. qc_lines: the QC summary (`qc.summary_lines`). outputs: one
    line per file written or left out, e.g. "positions.csv: 12,345 bytes". when: the time of the
    export, with its offset from UTC. Lengths are in px, mm and um as each line says.
    """
    version = provenance.tool_version()
    libraries = "; ".join(f"{name} {found}" for name, found in provenance.library_versions().items())
    devices = list(dict.fromkeys(run.device for run in session.runs if run.device))
    lines = [
        f"==== export, {when:%Y-%m-%d %H:%M:%S %z}, {version} ====",
        f"software: {version}",
        f"machine: {provenance.machine_text()}",
        f"libraries: {libraries}",
        f"device: {', '.join(devices) or 'not recorded'}",
        _model_line(session),
        _video_line(session),
        _time_line(session),
        "calibration:",
        *(f"  {line}" for line in _calibration_lines(session)),
        "runs:",
        *(f"  run {number}: {_run_text(session, run, arrays_by_track)}"
          for number, run in enumerate(session.runs, start=1)),
        *([] if session.runs else ["  none"]),
        "results: complete" if session.complete else
        "results: partial (the session says the tracking is not complete; the files hold the frames tracked so far)",
    ]
    if session.corrections:
        lines.append("corrections:")
        lines += [f"  {fix.time}: {fix.action}, track{'' if len(fix.tracks) == 1 else 's'} {', '.join(fix.tracks)}, "
                  f"frame {fix.frame}" for fix in session.corrections]
    else:
        lines.append("corrections: none")
    lines += ["qc summary:", *(f"  {line}" for line in qc_lines or ["no tracks"])]
    lines += ["outputs:", *(f"  {line}" for line in outputs)]
    return lines


def append_block(path: Path, lines: Sequence[str]) -> Path:
    """Add one block to run.log at `path` (UTF-8, LF line ends), after a blank line when the log has
    text already. What is there is kept as it is; the file is replaced in one step. Returns the path
    written: `path`, or `<stem>.new<suffix>` next to it when `path` stayed locked by another program.

    `probe` adds its entry to the same file (`cli_probe`); tests/test_export_log.py checks that the
    two commands leave the same bytes."""
    earlier = path.read_bytes() if path.is_file() else b""
    if earlier and not earlier.endswith(b"\n"):
        earlier += b"\n"
    block = ("\n".join(lines) + "\n").encode("utf-8")
    return fileio.atomic_write(path, lambda tmp: tmp.write_bytes(earlier + (b"\n" if earlier else b"") + block))


def _model_line(session: Session) -> str:
    """The model's name, and each model id and weights hash (SHA-256) the run records hold."""
    known = list(dict.fromkeys((run.model_id, run.weights_sha256) for run in session.runs
                               if run.model_id or run.weights_sha256))
    facts = [f"model id {model_id or 'not recorded'}; weights sha256 {weights or 'not recorded'}"
             for model_id, weights in known]
    return f"model: {session.processing.model}; " + ("; ".join(facts) or "model id and weights not recorded")


def _video_line(session: Session) -> str:
    """The video as the session identifies it: its path from the run folder (the absolute path
    when there is no relative one), size in bytes, hash, and what the file says about itself."""
    video = session.video
    path = video.relpath or video.abspath
    if not path:
        return "video: none named in the session"
    return (f"video: {path}; {video.size:,} bytes; sha256 of the first 64 MiB {video.sha256_first_64mib}; the file "
            f"says {float(video.fps_container):.2f} fps, {video.n_frames} frames; {video.width} x {video.height} px")


def _time_line(session: Session) -> str:
    time = session.time
    manifest = f", manifest {time.manifest_path}" if time.manifest_path else ""
    return f"time: fps_true = {time.fps_true!r} frames per s (source: {time.source}{manifest})"


def _calibration_lines(session: Session) -> list[str]:
    """Scale, stick, tape check, circle and axes; lengths in px of the image, mm and um."""
    cal = session.calibration
    k = session.world_frame().k_mm_per_px
    scale = f"scale: k = {k:.7f} mm per px ({1000 * k:.3f} um per px)"
    if cal.stick is not None:
        (u1, v1), (u2, v2) = cal.stick["p1_px"], cal.stick["p2_px"]
        stick = stick_scale(cal.stick["p1_px"], cal.stick["p2_px"], cal.stick["length_mm"], cal.click_sigma_px)
        scale += f" +- {100 * stick.rel_uncertainty:.2f} % (click precision {cal.click_sigma_px:g} px)"
        stick_line = (f"stick: ({u1:.3f}, {v1:.3f}) to ({u2:.3f}, {v2:.3f}) px ({stick.length_px:.1f} px), "
                      f"{cal.stick['length_mm']:g} mm")
    else:
        fit = cal.tracker_fit or {}
        scale += (f", fitted to the Tracker export (rms {1000 * float(fit.get('rms_mm', float('nan'))):.4f} um, "
                  f"{fit.get('n_points', '?')} points); no click uncertainty")
        stick_line = "stick: none"
    lines = [scale, stick_line, _tape_text(cal.check, k)]
    circle = session.circle
    if circle is None:
        lines.append("circle: none")
    else:
        (cu, cv), radius = circle.center_px, float(circle.radius_px)
        text = (f"circle: center ({cu:.3f}, {cv:.3f}) px, R = {radius:.3f} px, rms {float(circle.rms_px):.3f} px; "
                f"2R = {2 * radius * k:.4f} mm")
        if circle.dish_mm:
            off = 100 * (2 * radius * k - circle.dish_mm) / circle.dish_mm
            text += f", dish_mm = {circle.dish_mm:g} (2R is {off:+.2f} % off)"
        else:
            text += ", dish_mm not given"
        lines.append(text)
    u0, v0 = session.axes.origin_px
    lines.append(f"axes: origin ({u0:.3f}, {v0:.3f}) px, alpha = {session.axes.angle_deg:g} deg")
    return lines


def _tape_text(check, k: float) -> str:
    """The tape check at the scale `k` (mm per px): true and measured distance in mm, error in %."""
    if check is None:
        return "tape check: none"
    try:
        result = tape_check(check["p1_px"], check["p2_px"], check["true_mm"], k)
    except (KeyError, TypeError, ValueError) as err:
        return f"tape check: not usable ({err})"
    verdict = "" if result.ok else " (more than 1 %: check the scale)"
    return (f"tape check: true {float(check['true_mm']):g} mm, measured {result.measured_mm:.4f} mm, "
            f"error {100 * result.rel_error:+.2f} %{verdict}")


def _run_text(session: Session, run: RunRecord, arrays_by_track: Mapping[str, TrackArrays]) -> str:
    """One run: tracks, mode, start frame, step, frames, what the model saw, and the time it took."""
    total = run.frames_done * run.seconds_per_frame
    parts = [f"tracks {', '.join(run.tracks)}", run.mode, f"start frame {run.start_frame}",
             f"step {session.clip.step}", f"{run.frames_done} frames", _crop_text(session, run, arrays_by_track),
             f"{run.seconds_per_frame:.3f} s per frame, {total:.1f} s in all"]
    if run.device:
        parts.append(f"device {run.device}")
    if run.finished is None:
        parts.append("not finished")
    return "; ".join(parts)


def _crop_text(session: Session, run: RunRecord, arrays_by_track: Mapping[str, TrackArrays]) -> str:
    """What the model was shown in a run, from the grid cell stored with its start frame's record:
    the longer side of that image in px is 256 cells."""
    side = None
    for track_id in run.tracks:
        arrays = arrays_by_track.get(track_id)
        rows = [] if arrays is None else (arrays.frames == run.start_frame).nonzero()[0]
        if len(rows) and arrays.mode[rows[0]] == run.mode:
            side = round(float(arrays.cell_px[rows[0]]) * GRID_CELLS)
            break
    if side is None:
        return "crop not recorded (the run's start frame is no longer in the results)"
    if run.mode == "fine":
        return f"fine crop, W = {side} px"
    width, height = session.video.width, session.video.height
    if side == max(width, height):
        return f"whole frame ({width} x {height} px)"
    return f"dish crop, {side} px"
