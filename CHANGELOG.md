# Changelog

What changed in each version of Outline Tracker, newest first. Versions are the tags of this
repository (`vX.Y.Z`).

## Unreleased

Development version `0.2.0.dev0`. The files it writes say so.

### Changed

- The size check of a shape is now of the largest connected piece of the mask, not of the whole
  mask. Three columns of `shapes.csv` changed: `px_along_major`, `cells_along_major` and
  `shape_ok`, and with them the flag `LOWRES`. The formula in words: take the pixels of the
  largest piece, then the larger eigenvalue lambda1 of the covariance of their centers;
  `px_along_major` is 4 sqrt(lambda1) in camera pixels, `cells_along_major` is that length
  divided by the size of one model grid cell, and `shape_ok` is 1 if the smaller of the two
  reaches 20. Before, two stray pixels far from a body 15 px long made the whole mask read
  135 px, and the frame lost its `LOWRES` flag. The position, the area, `major_mm`, `minor_mm`
  and `eccentricity` are of the whole mask, as before. A run folder made by 0.1.0 shows the new
  values at its next export on the frames whose mask has more than one piece (`n_components`
  above 1); every other row is unchanged. No file format changed: nothing is tracked again.

## 0.1.0 (2026-10-07)

### Added

- Project skeleton: the `outline-tracker` command with `--version`, dependencies pinned to tested
  versions, and automatic tests on Ubuntu and Windows.
- `outline-tracker convert VIDEO...` and `outline-tracker check VIDEO...`: last week's
  `shrimp.convert` and `shrimp.check_video`, with the same messages. A problem with a file is now
  one `ERROR:` line instead of a traceback, and the next file is still processed.
- The EdgeTAM / SAM 2.1 backend, ported from last week's script: with the same click it gives the
  same positions (checked with the real model). New: several clicks per object, negative clicks,
  a preview on one frame, and a switch to the processor if the Apple GPU fails in mid-run.
- Exact frame access for the display: a jump to any frame shows exactly the frame the tracker
  sees, also in videos whose timestamps have gaps. `outline-tracker check VIDEO --seek` tests this
  on a file.
- `docs/OUTPUTS.md`: every output file, column, unit and flag, generated from the same tables the
  program writes its files from. It is the contract for analysis code.
- `outline-tracker probe VIDEO --rect NAME:u0,v0,u1,v1` (or `probe SESSION.json`): the mean red,
  green, blue and gray inside named boxes for every frame, written to `probes.csv`. No model.
- `outline-tracker from-tracker VIDEO EXPORT`: last week's way to start (calibration and one
  click per animal from a Tracker export), this week's files: `positions.csv`, `shapes.csv`,
  the Tracker-format folder that last week's loaders read, `overlay.mp4`, `run.log` and the
  rest, in one run folder next to the video. `--fine A,B` tracks the named animals in a
  close-up crop for their shapes.
- `outline-tracker export SESSION.json [--overlay]`: writes every output file again from the
  saved session and results, for example after a corrected scale or frame rate. No tracking,
  no model.
- `outline-tracker --version` and every `run.log` entry name the commit the tool was built from.
- `outline-tracker selftest`: last week's check of the installation, through this week's
  pipeline: it tracks a made-up shrimp, says OK or PROBLEM, and estimates the time a run takes
  on this computer.
- README: how to install, how to check the installation, and what to do if the app does not
  open (last week's workflow with this week's files).
- `outline-tracker` (no arguments) and `outline-tracker gui`: the window opens, so far empty,
  with the nine numbered panels. The panels are filled in next.
- The window opens a video (Open video, or `outline-tracker gui VIDEO`): the picture with zoom,
  pan, Fit and 1:1, and under it the frame slider, the step buttons and the frame box. The
  frame shown is exactly the frame the tracker reads.
- Calibration in the window (panels 3 and 4): click the two ends of a known length for the
  scale, check it with a second length, click points on the dish wall for the circle, and
  place the axes. The numbers appear as you click.
- Objects in the window (panel 6): add an object, click on the animal, and the model's outline
  appears within a second or two, before any tracking. Right click (or Alt-click) marks what
  does not belong; Head marks the front end.
- Panels 1 and 2 of the window: your name, the video's facts and warnings, the clip's start,
  end and step, and fps_true from the manifest or typed. Everything you set is saved in the
  run folder and is back after a restart (Open session, or `outline-tracker gui SESSION.json`).
- Track in the window (panel 7): an estimate of the time, then progress, seconds per frame and
  the time left; Cancel keeps what was tracked. The tracked outlines are drawn on the video as
  you step through it.
- Around the panels: the cursor's position in px and mm and the gray value in the status bar,
  File and Help menus, About, play and pause with Space, a Stopwatch dialog for fps_true, and
  the keys 1 to 9 to select an object.
- Export in the window (panel 9, File > Export): Export all writes the CSV files, the
  overlay video, the log and README.txt to the run folder and lists them; Open folder shows
  them.
- Review and fix in the window (panel 8): a table of the frames where the tracking may be
  wrong; a click jumps there. Re-track from here, End track here and Continue as new track
  repair a track without starting over.
- Safer and more complete: the panels that a run depends on are locked while it runs; Remove
  asks before it deletes tracked frames; the model and the device can be chosen in panel 7;
  a Fill switch shows the masks. The README has a ten-step Quickstart.
- Where the panels meet: Export all works while the model is still loading or could not be
  loaded; an export locks the same controls as a tracking run; the model cannot be changed
  once there are results; panel 9 says when the exported files are older than the session.
- `docs/ROADMAP.md`: the plan for the next phase (a general, open-source tool), with the
  facts it starts from in `docs/roadmap/`.
