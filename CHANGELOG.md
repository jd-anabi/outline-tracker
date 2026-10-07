# Changelog

What changed in each version of Outline Tracker, newest first. Versions are the tags of this
repository (`vX.Y.Z`).

## 0.1.0 (not released yet)

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
