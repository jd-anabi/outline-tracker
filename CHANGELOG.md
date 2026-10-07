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
