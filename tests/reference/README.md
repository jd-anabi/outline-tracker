# tests/reference: unmodified copies from the student template

Source: https://github.com/jd-anabi/shrimp-tracker-template at commit
`4ec8cd723c80a26cad67433b1a020bae8f68e272` (2026-10-04, "Calibration fix (two-point exports)").

These are the ten provided course files that SPEC.md §0 and CLAUDE.md allow us to copy.
They are byte-for-byte copies and must stay that way:

- `shrimp/` — `__init__.py`, `segment.py`, `_edgetam.py`, `video.py`, `convert.py`, `check_video.py`
- `template_tests/` — `conftest.py`, `test_segment.py`, `test_video.py`, `test_convert.py`

Rules:

- The `outline_tracker` package never imports anything from here.
- Tests that need last week's behavior as a baseline (SPEC §13.3) put `tests/reference` on
  `sys.path` and import `shrimp` from it.
- Do not edit these files. To check they are unchanged, compare with the template commit above:
  `git diff --no-index <template>/src/shrimp tests/reference/shrimp`.
