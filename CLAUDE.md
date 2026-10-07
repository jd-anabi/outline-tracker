# outline-tracker: rules for the coding agent

- The spec is SPEC.md. The plan, with checkboxes and a "Questions for J" section, is docs/PLAN.md: update it as tasks finish.
- Deadline: students install Thursday Oct 8, 1 pm Pacific. Work in the order of SPEC §14: core, `from-tracker` and `probe` before the GUI; P0 before P1.
- You will often work unattended. Don't stop for routine steps (uv, pytest, git add/commit/push, editing files in this repo). Push after each task.
- Test first. Expected values come from geometry, analytic shapes or synthetic ground truth, never from running the code and copying its output. Never weaken, skip or delete a test to make it pass. If a test seems wrong, mark it xfail(strict=True, reason=...), add it under "Questions for J" in docs/PLAN.md, and continue with other work.
- Run `uv run pytest -m "not slow"` after every change and report the result. Slow tests (real model, short synthetic clips only) are fine to run without asking once the HF segmenter is ported. Never run the model on long real videos yourself.
- Conventions (SPEC §3): Tracker's pixel convention (pixel centers at +0.5), mm, the session's origin and axes, y up, t_s = frame / fps_true, the frame grid. Units go in column names.
- Ported code keeps its behavior. You may copy from jd-anabi/shrimp-tracker-template only these provided files: src/shrimp/__init__.py, segment.py, _edgetam.py, video.py, convert.py, check_video.py and tests/conftest.py, test_segment.py, test_video.py, test_convert.py. tests/reference/ holds unmodified copies for regression tests and is never imported by the package.
- Nothing outside outline_tracker/gui imports Qt. torch and transformers are imported only inside outline_tracker/segmenter, lazily, except that the GUI entry point imports torch before PySide6 (Windows DLL issue).
- Never load a whole video into memory. Never keep full-frame float arrays per object beyond the current frame.
- Ask (in PLAN.md) before adding a dependency not listed in SPEC §15.
- Never commit videos, results.npz or model weights. Never force-push or rewrite history.
- This repo will be public. Never put answer keys, student data, rosters or personal paths in it.
- Finish every task by saying what changed, how it was verified, and what is uncertain.
