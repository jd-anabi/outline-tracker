# outline-tracker: rules for the coding agent

- The plan is docs/ROADMAP.md: one workstream per session, in the order of its section 5. Tick its boxes as steps finish, write the owner's answers under its decisions, and add one line to its log at the end of a session. Its section 2 holds the rules of this phase in full; this file states them in short. SPEC.md is the specification of version 0.1.0; docs/PLAN.md is the history of the first build.
- You will often work unattended. Don't stop for routine steps (uv, pytest, git add/commit/push, editing files in this repo). Commit and push after each step.
- Use the installed skills and plugins that fit the work: a written plan, test first, a review after each step, verification before a box is ticked. Where a skill's default differs from this file or docs/ROADMAP.md, these win; say where.
- Test first. Expected values come from geometry, analytic shapes or synthetic ground truth, never from running the code and copying its output. The one exception is a frozen baseline, named where it is used: taken from a run that an independent check confirmed, with a header that says how it was made.
- Never weaken, skip or delete a test to make it pass. A test that a deliberate change makes obsolete is replaced in the same commit, and the commit message names the old test, the new one and the reason. If a test fails unexpectedly and seems wrong, mark it xfail(strict=True, reason=...), add it as a decision in section 3 of docs/ROADMAP.md, and continue with other work.
- Run `uv run pytest -m "not slow"` after every change and report the result. A step is done when the fast tests are green here and on every CI job: don't wait for CI, but look at `gh run list` before the next step and fix a red run first. Slow tests (real model, short synthetic clips only) are fine to run without asking. Never run the model on long real videos yourself.
- Never mix a move with a change of behavior. A commit that moves or renames files changes nothing else: the same tests, the same counts.
- Saved files of an older version stay safe. A change of session.json or results.npz comes with a reader for the old version; the first save over an older file keeps that file beside the new one; opening and closing an old run folder without a change writes nothing. Details: docs/ROADMAP.md section 2, rule 5.
- Version 0.1.0 is not disturbed. The tag v0.1.0 is never moved or deleted, and no build of this phase says 0.1.0. A fix for 0.1.0 goes on a branch made from the tag, is tagged v0.1.1 and comes to main by cherry-pick. Until the next release, every install line of README.md names a tag and its heading `## Quickstart` stays; before the first change to README.md, its first lines point to the README of the tag. Details: docs/ROADMAP.md section 2, rule 1.
- Performance is proved by counting, not by timing: a test counts calls, bytes or array sizes; wall times go into a benchmark report.
- Requirements live in one place. Work handed to helpers has one written brief, and no second wording elsewhere.
- Conventions (SPEC §3): Tracker's pixel convention (pixel centers at +0.5), mm, the session's origin and axes, y up, t_s = frame / fps_true, the frame grid. Units go in column names.
- Nothing outside outline_tracker/gui imports Qt. torch and transformers are imported only inside outline_tracker/segmenter, lazily, except that the GUI entry point imports torch before PySide6 (Windows DLL issue).
- Never load a whole video into memory. Never keep full-frame float arrays per object beyond the current frame.
- Ask the owner before adding a dependency that pyproject.toml does not list.
- Never commit videos, results.npz or model weights. Never force-push or rewrite history.
- This repo will be public. Never put personal paths or private data in it.
- Finish every task by saying what changed, how it was verified, and what is uncertain.
