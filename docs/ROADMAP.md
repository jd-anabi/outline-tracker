# Outline Tracker: roadmap for the next phase

Written 2026-10-07. This file is the hand-over from the two-day build of version 0.1.0 to the
sessions that come after it. It is for the owner and for a coding agent that has never seen this
project. The inventories behind it were taken at commit `f73d29f` of `main`; the numbers in this
file were counted again after the last follow-up of the first build was merged (`f69afb7`).

How to use it:

- One workstream (section 4) per session. Start with the prompt in section 6.
- Each workstream has an inventory in `docs/roadmap/`: the facts it starts from, with file and
  line pointers. Its pointers and numbers are true for commit `f73d29f`; after a merge, a move
  or a rename, count again and search for the name.
- The inventories give facts. Where an inventory proposes its own order of steps or numbers its
  own decisions, sections 3 to 5 of this file replace them.
- Tick the boxes as steps finish, write the owner's answers under the decisions of section 3, and
  add one line to the log in section 7 at the end of every session. This file is the plan now;
  `docs/PLAN.md` is the history of the first build.
- The session that closes a workstream deletes that workstream's inventory from the tree. It
  stays in the git history.

## 1. Where the project stands

What exists (version 0.1.0):

- A desktop app (PySide6, pyqtgraph) with nine numbered panels: video and clip, frame rate,
  calibration, boundary circle and axes, probes (a hint line only), objects with click prompts
  and a preview of the outline, tracking in the background, review with quality flags and three
  corrections, export.
- Commands without a window: `from-tracker`, `export`, `probe`, `selftest`, `convert`, `check`,
  and hidden `synth` and `compare-tracks`.
- One tracking method: a video segmentation model (EdgeTAM; SAM 2.1 tiny is offered but was never
  run) that returns a mask per object per frame. Positions, shapes and outlines are measured from
  the masks and stored in pixels; the files in mm are derived at export.

What it rests on:

- About 2,700 fast tests (stand-in models, synthetic clips with ground truth), run on Linux and
  Windows in CI; 34 slow tests with the real model, run by hand on one Apple laptop.
- `docs/VALIDATION.md`: measured results. The real model through the whole window, on synthetic
  clips, is section 6.
- On 2026-10-07 the owner installed the tool on a Mac the way a user does, ran the selftest and
  went through every panel by hand, and reported that it works. This was said in conversation;
  `docs/VALIDATION.md` does not record it yet (a box of W0).

What nobody has done yet:

- Used the window by hand on Windows, or run the real model on Windows or on a CUDA machine.
- Tracked a long real video.

Known weak point, measured: in coarse mode the model loses small objects (a 14.5 px body was lost
on 15 of 60 frames). In fine mode the same object was found on every frame within 0.4 px
(`docs/VALIDATION.md` 6.3).

Where things are written down:

| document | what it is |
|---|---|
| `README.md` | install, Quickstart, fallback commands; written for the first class of users |
| `docs/DEVELOPER.md` | module map, tests, rules, how to add a panel or a model |
| `docs/OUTPUTS.md` | every output file and column; generated from `outline_tracker/schema.py` |
| `docs/VALIDATION.md` | what was measured, and how |
| `SPEC.md` | the specification of version 0.1.0, written for one lab class |
| `docs/PLAN.md` | the build plan of 0.1.0: decisions X1 to X22 (section 4), open points ("Raised during the work") |
| `CLAUDE.md` | rules for coding agents; several are about the first build and change in this phase (section 2) |
| `docs/design/gui_design.md` | the design note of the window: spacing, colors, wording rules |
| `docs/roadmap/*.md` | the seven inventories behind this roadmap |

## 2. Ground rules for this phase

Until `CLAUDE.md` and `docs/DEVELOPER.md` are rewritten (W1 step 1), this section wins wherever
they, or an inventory, say otherwise.

1. **Version 0.1.0 is not disturbed.**
   - It is tagged `v0.1.0` at the end of W0. No tag exists yet. No other workstream starts
     before the tag exists, and the tag is never moved or deleted.
   - The first commit on `main` after the tag sets the version to `0.2.0.dev0`, so that no file
     written by a build of this phase says 0.1.0.
   - The README on `main` is the page that 0.1.0 opens from Help > Quickstart. Before the first
     change to it in this phase, its first lines point to the README of the tag; a heading
     `## Quickstart` stays; no install line on `main` is without a tag until the next release.
   - A fix that the class needs is committed on a branch made from the tag, tagged `v0.1.1`, and
     brought to `main` by cherry-pick.
2. **What carries over from `CLAUDE.md`:** every rule that rule 3 below does not name. Among
   them: tests first; expected values come from geometry, analytic shapes or synthetic ground
   truth, never from the code's own output; the fast tests run after every change; the
   conventions (pixel centers at +0.5, mm, y up, t = frame / fps_true, the frame grid, units in
   column names); nothing outside `outline_tracker/gui` imports Qt; torch and the model libraries
   are imported lazily and only where the model lives, and the window's entry point imports torch
   before PySide6 (until W5 step 7 changes that with a test on Windows); a video is never read
   whole into memory and no full-frame float arrays are kept per object; no videos, results files
   or model weights in git; no force-push and no rewritten history; no personal paths or private
   data in the repository; ask before adding a dependency.
   - New in this phase: a step is done when the fast tests are green here and on every CI job.
     A push that changes only `.md` files or `docs/` starts no CI run today although fast tests
     read documents; until W1 step 0 changes that, start the workflow by hand for such a push.
   - One exception to "never from the code's own output", named where it is used: a frozen
     baseline, taken from a run that an independent check confirmed, with a header that says how
     it was made (W1 step 4).
3. **What the owner changed on 2026-10-07:** the tool is to be general, not a tool for one class.
   - These end when W1 step 1 is committed: the deadline, the order of `SPEC.md` section 14,
     "Questions for J", and the rule that a superseded test is marked and kept (decision 2).
   - These end with W1 step 5, in the same commit that removes the template: "ported code keeps
     its behavior" and the list of template files.
4. **Never mix a move with a change of behavior.** A commit that moves or renames files changes
   nothing else, so that the proof is simple: the same tests, the same counts.
5. **Saved files of an older version stay safe.**
   - A change of `session.json` or `results.npz` comes with a reader for the old version. Today
     both are checked by strict equality of their version and no migration exists.
   - A build of this phase never replaces a file of an older version without keeping it: the
     first save of a session that was read as version 1 leaves the old file beside it
     (`session.v1.json`; later `results.v1.npz`), and the window says so once. Opening and closing
     an old run folder without a change writes nothing. Version 0.1.0 refuses a newer file, so
     this is what lets a user go back.
   - A removed or renamed CSV column has no reader: the changelog and the generated README.txt
     name it and give the formula.
   - An old file for a test is built in the test from literal values; no results file is
     committed.
6. **Performance is proved by counting, not by timing.** A test counts calls, bytes or array
   sizes; wall times go into a benchmark report (W5).
7. **Requirements live in one place.** If a session hands work to helpers, the requirements are
   in one written brief. A second wording of them elsewhere caused superseded tests in the first
   build.

## 3. Decisions the owner makes first

Nothing below is decided. A line with a default lets the owner say "the default stands"; a line
that says "no default" needs an answer. Each line ends with the workstreams that need it. A
session writes the owner's answer under the line as "Decided <date>: ...". Until
`docs/design/decisions.md` exists (W2 step 3), this section is the record.

Rules and legal:

1. License of the tool. Default: Apache-2.0 with a NOTICE file. [W0, W7; W6's choice of
   dependencies]
2. The rule for a superseded test. Today a test may never be edited or deleted; it is marked
   `xfail(strict=True)` and waits. Default: a test that a deliberate change makes obsolete is
   replaced in the same commit, and the commit message names the old test, the new one and the
   reason; a test that fails unexpectedly still may not be weakened. [W1]
3. Every test marked `xfail(strict=True)` whose reason names a successor, and the two superseded
   but unmarked tests in `tests/gui/test_theme.py`: delete them. These are 22 marked tests (20
   on every system, 2 on some systems only). Three reason texts carry an alternative that the
   deletion closes: decision X8 stands, and the Tracker-format file gets no rows for frames
   without a record. Default: delete. [W1]
4. The class template: `tests/reference/` holds ten files copied from the owner's template
   repository, and functions ported from it stay in the package after the copies go. Default:
   the owner confirms before the repository is made public that this code may be published
   under the license of decision 1; the copies and the port-fidelity tests are retired in W1
   after reference numbers are frozen. Once the repository is public its history cannot be
   taken back. [W0, W1]

Making the tool general (inventory: domain):

5. The word for a tracked thing. Default: "object" everywhere. Head, heading, body and core
   keep their names. [W3]
6. The name field. Default: an optional label. The run folder is `<video stem>_outline/`, with
   the label appended when there is one. When that folder already holds a session, Open video
   offers to open it or to start a new run in `<video stem>_outline_2/`. A label typed before
   results exist renames the empty run folder. `--student` stays as a hidden alias of `--label`
   for one release. Old folders open as they are. [W3]
7. Tracker (the physics video tool). Default: keep the import (`from-tracker`) and the
   Tracker-format export as general features in neutral words; the export becomes a switch that
   is off by default and that `from-tracker` turns on. [W3]
8. Boundary. Default: a circle only, optional, named "Boundary". Without one, panel 4 is done
   from the start (the axes exist, origin at the frame's center) and its hint says that a
   boundary is optional; the crop box is off and disabled until a boundary exists. The two
   distance-to-wall columns of `shapes.csv` are removed: such quantities are the user's own
   analysis, and the boundary's center and radius are in `session.json`. [W3]
9. Brightness probes. Default: keep, with the neutral default name `probe1`. The controls of
   panel 5 are a feature for the list of W7, not part of W3. [W3, W7]
10. Frame rate. Default: the file's own frame rate becomes a source, preselected and labelled
    "from the file: check it for slow-motion recordings"; the stopwatch helper and typing stay;
    the class's manifest file and the warning under 100 fps go; `fps_true` keeps its name in
    files. [W3]
11. The checks of a phone video that run on every opened video. Default: only in the `check`
    command, in neutral words. [W3]
12. `session.json` version 2. Default, as one table that the owner approves before any code:
    `student` becomes `label`; `circle` becomes `boundary` with `shape: "circle"`;
    `circle.dish_mm` becomes `boundary.diameter_mm`; `processing.dish_crop` becomes
    `processing.boundary_crop`; `time.manifest_path` is dropped; a `time.source` of `manifest`
    is read as `typed` with its value kept; the new value `file` is added. [W3]
13. Units and defaults. Default: mm stays; the stick and tape boxes start empty; the limit of
    the `JUMP` flag has no built-in value and the flag is off until the user sets one. [W3]
14. What else of the first class goes (inventory: domain 1.7, 2.3, 2.5, 2.10). Default: the
    guard that refuses a folder with other `.csv` files goes; `convert` writes
    `<stem>_h264.mp4`; `from-tracker` keeps its three checks in neutral words; the part "What
    goes into git" of the generated README.txt goes. [W3]
15. `SPEC.md` and `docs/PLAN.md`. Default: `SPEC.md` moves unchanged to `docs/design/` as the
    frozen specification of 0.1.0. The tests that read it keep doing so for version 1: they
    become the tests of the version 1 reader and of the 0.1.0 column lists. The contract of
    every changed format is written first, by hand, in `docs/design/formats.md`, and new tests
    take their expected values from that page. Decisions X1 to X22 move to
    `docs/design/decisions.md`. Before `docs/PLAN.md` leaves the tree, its still-open points go
    into this list or into issues (decisions 26 to 28 below are three of them). [W2, W3]

Tracking methods (inventory: backends, section C7):

16. What "scale by orders of magnitude" means: more things (hundreds to 100,000 points), longer
    clips (10,000 to a million frames), smaller or larger objects, or speed. Is there a target,
    for example N points over M frames in so many minutes on a laptop? No default: ask. [W6]
17. Is the tool still about outlines, or about positions and, where a method gives one, outlines?
    Default: both; tracks without a shape are allowed, and the hybrid comes first (a point
    method steers the crop, a mask model draws the outline). [W6]
18. CoTracker3 is licensed CC-BY-NC 4.0, code and weights. Default: not shipped and not a
    dependency; at most a method the user installs, clearly labelled. [W6]
19. May torch become optional, so that a small install can run the classical methods? Which
    method is then the default and the selftest? No default: ask. [W6]
20. May a method hold a window of frames in memory? What bound replaces "never the whole
    video"? No default: ask. [W6]
21. One method per session, or per object? Default: per object. [W6]

The window and the project:

22. Help: are small "?" buttons and a help page in a dialog allowed? The design note
    (`docs/design/gui_design.md`) has two rules against them (inventory: help, section 3).
    Default: yes to both. [W4]
23. Supported systems: the lowest macOS, Linux yes or no, the Python versions CI must test. No
    default: ask. [W7]
24. Pins. Exact pins suited one class on one day. Default for a published tool: ranges in
    `pyproject.toml`, the lock file for development, dependency updates automated. [W7]
25. Track colors: a list that is safer for red-green color blindness is in the design note.
    Default: switch to it. [W3]
26. Two findings about the model, each with slow tests marked `xfail`. Default: the size checks
    use the largest piece of a mask (two stray pixels switched a check off); the 0.02 limit of
    fine mode is for the variation of solidity (RMS after the mean offset is taken out), with a
    click on the body and on each thin part. [W1]
27. Two open readings of the spec (the plan's question 15): one jump of the axis flags one frame
    as `ORIENT`, not the rest of the track; long shapes always take the core fallback. Default:
    as built. [W1]
28. Two open points of the window: a stored fine window cannot tell "auto" from "set by hand"
    (an Auto button exists); the smallest window stays 960 px wide. Default: as built. [W1, W4]

## 4. Workstreams

### W0. Close version 0.1.0 (the first build's own session)

- [x] The last follow-up of the first build is merged (task C9: export without the model, one
      lock for runs and exports, the model box after results). It changed decision X7: the model
      is made in a short-lived thread of its own and used in the one worker thread, so that an
      export can run while a model loads. Checked with the real model on a Mac only.
- [x] After that merge, the numbers of this file were counted again. All seven tests the
      follow-up marked have successors (decision 3).
- [x] The README carries the advice on fine mode for a lost object (in Troubleshooting).
- [x] The owner's check on the Mac is recorded in `docs/VALIDATION.md` (section 6.4).
- [ ] Both CI systems are green on the merged commit.
- [ ] The owner's check on a Windows machine, on the commit that will be tagged: install,
      selftest, the window opens, "Model ready" appears, a short run and an export finish.
- [ ] Decisions 1 and 4; then a LICENSE file, the tag `v0.1.0`, both install lines of the README
      with the tag.
- [x] This file, `docs/roadmap/` and `docs/design/gui_design.md` are committed and pushed.
- [ ] The repository is made public.

### W1. A clean base (inventory: `docs/roadmap/tests.md`)

Goal: no marked tests, no template in the tree, rules that fit a general tool. Why first: the
port-fidelity tests pin the source text of functions that hold the class's words, so W3 cannot
change them until the template is retired. Start: 25 strict `xfail` marks (20 superseded
everywhere, 2 on some systems only, 3 real findings about the model); 25 helper files in three
test folders; the lock simulation written in 11 files; 22 test files over 400 lines; three GUI
tests open a gate from a timer thread and one sleeps.

- [ ] 0. The version becomes `0.2.0.dev0` (`pyproject.toml`, `outline_tracker/__init__.py`, the
      test of the version line, the README's example line). CI also runs for pushes that change
      only documents.
- [ ] 1. Rewrite `CLAUDE.md` for this phase (section 2 of this file; decision 2), and
      `docs/DEVELOPER.md` where it repeats the rules. `tests/gui/test_finish.py` pins phrases of
      `docs/DEVELOPER.md`, among them the xfail rule: change that test first. Done when the
      owner has approved the new `CLAUDE.md`.
- [ ] 2. Delete the superseded tests (decision 3); correct the texts that mention them. Done
      when the fast suite reports 0 xfailed on macOS, Linux and Windows.
- [ ] 3. Carry out decisions 26 and 27: rewrite the slow tests to the decided rules, and record
      the limits in `docs/VALIDATION.md`.
- [ ] 4. Freeze reference numbers while the template is still there: the positions of the two
      slow regression runs, the Tracker-format golden files, the weights' hash, as small text
      files under `tests/data/` with a header that says how each was made (commit, machine,
      system, library versions). They come from a run on the processor in which the template's
      code still agrees within 0.01 px. Done when the old comparison and the new test both pass
      in the same run. The limit is 0.01 px on the machine that froze the numbers; on another
      machine it is measured, not assumed. Mind the line ends of the golden files on Windows.
- [ ] 5. Remove `tests/reference/`, the port-fidelity tests and what else imports the template
      (14 test files); keep their value as tests against the frozen numbers. `CLAUDE.md` loses
      the two port rules in the same commit.
- [ ] 6. Ordinary `conftest.py` files for the fixtures; the duplicated helpers joined where they
      are (their move into one folder is W2 step 3); one lock simulation as a fixture. The
      shapes: keep `outline_tracker/synthetic_shapes.py` and delete `tests/analytic_shapes.py`
      only after a test shows that they give the same shapes.
- [ ] 7. Replace the timer threads and the one `sleep` in GUI tests by events; test the bottom
      row by rule (nothing clipped, nothing overlapping at 960 px with each CI system's font).
- [ ] 8. CI: a macOS job; the slow tests that need no weights.

### W2. Reorganize the repository (inventory: `docs/roadmap/repo.md`)

Goal: a newcomer finds things. Start: 84 package files, 45 of them side by side at the top level
in prefix families (`tracking*.py`, `export*.py`, `cli_*.py`); 7 package files at or over the
400-line limit (13 over 380); eight kinds of imports that go against the layers. The
inventory's table was made before two modules were added (`gui/panel_parts.py`,
`gui/prompt_drawing.py`): give them rows.

- [ ] 1. Re-key the tests that pin paths (`tests/test_repo_rules.py`, the DEVELOPER page test,
      the import-order tests) so that they follow a table, before anything moves.
- [ ] 2. Commit 1: the package into subpackages, with `git mv` and the import lines, nothing
      else. The proposed mapping of every module is in the inventory, section 1.2 (`cli/`,
      `core/`, `video/`, `measure/`, `tracking/`, `segmenter/`, `export/`, `workflows/`,
      `synthetic/`, `gui/`). Done when the fast suite gives the same counts.
- [ ] 3. Commit 2: tests into the same subjects, helpers into one folder. Commit 3: documents
      (`docs/user/`, `docs/developer/`, `docs/design/`; decision 15).
- [ ] 4. Make the layers true, one small commit each: the constants and helpers that the library
      and the window import from command-line modules; the model's factory out of the
      from-tracker module; shared panel widgets into one module (two classes are named `Message`).
- [ ] 5. A module index generated from each module's first docstring line, kept current by a
      test, as `docs/OUTPUTS.md` is.
- [ ] 6. Split the files over the limit along the new layout.

### W3. Make the tool general (inventory: `docs/roadmap/domain.md`)

Goal: no trace of the first class in names, texts or behavior; a boundary that is optional.
Start: the package computes no reaction time; what served that exercise is the dish circle, two
distance-to-wall columns and the probe. Tracking and export already work without a circle. Class
words sit in window texts (about 45 lines), command help and console lines, generated file
descriptions, the README (all of it), the session keys `student`, `circle.dish_mm`,
`processing.dish_crop` and `time.manifest_path` and the `time.source` value `manifest`, two
column names, the run folder's name, the model cache folder `~/.cache/shrimp-models`, and the
synthetic scenes.

- [ ] 1. `session.json` version 2 (decision 12): the table goes into `docs/design/formats.md`
      and is approved; then the reader lands and maps every row; steps 2 to 4 switch the code
      over one row at a time. No tag is made between step 1 and step 4. Done when a version 1
      file of each kind (made in the window, made by `from-tracker`) opens and exports; after
      opening and closing, the version 1 file is byte for byte what it was; after a change,
      `session.v1.json` holds it (rule 5).
- [ ] 2. Dish to Boundary (decision 8): session keys, panel 4, the crop's names, run.log, the
      tool; remove the two distance columns and regenerate `docs/OUTPUTS.md`.
- [ ] 3. The name field becomes an optional label, with the folder rules of decision 6.
- [ ] 4. Frame rate sources (decision 10), the phone-video checks (decision 11), the switch for
      the Tracker-format export (decision 7), and what else goes (decision 14).
- [ ] 5. Texts: the window, command help, console lines, generated descriptions, docstrings and
      comments, `docs/VALIDATION.md`, `docs/DEVELOPER.md`, `CHANGELOG.md`, `pyproject.toml`, the
      CI file; neutral defaults (decision 13); "object" everywhere (decision 5); the track
      colors (decision 25).
- [ ] 6. Neutral names inside: the synthetic scenes and shapes, the selftest's wording and its
      path through a made-up Tracker export, fixtures and test names (the fixture `dish_clip` is
      used about 930 times in 38 test files: a mechanical rename, its own commit).
- [ ] 7. The model cache: a neutral folder and variable. The old folder and the variable
      `SHRIMP_MODEL_CACHE` are still read, in place; the old folder is never moved or deleted.
- [ ] 8. The README for a stranger: what the tool is, a picture, install, Quickstart (rule 1
      says what stays for 0.1.0). Material that only the first class needs moves to one page.
- [ ] 9. A rule test: the class's words (the list of the inventory, section 1.1) do not occur in
      the package, the window's texts or the generated documents, with a short list of allowed
      uses (Tracker as the name of a file format).

### W4. Help inside the app (inventory: `docs/roadmap/help.md`)

Goal: every control explains itself. Start: 66 of 76 controls have a tooltip and nothing else;
no menu item and no table header is described, and no entry of a combo box has a text of its
own; keys and gestures are explained only in scattered tooltips (11 of 14 keys, 5 of 10
gestures; Esc, the undo key and the keys 1 to 9 appear nowhere, and there is no list); a control
that is off shows the reason in place of its description; 22 terms need a definition or a fuller
one (clip step, coarse and fine, the nine flags, the "±" of the scale).

- [ ] 1. One table of help texts without Qt (key, label, one sentence, a longer text, glossary
      terms), with the flags' texts taken from the schema.
- [ ] 2. An object name on every control; the table applied to tooltips, What's This texts and
      accessible names; "what it is, then why it is off" for controls that are off.
- [ ] 3. "?" buttons on panel headers and beside the hard terms, with a popover (decision 22); a
      Help menu with controls and terms, keys and mouse, Quickstart, About, all readable offline.
- [ ] 4. `docs/CONTROLS.md` generated from the table.
- [ ] 5. Tests that keep it complete: a walk of the widget tree fails for a control without help;
      no entry without a control; a wording check; the generated page is current.

### W5. Performance (inventory: `docs/roadmap/perf.md`)

Goal: the window stays fluid with 10 objects and scales to long clips. Tracking speed itself is
the model's; this list is the app's own share. Start (10 objects, 1,200 frames each, measured
with stand-ins): one tick of Play costs 55 ms against a tick of 33 ms; a click 218 ms; the
window stands still 0.4 s at every autosave; the flags table takes 5.2 s and the export derives
everything again; `results.npz` is rewritten whole at each autosave (computed for 10 objects over
36,000 frames: 73 GB written); a coarse job decodes from frame 0 twice, a fine job three times;
overlay, probes, export and the flags listing have no progress and no cancel.

- [ ] 1. A `benchmarks/` folder outside pytest (layout in the inventory, section 5), and its
      first report as the baseline.
- [ ] 2. P1: the results store keeps columns, not records; cheap questions (`frames`, `has`,
      `last_frame`); the panels ask once per version of the store.
- [ ] 3. P2: one read-only store for the whole window; appended blocks while tracking instead of
      whole rewrites. Done when a run killed at any frame loses at most the last block and the
      next open brings the blocks into `results.npz` (a test kills the job), and a folder
      written by 0.1.0 opens unchanged. `results.npz` itself stays version 1 here; its next form
      is W6 step 3, one change only.
- [ ] 4. P3 and P4: one cached derivation shared by flags, export and the command line; array
      code for hull, Feret and rays; the contact check decided cheaply for most frames.
- [ ] 5. P5: a run starts from a verified seek; a start frame is decoded once.
- [ ] 6. P6: progress and cancel for overlay, probes, export and the flags listing; work that
      needs no model does not wait behind the model.
- [ ] 7. P7 to P14: start-up (torch before the window only where it is needed, proved on a real
      Windows machine), opening a video off the GUI thread, the bottom bar on long clips, the
      frame cache, overlay items reused, work on the 256-cell grid instead of the full frame,
      the opening of large masks.
- [ ] 8. The test suite (190 s in one process): one selftest clip per session, stand-ins that
      work inside the object's window, tracked folders from session fixtures.

### W6. More tracking methods (inventory: `docs/roadmap/backends.md`)

Goal: the user chooses a method that fits the task, and the tool is honest about what each
method gives. Start: the whole tool assumes one mask per object per frame, in a forward loop;
the list of models lives in six places; a method that returns a point would lose even the
position today, because an empty mask is a lost frame. Licenses read from the sources: SAM 2.1,
EdgeTAM and the TAPIR family are Apache 2.0 for code and weights; CoTracker3 is CC-BY-NC 4.0;
phase correlation with an upsampled DFT (scikit-image; the owner's "DFT upsampling") and
pyramidal iterative Lucas-Kanade (OpenCV; the owner's "Iterative Lucas-Kanade") need no weights
and are installed already. A first probe: Lucas-Kanade took 1.8 ms per frame pair for 100 points
and 49 ms for 100,000; chained phase correlation drifted 0.4 to 2.5 px over 29 steps.

- [ ] 1. A design session with the owner on decisions 16 to 21; write the result to
      `docs/design/backends.md`.
- [ ] 2. One registry of methods with declared capabilities (masks or points, several objects,
      online or a window of frames, devices, weights with address, size, hash and license); the
      commands and the model box read it.
- [ ] 3. Results format version 2: the method per track, a position that does not need a mask;
      a reader for version 1 (rule 5). Outputs, flags and overlays say plainly what a method
      does not give.
- [ ] 4. The classical methods as built-ins: pyramidal iterative Lucas-Kanade, and phase
      correlation with an upsampled DFT for sub-pixel shifts, against a fixed reference patch,
      with a quality measure.
- [ ] 5. The benchmark: synthetic clips swept over object size, count and length, ground truth
      for body-fixed points, one generated table across methods and devices in the docs.
- [ ] 6. SAM 2.1 in its four sizes, each with its slow test first (none was ever run here).
- [ ] 7. The hybrid: a point method steers fine mode's crop (aimed at the measured weak point).
- [ ] 8. The TAPIR family as an optional install (git only today, so vendored or installed by
      hand); torch as an optional extra if decision 19 says so; CoTracker3 only as decision 18
      allows.

### W7. Open-source readiness (inventory: `docs/roadmap/oss.md`)

Goal: a stranger can install, trust, use and contribute. Start: no LICENSE, CONTRIBUTING, code
of conduct, security note, citation file or issue forms; no tag and no release; not on a package
index; CI has no macOS job, lint, type check, coverage, or wheel and sdist build check (the
Windows job does install the tool from the checkout); the model download pins no revision,
compares its hash with nothing and reads a checkpoint with
`torch.load(..., weights_only=False)`; `run.log` is the only thing a user can send when
something fails.

- [ ] 1. Hold the brainstorm with the owner: the inventory's section 3 lists about 50 candidates
      in six groups (stability, usability, features, distribution, community, quality gates),
      each with its value and a rough size. Record what was chosen, in order, in this file.
- [ ] 2. The files every public project has (decision 1): LICENSE, NOTICE, CONTRIBUTING, code of
      conduct, security note, citation, issue forms, project metadata in `pyproject.toml`.
- [ ] 3. Trust in the first run: pinned model revisions, a checked hash, no pickle loading.
- [ ] 4. When something fails: a log outside the run folder, a crash handler, a diagnostics file
      that Help can write.
- [ ] 5. Releases: one source for the version, a release workflow, a package index
      (decisions 23 and 24).
- [ ] 6. Quality gates: lint and format, type check, coverage, a Python version matrix, a build
      and install check.
- [ ] 7. From the first plan, never built: the controls of the probe panel, a live view during a
      run, draggable handles, the flag timeline, overlay extras, resume after cancel, a headless
      `run` command, convert from the window, screenshots (the plan's tasks D1 to D11); backward
      tracking, box prompts, undo beyond clicks, image sequences (the spec's P2 list). They
      belong on the brainstorm's list.

## 5. Order, and what can run side by side

1. Decisions 1 and 4, then the rest of W0. Decisions 2 and 3 come before W1.
2. W1: the base must be clean before files move.
3. W2: a pure move. Nothing else may be open while it runs: it touches every file.
4. W3: the renaming lands at the final paths.
5. Then, side by side: W4 (window texts and a new help module) and W5 (store, derive, video,
   worker). They meet in the panels; let one finish a panel before the other changes it.
6. W6 needs W2 and W3 for its code, and W5's store for results version 2. Its design session
   (step 1) can happen at any time after W0.
7. W7: steps 1 and 2 can happen at any time after W0; steps 3 to 6 after W2.

Never side by side: W2 with anything; two sessions that both change a saved format.

## 6. Starting a session

Paste this, with the workstream filled in:

```text
You are continuing Outline Tracker, a desktop app and command line tool that tracks objects in
videos and exports positions and outlines. Version 0.1.0 was built for one lab class; this phase
makes it a general, open-source tool.

Your workstream: W_ (as named in docs/ROADMAP.md section 4).

Read first, in this order: CLAUDE.md, docs/ROADMAP.md (all of it), the inventory of your
workstream in docs/roadmap/, docs/DEVELOPER.md. Section 2 of docs/ROADMAP.md holds the rules of
this phase; where CLAUDE.md, docs/DEVELOPER.md or an inventory says otherwise, section 2 wins.

Before any change:
1. Check the start: `git tag -l v0.1.0` prints the tag, `git status` is clean, and the
   workstreams that section 5 puts before yours have their boxes ticked. If not, stop and tell me.
2. In one message, list the decisions of section 3 that your workstream needs and that have no
   "Decided" line, with their defaults, and ask me to confirm or change them. Wait. Write my
   answers under those lines.
3. Write a short plan for this session: the steps you will do, in order, each with its tests and
   how I can check it. Wait for my approval.

While working: test first; commit and push after each step. Do not wait for CI, but look at
`gh run list` before the next step and fix a red run first. Slow tests with the real model on
short synthetic clips are fine; never run the model on a long real video.

At the end: tick the finished boxes in docs/ROADMAP.md, add one line to its log (section 7), and
tell me what changed, how it was verified, and what is uncertain.
```

For the brainstorm (W7 step 1) and the design session (W6 step 1), replace the line "Your
workstream" by: "This session is a discussion, not a build: W7 step 1 (or W6 step 1). Go through
the candidates and the open decisions with me, ask what you need to know, and write the outcome
into docs/ROADMAP.md. Change no code."

## 7. Log

One line per session: date, workstream, what was done, the last commit.

- 2026-10-07: roadmap written from seven inventories of the code (taken at `f73d29f`), and
  checked once against the code and by a cold read.
- 2026-10-07: W0: the last follow-up of the first build merged (`f69afb7`); numbers counted
  again; README advice on fine mode; the owner's Mac check recorded.
