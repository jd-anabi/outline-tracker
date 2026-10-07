# Roadmap inventory: what is specific to the shrimp lab

> Inventory for `docs/ROADMAP.md`: what is specific to the first class of users. Read-only findings at commit `f73d29f` (2026-10-07).
> File and line pointers are true for that commit. "The ledger", "the build ledger" and "the
> scratch folder" are private working notes of the first build; they are not in the repository.
> The design note of the window is `docs/design/gui_design.md`.

Read-only inventory of main at f73d29f (2026-10-07). No test was run, no file was changed. `.worktrees/` was not looked at.
Paths are relative to the repository root; `pkg/` stands for `outline_tracker/`. Counts are lines of tracked files that match a word (case ignored), with the number of files in brackets. They are not counts of occurrences.

## 0. The short version

- The package computes no reaction time. `reaction` matches 0 lines in `outline_tracker/` and in `tests/`; it is in SPEC.md only (:54, :168). What serves the reaction-time exercise is three helpers: the dish circle, the two wall-distance columns, and the LED brightness probe.
- Tracking and export already work without a circle (section 3). "Optional" is mostly a matter of texts, the panel's state and two column names.
- Three names in saved files are class names: the session keys `student`, `circle.dish_mm` and `processing.dish_crop`, and the shapes.csv columns `wall_dist_centroid_mm` and `wall_dist_min_mm`. No reader for old files exists today (section 4).
- The largest block of class history is not wording but structure: code "ported from last week" whose source text is pinned by tests against `tests/reference/shrimp/` (section 2.8). Many texts with "shrimp" sit inside those pinned functions.

## 1. Words, by where they are

### 1.1 Counts per group

| group | shrimp | antenna | dish | wall | animal | student | last week | manifest | stopwatch | LED |
|---|---|---|---|---|---|---|---|---|---|---|
| package, core (no gui) | 60 (14) | 18 (5) | 68 (14) | 25 (8) | 17 (8) | 54 (12) | 76 (22) | 67 (16) | 21 (10) | 16 (5) |
| package, gui | 1 (1) | 2 (1) | 55 (8) | 8 (4) | 21 (8) | 41 (7) | 3 (2) | 48 (2) | 42 (4) | 1 (1) |
| tests (without reference) | 163 (35) | 28 (9) | 1305 (80) | 86 (21) | 52 (13) | 170 (37) | 95 (32) | 157 (18) | 55 (13) | not counted |
| tests/reference | 57 (11) | 1 (1) | 0 | 0 | 0 | 1 (1) | 1 (1) | 16 (3) | 5 (4) | 0 |
| README.md | 7 | 1 | 1 | 1 | 14 | 3 | 10 | 4 | 5 | 0 |
| SPEC.md | 22 | 14 | 37 | 13 | 2 | 28 | 24 | 7 | 7 | 4 |
| docs/OUTPUTS.md | 0 | 2 | 6 | 3 | 8 | 0 | 4 | 0 | 0 | 1 |
| docs/DEVELOPER.md | 0 | 0 | 1 | 0 | 0 | 2 | 5 | 3 | 3 | 0 |
| docs/VALIDATION.md | 14 | 12 | 9 | 3 | 2 | 3 | 23 | 0 | 0 | 0 |
| docs/PLAN.md | 38 | 12 | 47 | 7 | 2 | 34 | 49 | 11 | 7 | 7 |
| CLAUDE.md, CHANGELOG.md, pyproject.toml, CI | 4 | 0 | 1 | 1 | 3 | 5 | 10 | 1 | 1 | 0 |

- Words with almost no hits: `tap` 0 everywhere; `reaction` SPEC.md 2; `nauplii` SPEC.md:229 and 1 test line; `lab` 1 test line and 2 in tests/reference; `stimulus` pkg/cli.py:101, pkg/probes.py:3, SPEC.md 2.
- `course` (the code's word for the class): 10 lines in the package, e.g. pkg/geometry.py:36, pkg/gui/tools.py:38 and :315, pkg/gui/panels/time_panel.py:1 and :7.
- `dish` in tests: 928 of the hits are the fixture name `dish_clip` (tests/helpers.py:20). 262 test functions have "dish" in their `def` line, 17 in the test's own name.
- The program "Tracker" (not "Outline Tracker"): 169 lines in 47 package files. 54 of them are the convention phrase ("Tracker's convention", "as Tracker counts"). See 1.7.

### 1.2 Texts a user sees in the window (about 45 lines)

- Panel titles and start hints, pkg/gui/main_window.py:51-56: "Student and video", "(a _tracker.mp4 file)", "measure it with the stopwatch", "on the ruler", "Dish and axes", "inner wall of the dish", "Add a box over the LED", "click on one animal".
- Panel 1, pkg/gui/panels/video_panel.py:48-50, :220, :253: "Type your name. The run folder is named after you.", "Open your video (a _tracker.mp4 file)."
- pkg/gui/session_controller.py:43 (".MOV file, run outline-tracker convert"), :45 `NAME_FIRST`, :59 `TRACKER_FILES` ("looks like a folder of Tracker files").
- Panel 2, pkg/gui/panels/time_panel.py:39-46 (source words "from the manifest", "from the stopwatch", "from the Tracker export"; `HINT_UNDER` "under 100 fps ... re-timed copy"), :85-90 (buttons "Stopwatch…", "Choose manifest").
- Panel 3, pkg/gui/panels/calibration_panel.py:210-216: "Stick", "Tape", "on the ruler", "two other ruler marks". These are Tracker's words on purpose (SPEC.md:72).
- Panel 4, pkg/gui/panels/dish_panel.py:44-60 ("Dish", "Dish diameter", "inner wall of the dish", "Crop to dish for tracking", "sharper on the animals"), :138-141 ("Dish: R = ...", "No dish circle."); pkg/gui/tools.py:40 and :382.
- Panel 6, pkg/gui/panels/objects_panel.py:50-61: "animal" in 8 lines, and "If the antennae matter, click on each antenna too" (:52-53).
- Panel 7 and 8: pkg/gui/panels/track_panel.py:93; pkg/gui/panels/review_panel.py:55, :56, :68; pkg/gui/prompts.py:51, :57 ("animal {id}").
- Help menu: pkg/gui/menus.py:28 links to the README's Quickstart, which is written for the class.

### 1.3 Command line help and console output

- pkg/cli.py:70-75 (`convert`: "a phone video that Tracker can open", iPhone, 240 fps), :98-101 (`probe`: "an LED", "a stimulus LED (Tracker's RGB Region)"), :115-119 (`--fps` default "from data/manifest.csv"; `--student`), :130-132 (`synth dish|closeup`: "small shrimp in a dish", "one shrimp beating its antennae"), :184 and :223 ("your stopwatch clip"), :285-288 ("dish wall", "LED box").
- pkg/cli.py:33-35 and pkg/convert.py:15: an example path with the class's file naming (`data/raw/groupB_..._main.MOV`).
- pkg/cli_from_tracker.py:18-25, :41-45 ("last week's way to start, this week's files"), :50-72 (`--student`, "--no-video is last week's spelling", "data/manifest.csv").
- pkg/cli_selftest.py:21-23, :37 ("shrimp-sized ellipse", "the test shrimp").
- pkg/cli_probe.py:42 (`LED1`, `--student ana`), :53 (`UNKNOWN_FPS` "fill in data/manifest.csv"), :217, :221, :246-247.
- Console lines: pkg/from_tracker.py:177 ("{n} shrimp (A, B)"), :67 ("two shrimp touching"), :244-251 ("student name: ..."); pkg/selftest.py:57-60 ("followed the test shrimp", "one shrimp about ... 10 shrimp together"); pkg/tracker_io.py:231-232 ("Every shrimp must be marked ..."), :243; pkg/video.py:196-222 (see 2.6), :234 ("made by shrimp.convert": names a module users do not have).

### 1.4 Output file descriptions (README.txt and docs/OUTPUTS.md, both built from schema)

- pkg/schema.py:80 ("pieces of the same animal"), :92 ("whole frame or dish"), :105-110 ("antennae"), :120-123 (wall distance), :138 ("default LED1"), :182 ("dish crop"), :229-232 ("last week's Tracker files", "wall distance"), :237 ("the LED"), :239-240 ("plays in PowerPoint and Google Slides").
- pkg/schema_docs.py:52 and :124 ("as last week"), :149 ("at dish scale"), :346-359 (glossary: "Tracker: the video-analysis program you used last week", "one animal", "antennae"), :381 ("the contract between the tracker and the analysis template").
- "What goes into git": pkg/schema_docs.py:152-156, :247-250, :276, :338, :400-402, and the field `FileSpec.in_git` (pkg/schema.py:220). It is the class's group-repository rule ("share them through Drive"). It also names `calibration.json`, which nothing in the package reads or writes (the only hit is pkg/schema_docs.py:153).
- docs/OUTPUTS.md is generated: change pkg/schema.py and pkg/schema_docs.py, then regenerate (pkg/schema_docs.py:5).

### 1.5 README.md (230 lines; the whole page addresses a student of the class)

- :3 "follows animals"; :5 "you have both from last week"; :45-55 selftest ("made-up shrimp", "last week's selftest", "send ... to your instructor").
- Quickstart :71-79: your name, `_tracker.mp4`, manifest, Stopwatch, ruler, dish wall, antennae.
- "If the app does not open" :91-124: the "Last week / This week" option table, `python -m shrimp.segment`, `data/manifest.csv`, "2 shrimp (A, B)", "last week's three checks".
- :128 (name from the export's folder, `extra`), :136-137 ("format of last week's files", "for your slides"), :143 (git and Drive), :145-151 (`load_tracks` of the class notebook), :181-194 (stopwatch clip), :221, :223-230 (phone section).
- tests/test_readme.py (32 tests) pins much of this: the three section titles (:46), the last-week option table and `--student` (:292-306), "panel 4" in step 5 (:359-365), "antenna" in step 6 (:376-380).

### 1.6 Other documents

- SPEC.md (938 lines) is the class's spec throughout: decisions D6 to D9 (:54-57), assumptions A1 to A7 (:60-66), the workflow (:74-86), the Thursday plan (:739), Appendix A (:886).
- docs/PLAN.md (1744 lines) is the two-day build history. docs/VALIDATION.md (626 lines) reports measurements "against last week's script" (:7-103, :250-291, :474-534).
- docs/DEVELOPER.md:3 ("Students read README.md"), :24, :29, :31, :55-56, :92, :94, :123-124, :141, :173.
- CHANGELOG.md:12-13, :15, :25-28, :34-38, :45, :47, :51, :57.
- pyproject.toml:8-10, :22-23 (comments on students and last week), :47-49 (`tests/reference` on the test path, "package `shrimp`"). The description (:4) is already general.
- .github/workflows/tests.yml:20, :61 ("students").

### 1.7 "Tracker": format support or class history

Format support (a general feature if the owner keeps it):
- Import: pkg/tracker_io.py (`read_tracker_export` :55, `fit_calibration`, `make_plan` :219), pkg/from_tracker.py, pkg/from_tracker_session.py, pkg/cli_from_tracker.py; session keys `calibration.tracker_fit` and `time.source = "tracker-export"` (pkg/session_parts.py:138, :158).
- Export: the folder `<model>/<id>.csv` in Tracker's layout (pkg/schema.py:51, :146-153; pkg/export_tables.py:187), written by `write_tracker_file`. Hidden command `compare-tracks` (pkg/cli.py:141, :298).
- `convert`: an H.264 copy named `<stem>_tracker.mp4` (pkg/convert.py:42).

Convention (neutral in substance): "Tracker's pixel convention (pixel centers at +0.5)" in about 54 docstring lines, in CLAUDE.md:8 and in pkg/schema.py:81, :88-89, pkg/schema_docs.py:104, :113.

Class history:
- "last week" next to Tracker (76 lines in 22 core files; most in pkg/from_tracker.py 13, pkg/geometry.py 7, pkg/from_tracker_session.py 7, pkg/segmenter/hf.py 6, pkg/qc.py 5).
- "the ..._tracker.mp4 copy you opened in Tracker" (pkg/cli_from_tracker.py:21, :50; pkg/from_tracker.py:225); the window's hint that a video is a `_tracker.mp4` file (pkg/gui/main_window.py:51).
- The export's home folder called `extra` gives the student's name (pkg/from_tracker_session.py:83-92).
- The guard that refuses a folder with other .csv or .txt files, because the class notebook reads every such file as a track (pkg/run_folder.py:20, :46-66; pkg/gui/session_controller.py:59; pkg/cli_probe.py:221; README.md:219-221).
- Tracker's tool words in the window (Stick, Tape, Circle, Origin to Center, Axes; SPEC.md:72).

### 1.8 Template names

- Model cache: `~/.cache/shrimp-models/edgetam` and `SHRIMP_MODEL_CACHE` (pkg/segmenter/edgetam_convert.py:85, :161; docstrings :9, :84; pkg/segmenter/hf.py:96).
- `EXTRA_PER_SHRIMP` (pkg/selftest.py:28), temp folder prefix `shrimp-selftest-` (:43), `student="selftest"` (:46).
- `shrimp-tracker-template`: CLAUDE.md:9, tests/reference/README.md, tests/test_repo_rules.py:22-34.
- Synthetic scenes: class `Shrimp` and `shrimp_shape` (pkg/synthetic_shapes.py:70, :113), class `Led` (pkg/synthetic.py:59), `Scene.dish` and `Scene.led` (:85-96), `dish_scene` (:281-306), `closeup_scene` (:310-326), the shapes scene "inside a dish circle" (:345-362), `selftest_clip` (:370-380).
- Tests: file names tests/gui/test_dish_panel.py, tests/test_from_tracker*.py (4), tests/test_port_*.py (2), tests/test_measure_port_guard.py; helpers `dish_circle` (tests/tracking_helpers.py:34), `dish_run` (tests/gui/review_helpers.py:75, tests/slow/test_coarse_lowres.py:54), `write_manifest` (3 copies), `wall` (tests/gui/test_dish_panel.py:27, tests/gui/test_tools.py:34), `exact_shrimp_solidity` (tests/test_synthetic.py:133). Test names: 20 with `last_week`, 30 with `tracker`, 19 with `manifest`, 9 with `shrimp`, 5 with `stopwatch`, 5 with `student`.

## 2. Behavior that exists only for the class

1. Reaction time. Nothing computes it (SPEC.md:57, D9: "The app computes no derivatives or statistics"). Its helpers: the circle (section 3), the columns `wall_dist_centroid_mm` and `wall_dist_min_mm` (SPEC.md:332-338; pkg/derive.py:63-65, :168-173; pkg/schema.py:120-123), and the brightness probe "made to time a stimulus LED" (pkg/probes.py:3; default name `LED1`, pkg/session_parts.py:204; panel 5's hint, pkg/gui/main_window.py:55). Panel 5 has no controls yet (docs/DEVELOPER.md:98); the `probe` command does the work.
2. Frame rate. The file's own frame rate is never a source ("displayed, never used", SPEC.md:120; pkg/gui/panels/video_panel.py:229). The sources are typed, stopwatch, manifest and tracker-export (pkg/session_parts.py:138). So a user with an ordinary video must type the number.
   - Manifest `data/manifest.csv`: pkg/geometry.py:36, :347-400 (`find_manifest`, `fps_from_manifest`, `dish_mm_from_manifest`); searched when a video is opened (pkg/gui/panels/time_panel.py:174-187); remembered in the settings (:37, :126); CLI default (pkg/from_tracker_session.py:57-76; pkg/cli_probe.py:226-239); `make_plan`'s default argument (pkg/tracker_io.py:219).
   - Stopwatch helper: pkg/geometry.py:231-248, pkg/gui/stopwatch_dialog.py. It is general in substance; only its place as "the" source is the class's.
   - The 100 fps warning: pkg/gui/panels/time_panel.py:36, :43; pkg/cli_probe.py:54, :242-247; pkg/from_tracker.py:244-247; pkg/video.py:192-199.
3. Defaults at the class's scale: stick 30 mm and tape 20 mm (pkg/gui/tools.py:38, "the course's ruler"); JUMP above 100 mm/s (pkg/session_parts.py:195; pkg/from_tracker.py:45); clip step 2 (pkg/session_parts.py:132); mm as the only length unit, in column names; `from-tracker` tracks 10 s by default (pkg/tracker_io.py:247); the selftest estimate "10 s at step 2 (1,200 frames)" assumes 240 fps (pkg/selftest.py:59).
4. Student name. Required: without it nothing is saved, tracked or exported (pkg/gui/session_controller.py:350-356, :312; pkg/gui/menus.py:87; pkg/gui/panels/export_panel.py:69). It names the run folder `<video stem>_outline_<student>` (pkg/run_folder.py:26-43). Flags `--student` in `from-tracker` and `probe` (pkg/cli_from_tracker.py:65; pkg/cli.py:117); `probe` needs `--student` or `--out` (pkg/cli_probe.py:217). Session key `student` (pkg/session.py:143).
5. `from-tracker` as "last week's workflow" (pkg/from_tracker.py:1): last week's console lines, three `CHECK:` rules and messages are kept on purpose (decision X22, docs/PLAN.md:434; pkg/from_tracker.py:45-68). The name comes from the export's folder (2.4, 1.7).
6. Phone-video checks, ported: pkg/video.py:192-222. They run on every video the window opens, and their warnings are shown (pkg/gui/panels/video_panel.py:191, :377-378). An ordinary 30 or 60 fps video gets "probably a re-timed 'compatible' copy ... Transfer the ORIGINAL file" (:196-201). A frame under 720 px gets "Shrimp may be only a few pixels long" (:210-212). The note "Real time comes from your stopwatch clip (fps_true in data/manifest.csv)" is printed by `check` (:219-221; pkg/cli.py:218).
7. Selftest: a shrimp-sized ellipse at 5 mm/s and 0.0324 mm/px, run through `from_tracker` with a made-up Tracker export (pkg/selftest.py:3-7, :46; pkg/synthetic.py:370-380). Its wording is last week's (1.3).
8. Port fidelity. tests/reference/ holds 10 files of the class template (1,610 lines; tests/reference/README.md). Tests that tie the package to it:
   - tests/test_port_fidelity.py (22 tests) compares the source text of ported functions with the reference (:1-26). Pinned: 9 names in pkg/video.py plus `iter_rgb_frames` and `check_video`; 2 in pkg/convert.py; 9 in pkg/tracker_io.py; `mask_center`; `from_tracker._flags`; 4 in pkg/segmenter/edgetam_convert.py (`load_edgetam` holds the cache path); `hf.load_model`; `selftest.selftest`; constants `EXTRA_PER_SHRIMP`, `MAX_SPEED_MM_S`, `MODELS`, `KEEP_FRAMES` (:44-67, :136-175).
   - tests/test_port_equivalence.py (7), tests/test_from_tracker_port.py (5), tests/test_measure_port_guard.py (3); tests/test_repo_rules.py:76-97 (hashes of the 10 files; tests/conftest.py must equal the template's; the template's 27 tests must pass); tests/slow/test_regression_pipeline.py and test_regression_reference.py. 14 test files import `shrimp` or read tests/reference.
   - Wiring: pyproject.toml:49; .gitattributes (`tests/reference/** -text`).
   - The repository has no LICENSE file (`git ls-files`). Whether the template's code may be redistributed was not verified.
9. CLAUDE.md rules tied to the class: :3 ("Questions for J"), :4 (the deadline and the order of SPEC §14), :8 (Tracker's convention), :9 ("Ported code keeps its behavior", the list of template files, tests/reference), :12 (SPEC §15), :14 (answer keys, student data, rosters). docs/DEVELOPER.md:129-142 repeats them.
10. Smaller ones: track colors are last week's overlay colors (X14; pkg/from_tracker_session.py:33-35); track ids A, A2 follow "the analysis template's convention" (pkg/tracking_ids.py:4-5); the overlay's look and encoding are last week's (pkg/overlay.py:5-7, :87, :169); the Tracker-format files keep the platform's line ends "as last week" (X11).
11. Borderline (animal-shaped but usable for any elongated object): the Head tool and `head_px`, `theta_rad`, the "core" that cuts thin parts away (`core_open_frac`, pkg/session_parts.py:194), the flags HEADGUESS and ORIENT, the body frame of radial.csv and outlines.npz.

## 3. Dish to Boundary, and optional: every place the circle is named or used

| place | what | pointer |
|---|---|---|
| session.json | `circle` (None allowed): `points_px`, `center_px`, `radius_px`, `rms_px`, `dish_mm` | pkg/session.py:151; pkg/session_parts.py:171-179 |
| session.json | `processing.dish_crop`, default true | pkg/session_parts.py:190 |
| manifest | column `dish_mm` fills the diameter box | pkg/geometry.py:389-400; pkg/gui/tools.py:312-321 |
| geometry | `fit_circle`, `CircleFit` (no class word in the names) | pkg/geometry.py:257-305 |
| panel 4 | module `dish_panel`, class `DishPanel`, title "Dish and axes", group "Dish", "Dish diameter", "Crop to dish for tracking" | pkg/gui/panels/__init__.py:19; pkg/gui/panels/dish_panel.py:27-64; pkg/gui/main_window.py:54 |
| tools | `CircleTool` ("The dish wall"), `set_dish_mm`, the drawing method `dish()` | pkg/gui/tools.py:295-390; pkg/gui/tool_items.py:139 |
| axes | default origin: the circle's center if there is one, else the frame's center; "Origin to Center" | pkg/gui/tools.py:243-245, :257-260, :271-276 |
| model input | `DISH_MARGIN`, `dish_box`, `view_box`: the square around the circle | pkg/tracking_plan.py:32, :59-94; pkg/tracking.py:50, :54, :240 |
| preview | the outline after a click uses the same `view_box` | pkg/gui/click_rules.py:167-176; pkg/gui/prompts.py:17 |
| derive | wall distances from center and radius | pkg/derive.py:116, :168-173 |
| export | shapes.csv columns `wall_dist_centroid_mm`, `wall_dist_min_mm` | pkg/schema.py:120-123; pkg/export.py:115 |
| run.log | "circle: none", or center, R, 2R, `dish_mm`; "dish crop, N px" | pkg/export_log.py:113-124, :171-174 |
| review | the circle is part of what makes the flags table stale | pkg/gui/review_table.py:314; pkg/tracking_flags.py:49 |
| synthetic | `Scene.dish`, `dish_scene`, `synth dish` | pkg/synthetic.py:85-96, :124-125, :281-306; pkg/cli.py:130, :272, :284 |
| flags | EDGE: "border of the model's input (frame, dish crop or fine crop)". It uses the crop, not the circle. CONTACT does not use the circle (no hit for "circle" in pkg/qc.py and pkg/qc_contact.py). | pkg/schema.py:181-183 |
| docs | README.md:74; docs/OUTPUTS.md:17, :59, :79, :105-106, :227, :233; SPEC.md:162-168, :209, :332 | |
| tests | tests/gui/test_dish_panel.py (100 lines with "dish"), tests/gui/test_tools.py (112), tests/test_tracking_coarse.py (84); `wall_dist` in 5 test files (15 lines) | |

What works without a circle today:
- Tracking: `view_box` gives the whole frame when `session.circle` is None, whatever `dish_crop` says (pkg/tracking_plan.py:92-94). The window's list of reasons not to start has no circle in it (pkg/gui/worker_jobs.py:186-209). Test: tests/test_tracking_coarse.py:76.
- Export: the two wall columns are empty cells (pkg/derive.py:168; pkg/schema.py:121, :123); run.log says "circle: none".
- The axes: the origin defaults to the frame's center (pkg/gui/tools.py:245).
- Every `from-tracker` session has no circle (pkg/from_tracker.py:7; pkg/from_tracker_session.py:162-174).

What assumes one, or nudges toward one:
- Panel 4's title and start hint ask for it. The panel shows "done" without a circle only when the crop box is off and the origin was placed by hand (pkg/gui/panels/dish_panel.py:140-143; test tests/gui/test_dish_panel.py:254-261). With the default `dish_crop = true` it stays "todo". This does not block tracking.
- The boundary can only be a circle. The crop is the square around it.
- shapes.csv always has the two wall columns. Their names and meanings say "dish" and "wall".
- `_crop_text` calls every coarse crop that is not the whole frame a "dish crop" (pkg/export_log.py:171-174).
- The Quickstart's step 5 and tests/test_readme.py:359-365 expect panel 4 as a step.

## 4. What a rename costs

How versions are handled today:
- session.json: `SESSION_SCHEMA_VERSION = 1` (pkg/schema.py:30). `Session.from_json` refuses every other value with `SessionVersionError` (pkg/session.py:171-173; message :57-69; the window shows it, pkg/gui/session_controller.py:164-165). There is no reader for another version and no migration step anywhere.
- Keys the tool does not know are kept in `extra` and written back (pkg/session_parts.py:20, :112-120). So if only the version check were loosened, an old `circle` or `student` key would be kept but silently ignored. A rename needs a real step: map a version 1 object to version 2 before `from_json`.
- Precedent for adding without a version step: optional keys, written only when set (decision X18, docs/PLAN.md:430; pkg/session_parts.py:40-42).
- results.npz: `RESULTS_VERSION = 1`, also strict equality (pkg/results.py:199-205). None of its keys is a class word (pkg/schema.py:266-298), so the rename does not touch it.
- CSV files have no version field. The header line is the contract (pkg/schema.py:337-339). outlines.npz has `meta.tool_version` (pkg/schema.py:308-313).

Names in saved files or in commands (a rename breaks old files, notebooks or scripts):
- session.json keys: `student`; `circle` and `circle.dish_mm`; `processing.dish_crop`; `time.manifest_path`; the values of `time.source` (`manifest`, `stopwatch`, `typed`, `tracker-export`); `calibration.tracker_fit`. `calibration.stick` and `calibration.check` are Tracker's words, not the class's.
- shapes.csv: `wall_dist_centroid_mm`, `wall_dist_min_mm`. Any notebook that reads them by name breaks. All other column names are general (pkg/schema.py:84-143); the Tracker-format columns `t, frame, x, y, pixelx, pixely` belong to that format.
- File and folder names: the run folder `<video stem>_outline_<student>/`; the folder `<model>/` of Tracker-format files; `<stem>_tracker.mp4` from `convert`. How the window treats an opened session whose folder does not follow the name rule was not verified.
- The model cache `~/.cache/shrimp-models` and `SHRIMP_MODEL_CACHE`: after a rename every installed copy converts the model again, unless the old place is still read. The function that holds the path is source-pinned (2.8).
- Command line: `--student`, the command name `from-tracker`, `--no-video`, the hidden `synth dish|closeup`.
- The settings key `manifest_path` (pkg/gui/panels/time_panel.py:37, :52).
- Default values, not keys: probe name `LED1`, stick 30 mm, tape 20 mm.

Only texts (no saved file changes), but tests pin many of them:
- Panel titles, hints, tooltips, messages: tests/gui/test_shell.py:23-40, tests/gui/test_dish_panel.py:21-22, :47-48, :261, tests/gui/test_tools.py:69, :252, tests/gui/test_panel.py:58-59, tests/gui/test_finish.py:98.
- README.md: tests/test_readme.py (32 tests). README.txt and docs/OUTPUTS.md: built from schema, and a test fails when OUTPUTS.md is stale (pkg/schema_docs.py:4-5).
- Console lines and messages inside ported functions ("shrimp" in pkg/tracker_io.py:231, pkg/from_tracker.py:67, pkg/video.py:211, pkg/selftest.py:57-60): each change needs a row in `ADAPTED` of tests/test_port_fidelity.py, or the port-fidelity tests and tests/reference are retired first.
- Module, class, fixture and test names (`dish_panel`, `DishPanel`, `dish_box`, `dish_scene`, `Shrimp`, `dish_clip` with 928 uses): mechanical, wide, no file format involved. docs/DEVELOPER.md's module map must follow.

## 5. Decisions the owner must make

1. The word for a tracked thing: object, target or animal. Today the panel is "Objects" and the session key is `tracks`, but 16 lines of window text say "animal". Also: do Head, heading, body and core stay as they are (2.11)?
2. The student name: drop it, or make it an optional label. Then what is the default run folder name, and what happens to `--student`?
3. Tracker: keep the import (`from-tracker`) and the Tracker-format export as a general feature, make the export opt-in, or drop both. Keep `convert` and the `_tracker.mp4` name? Keep the guard against folders with other .csv files?
4. Boundary: only a circle, or also a rectangle or polygon. Is the crop to the boundary on or off by default? Do the two distance columns stay under a new name (for example `boundary_dist_*`), become opt-in, or go, since students are to compute this themselves?
5. Brightness probes: keep as a general feature with a neutral default name, or remove as reaction-time help. Panel 5 is still empty.
6. Frame rate: may the file's own frame rate be a source? Does the manifest (`data/manifest.csv`, with `fps_true` and `dish_mm`) go? Does the stopwatch helper stay? Does the 100 fps warning go? Does the name `fps_true` stay?
7. The phone-video checks on every opened video: keep as an optional check with neutral wording, or remove from the window.
8. Ported code: retire tests/reference and the port-fidelity tests (and settle the missing license), or keep them as a regression baseline. This decides how freely texts in 1.3 can change, and whether the cache folder can be renamed. CLAUDE.md needs new rules either way.
9. Old files: write session schema 2 with a reader for version 1, or declare version 0.1.0 files of the one class unsupported. The same for the shapes.csv column names.
10. Units and defaults: mm only, or a unit the user chooses; the defaults of 2.3.
11. The class documents SPEC.md, docs/PLAN.md and docs/VALIDATION.md: keep as history in their own folder, or rewrite. VALIDATION's numbers are measured against last week's script.
12. The synthetic scenes and the selftest: neutral names and wording (`dish`, `closeup`, `Shrimp`, "test shrimp"), and a selftest that does not go through a made-up Tracker export.
