# Roadmap inventory: repository layout

> Inventory for `docs/ROADMAP.md`: the repository's layout. Read-only findings at commit `f73d29f` (2026-10-07).
> File and line pointers are true for that commit. "The ledger", "the build ledger" and "the
> scratch folder" are private working notes of the first build; they are not in the repository.
> The design note of the window is `docs/design/gui_design.md`.

State read: `main` at f73d29f (2026-10-07). Paths are relative to the repo root; in the package table, to `outline_tracker/`. Line counts are `wc -l`. Import facts come from an AST scan of every `import` in the package (scratch script, nothing in the repo changed). Tests were not run. Branch `lane-j` (1 commit, not merged; read from git objects only) adds `gui/panel_parts.py` (106), `gui/prompt_drawing.py` (94), `tests/gui/test_joins.py`, `tests/gui/joins_helpers.py`, and brings `gui/prompts.py` to 384 and `gui/panels/video_panel.py` to 321 lines. Redo the affected rows after it merges.

## 1. Today

### 1.1 Top level, docs, .github (tracked files)
- `README.md` (230): for the students of the class: install, selftest, ten-step Quickstart, fallback commands, troubleshooting. Pinned by `tests/test_readme.py` and `tests/gui/test_finish.py`.
- `SPEC.md` (938): the class's specification. Holds the Thursday plan (§14, line 739) and a copy of CLAUDE.md (Appendix A, line 886).
- `CLAUDE.md` (15): rules for the coding agent of this build (deadline, "Questions for J", the template's file list).
- `CHANGELOG.md` (67), `pyproject.toml` (66), `uv.lock` (1714), `.python-version` (1), `.gitignore` (34), `.gitattributes` (2).
- `conftest.py` (26): settings for every test run; registers `tests/helpers.py` as a plugin (line 26).
- `docs/DEVELOPER.md` (179): module map, tests, rules, adding a panel or a model. `docs/OUTPUTS.md` (283): generated from the schema (its line 1). `docs/PLAN.md` (1744): the build plan. `docs/VALIDATION.md` (626): measured results, one section per build task (headings at lines 7, 87, 211, 231, 474, 536).
- `.github/workflows/tests.yml` (70): fast tests on Ubuntu and Windows. Nothing else is in `.github/`.
- Missing: LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CITATION, issue and PR templates (`ls` at the root). `docs/PLAN.md:479` lists LICENSE; task E2 (`docs/PLAN.md:1688`) is open.
- Git-ignored, so not in a clone: `.superpowers/` (`.gitignore:26`), `.worktrees/` (`.gitignore:34`), `.venv/`, `.pytest_cache/`.

### 1.2 The package: 82 files, 18,142 lines (45 flat at the top, 32 in `gui/`, 5 in `segmenter/`), with the proposed new path
"(2)" marks a move that is optional and belongs to a second step (see 3). Longer descriptions per module: `docs/DEVELOPER.md:11-96`.

| old path | lines | what it is | new path |
|---|---|---|---|
| `__init__.py` | 3 | the version | stays |
| `launch.py` | 27 | what the installed command starts; no argument opens the window | stays (entry point) |
| `cli.py` | 400 | parser, `main`, and the commands convert, check, synth, compare-tracks written inline (`cli.py:162-382`) | `cli/main.py`, plus a new 3-line `cli/__main__.py` |
| `cli_export.py` | 186 | the `export` command | `cli/export.py` |
| `cli_from_tracker.py` | 100 | the `from-tracker` command | `cli/from_tracker.py` |
| `cli_gui.py` | 58 | the `gui` command | `cli/gui.py` |
| `cli_probe.py` | 268 | the `probe` command | `cli/probe.py` |
| `cli_selftest.py` | 67 | the `selftest` command | `cli/selftest.py` |
| `schema.py` | 418 | columns, flags, file names, formats of every output file | `core/schema.py` |
| `schema_docs.py` | 419 | README.txt and docs/OUTPUTS.md built from the schema; runs with `-m` | `core/schema_docs.py` |
| `fileio.py` | 142 | atomic writes, Windows lock retry, the hash that identifies a video | `core/fileio.py` |
| `provenance.py` | 99 | tool version, commit, machine facts for run.log | `core/provenance.py` |
| `geometry.py` | 400 | world frame, stick, tape, stopwatch, circle fit, frame grid, the class's manifest | `core/geometry.py` |
| `tracker_io.py` | 362 | Tracker's files, `Calibration`, `make_plan`, `compare_tracks` (ported) | `core/tracker_io.py` (geometry imports it) |
| `session.py` | 265 | session.json to dataclasses and back | `core/session.py` |
| `session_parts.py` | 266 | the records below the whole session | `core/session_parts.py` |
| `run_folder.py` | 68 | default run folder; guard against a folder of Tracker files | `core/run_folder.py` |
| `video.py` | 323 | probe, check, sequential decode, timestamps, frame hash (partly ported) | `video/reader.py` |
| `frame_source.py` | 318 | exact random access to frames | `video/frame_source.py` |
| `convert.py` | 54 | a copy of a phone video that Tracker can open (ported) | `video/convert.py` |
| `probes.py` | 123 | brightness probes: mean color in named rectangles per frame | `video/probes.py` |
| `measure.py` | 308 | a mask to its pixel-space record | `measure/mask.py` |
| `results.py` | 285 | the store behind results.npz | `measure/results.py` |
| `derive.py` | 186 | world quantities from records and calibration | `measure/derive.py` |
| `derive_heading.py` | 140 | head direction per frame | `measure/heading.py` |
| `derive_outline.py` | 219 | outline geometry: rays, radial profile, hull, resampling | `measure/outline.py` |
| `qc.py` | 165 | quality flags and their summary | `measure/qc.py` |
| `qc_contact.py` | 157 | distance between two outlines (CONTACT) | `measure/contact.py` |
| `tracking.py` | 397 | `run_job`, the coarse runner; re-exports the edit API (`tracking.py:45-53`) | `tracking/job.py` |
| `tracking_plan.py` | 244 | which objects run together, on which frames and crop | `tracking/plan.py` |
| `tracking_fine.py` | 281 | fine mode: a crop that follows one object | `tracking/fine.py` |
| `tracking_guard.py` | 122 | frame-hash check at the start of a job | `tracking/guard.py` |
| `tracking_ids.py` | 65 | names and colors of tracks | `tracking/ids.py` |
| `tracking_edit.py` | 341 | edits of session and results without Qt | `tracking/edit.py` |
| `tracking_corrections.py` | 216 | re-track, end track, continue as new track | `tracking/corrections.py` |
| `tracking_flags.py` | 57 | the flags table of a run folder (calls derive and qc) | `tracking/flags.py` |
| `export.py` | 260 | every output file of a run folder except the overlay's frames | `export/pipeline.py` |
| `export_tables.py` | 236 | rows of the CSV files, outlines.npz, the Tracker-format folder | `export/tables.py` |
| `export_log.py` | 174 | the block an export appends to run.log | `export/log.py` |
| `overlay.py` | 198 | overlay.mp4 | `export/overlay.py` |
| `from_tracker.py` | 289 | the class's workflow from a Tracker export; also `load_segmenter`, the real model's factory (`from_tracker.py:92`) | `workflows/from_tracker.py` |
| `from_tracker_session.py` | 174 | fps, name, run folder, session for that workflow | `workflows/from_tracker_session.py` |
| `selftest.py` | 62 | installation check and timing | `workflows/selftest.py` |
| `synthetic.py` | 400 | synthetic clips with ground truth; the selftest clip | `synthetic/clips.py` |
| `synthetic_shapes.py` | 176 | shapes and paths as implicit functions | `synthetic/shapes.py` |
| `segmenter/__init__.py` (7), `base.py` (100), `hf.py` (419), `edgetam_convert.py` (181), `fake.py` (242) | 949 | protocol and records; SAM 2.1 and EdgeTAM through transformers; checkpoint conversion; stand-ins for tests | stay (a rename to `backends/` belongs to the backends work) |
| `gui/__init__.py` (11), `app.py` (49), `main_window.py` (321), `menus.py` (110), `dialogs.py` (101), `about.py` (50), `theme.py` (192) | 834 | the window's shell | stay |
| `gui/session_controller.py` (365), `estimate.py` (74) | 439 | owner of the open video and session; time estimates as text (no Qt import) | stay |
| `gui/worker.py` (363), `worker_engine.py` (152), `worker_jobs.py` (352) | 867 | the worker thread: handle, engine, tracking jobs | stay |
| `gui/panel.py` | 121 | the frame of one numbered panel | (2) `gui/widgets/panel_frame.py` |
| `gui/video_view.py` | 198 | one frame, zoom, pan, clicks in video px | (2) `gui/view/video_view.py` |
| `gui/navigation.py` | 345 | bottom bar: slider, frame buttons, play | (2) `gui/view/navigation.py` |
| `gui/status_bar.py` | 113 | read-outs of the status bar | (2) `gui/view/status_bar.py` |
| `gui/tools.py` | 390 | calibration tools: Stick, Tape, Circle, Axes | (2) `gui/view/tools.py` |
| `gui/tool_items.py` | 266 | graphics of those tools | (2) `gui/view/tool_items.py` |
| `gui/click_rules.py` | 205 | rules of a click on the video | (2) `gui/view/click_rules.py` |
| `gui/prompts.py` | 430 | point tools and the outline after a click | (2) `gui/view/prompts.py` |
| `gui/overlays.py` | 240 | stored results drawn on the picture | (2) `gui/view/overlays.py` |
| `gui/stopwatch_dialog.py` | 146 | the Stopwatch dialog of panel 2 | (2) `gui/panels/stopwatch_dialog.py` |
| `gui/review_table.py` | 379 | the table of panel 8 | (2) `gui/panels/review_table.py` |
| `gui/panels/__init__.py` (37), `video_panel.py` (403), `time_panel.py` (225), `calibration_panel.py` (310), `dish_panel.py` (151) | 1126 | `PANEL_MODULES`, `build_bodies`; panels 1 to 4 | stay (`dish_panel.py` is renamed by the generalization work, not here) |
| `gui/panels/objects_panel.py` (372), `track_panel.py` (389), `review_panel.py` (415), `export_panel.py` (400) | 1576 | panels 6 to 9; panel 5 has no module (`docs/DEVELOPER.md:98`) | stay |

### 1.3 tests/: 151 files (140 files, 36,187 lines of this tool; 11 files, 1,610 lines of reference copies)
- `tests/`: 67 test files, flat. By subject: core 8 (schema, schema_docs, geometry, fileio, session, run_folder, provenance, tracker_io); video 7 (video, video_frames, frame_source x3, convert, probes); measure 12 (measure x4, measure_port_guard, results x2, derive x2, qc x3); tracking 11 (tracking_* x9, corrections x2); segmenter 3 (fakes x2, hf_helpers 802); export 8 (export x7, overlay); workflows 5 (from_tracker x4, selftest); synthetic 2; cli 7 (cli, cli_compare x2, cli_export x2, cli_probe x2); repository 4 (repo_rules 272, readme 545, port_fidelity 679, port_equivalence 304).
- `tests/`: 13 helper modules next to them: `conftest.py` (49, the template's file, unchanged), `helpers.py` (159, fixtures, also the window's), `analytic_shapes.py` (175), `tracking_helpers.py` (253, imported by 35 files), `export_helpers.py` (245), `overlay_helpers.py` (172), `from_tracker_helpers.py` (159), `results_helpers.py` (143), `derive_helpers.py` (107), `qc_helpers.py` (101), `cli_probe_helpers.py` (87), `cli_compare_helpers.py` (83), `cli_export_helpers.py` (80).
- `tests/gui/`: 41 test files (shell and app 13, view 12, session 5, panels 9, worker 2) and 9 helper modules: `gui_helpers.py` (190, imported by 30 files), `prompt_helpers.py` (192), `session_helpers.py` (109), `track_helpers.py` (205), `calibration_helpers.py` (154), `review_helpers.py` (228), `export_panel_helpers.py` (231), `finish_helpers.py` (141), `last_controls_helpers.py` (116).
- `tests/slow/`: 8 test files (real model; `pytestmark = pytest.mark.slow` in each) and 2 helpers: `pipeline_helpers.py` (88), `memory_child.py` (87).
- `tests/reference/`: `README.md` (18), `shrimp/` 6 files (1,162 lines), `template_tests/` 4 files (430 lines).
- No `__init__.py` anywhere under `tests/` except in the reference copy. Helpers are imported by bare name (`from tracking_helpers import ...`). No two files share a basename (checked with `uniq -d`).

### 1.4 The layers as they are
All 274 intra-package import lines are absolute (`from outline_tracker.x import`); none is relative. Module-level imports have no cycle. Read each line left to right; a line imports only from lines below it, except as listed in 1.5.
```
launch -> cli -> cli_* --(inside functions)--> any module below, and gui.app
gui/*  -> the modules below (plus the exceptions of 1.5)
selftest -> from_tracker -> from_tracker_session, export, tracking, tracker_io
export -> export_tables, export_log, overlay, qc, derive, schema_docs, provenance
tracking -> tracking_edit -> tracking_flags -> qc;  tracking_corrections, tracking_fine, tracking_guard, tracking_plan, tracking_ids
synthetic -> convert, measure, tracker_io, synthetic_shapes;  segmenter.hf, segmenter.fake -> measure, segmenter.base
qc, qc_contact, derive (+ derive_heading, derive_outline) -> results -> measure -> schema  (and segmenter.base: 1.5 item 5)
session -> session_parts, geometry -> tracker_io;  run_folder, probes, frame_source, convert -> video, fileio, schema
```
Imported by the most modules (from the scan, approximate): `schema` 22, `session` 21, `results` 17, `geometry` 16, `video` 13, `measure` 12, `fileio` 11. `segmenter/fake.py` is imported by no package module, only by tests; three package modules still carry a hook for it (`tracking.py:287`, `tracking_fine.py:259`, `gui/worker_engine.py:133`).

### 1.5 Imports that go against the layers
1. Library imports command line: `from_tracker.py:33` takes `SLOW_MOTION_FPS`, `UNKNOWN_FPS` from `cli_probe.py:53-54`.
2. Window imports command line: `gui/panels/export_panel.py:46` takes `UNUSABLE`, `WARNING_PREFIX`, `one_line`, `size_text` from `cli_export.py`.
3. The real model's factory lives in the class's workflow module: `load_segmenter` (`from_tracker.py:92`), imported by `gui/app.py:39` and `gui/worker.py:108`. Model facts sit in four places: `segmenter/hf.py:53` (`MODELS`), `from_tracker.py:92`, `gui/panels/track_panel.py:68` (`MODEL_NAMES`), `selftest.py:28` (`EXTRA_PER_SHRIMP`).
4. Core geometry imports the Tracker module, one name of it private: `geometry.py:31` (`Calibration`, `_fps_from_manifest`).
5. Measurement and segmenter depend on each other by topic: `measure.py:40` imports `MaskResult` from `segmenter/base.py`; `segmenter/hf.py` and `segmenter/fake.py` import `measure` (`fake.py`: `mask_center`).
6. View code imports a panel: `gui/overlays.py:47` takes `settings` from `gui/panels/time_panel.py:49`. A dialog imports a panel: `gui/stopwatch_dialog.py:28`.
7. Panels import small widgets from other panels (eight import lines, see 5.3). The chain `export_panel -> track_panel -> overlays -> time_panel -> stopwatch_dialog -> video_panel` makes panel 9 import four other panel modules.
8. Two cycles closed by an import inside a function: `schema.py:412-418` and `schema_docs`; `tracking_edit.py:64` and `:339` with `tracking_corrections`.

## 2. What does not belong in a public repository of a tool

| item | verdict | what depends on it |
|---|---|---|
| `docs/PLAN.md` (1744) | Remove from the tree after the release tag; it stays in the git history. Before that, extract: decisions X1 to X22 with the measured test conditions (lines 407-449) to `docs/design/decisions.md`; the open points of "Raised during the work" (lines 203-360) to issues. | About 56 lines in 30 package files and 55 lines in 36 test files cite a decision id (regex count, approximate). `tests/test_tracking_guard.py:101` names the file. `docs/DEVELOPER.md:3, 134, 139` and `CLAUDE.md:3, 12` point to it. |
| `SPEC.md` (938) | Move unchanged to `docs/design/SPEC-v0.1.md`; add a short note beside it that it is the frozen first specification of a class tool. Do not delete. | Three tests read it as the source of expected values: `tests/test_schema.py:74`, `tests/test_schema_docs.py:405`, `tests/test_session.py:55`. 360 lines in 78 of the 82 package files and 387 lines in tests cite "SPEC n.m". |
| `CLAUDE.md` (15) | Rewrite: keep the rules that are general (test first, import boundaries, memory rule, no videos or weights in git, no history rewrite); drop the deadline, the SPEC §14 order, "Questions for J", the template's file list. The same rules for people go to `CONTRIBUTING.md`. | `docs/DEVELOPER.md:131`; comments in `tests/test_repo_rules.py:19, 40`, `.gitignore:1`, `tests/reference/README.md:6`. |
| `tests/reference/` (11 files) | Keep until the owner retires the rule "ported code keeps its behavior"; then remove it together with its tests in one change. Until then move it with those tests to `tests/port/`. The copies name the class (`tests/reference/shrimp/__init__.py:1`) and state no license of their own; `_edgetam.py:6-7` cites Apache-2.0 code of Hugging Face. The template's license: not verified. | Imported as `shrimp` by 10 test files (among them `tests/test_port_fidelity.py`, `tests/test_port_equivalence.py`, `tests/slow/test_regression_reference.py`, `tests/slow/test_regression_pipeline.py`). SHA-256 and "27 passed" pinned in `tests/test_repo_rules.py:27-38, 76-97`. `tests/test_readme.py:44` reads `shrimp/segment.py`. `pyproject.toml:49-50`, `.gitattributes:2`. |
| 15 `xfail(strict=True)` marks on superseded tests | Remove, on the owner's word (`CLAUDE.md` line 6 forbids deleting without it). Each has a successor named in `docs/PLAN.md:214-345`. Two of them fail only off macOS (`test_shell.py:101`, `test_from_tracker_cli.py:116`, per `docs/PLAN.md:220, 307`). | `tests/gui/test_controller.py:295`, `test_menus.py:36, 161`, `test_navigation.py:78`, `test_panel.py:95`, `test_session_panels.py:398`, `test_shell.py:101, 178`, `test_video_view.py:142`; `tests/test_export_pipeline_gaps.py:76, 126`, `test_from_tracker_cli.py:116`, `test_tracking_fine_cases.py:162`, `test_tracking_guard.py:101`, `test_tracking_job.py:269`. |
| 3 `xfail` marks on real-model limits | Keep until the open questions are answered; then turn into documented limits in the validation page. | `tests/slow/test_coarse_lowres.py:88`, `tests/slow/test_fine_mode.py:151, 156`; questions at `docs/PLAN.md:281-306`. |
| 24 test helper modules in three folders | Move 23 of them to one folder, `tests/support/`, on pytest's `pythonpath` (`tests/conftest.py` stays, see 4.12). Merge `tests/analytic_shapes.py` with `outline_tracker/synthetic_shapes.py` where they hold the same shapes (same shapes: not verified line by line). | Bare-name imports in most test files; `tests/slow/test_gui_worker_real_model.py:29` adds `tests/gui` to `sys.path` by hand. |
| `docs/VALIDATION.md` (626) | Rewrite by topic (backend, frame access, pipeline, selftest, window); drop task ids and notes "for J". | Cited by 8 slow tests and `tests/test_readme.py:378-380`. |
| `README.md`, `CHANGELOG.md` | Rewrite the wording for a general tool (other inventories). Layout: the README stays at the root. | `tests/test_readme.py` (545 lines) pins sections, commands and example output. |
| `.gitignore:15-16` (name of a private notes file), `:33-34` (worktrees of the build) | Remove both entries when the build is over. | Nothing. |
| `.superpowers/`, `.worktrees/` | Not in the repository. Nothing to do; no tracked file may point into them. | `git grep` finds no tracked reference other than `.gitignore`. |
| `tests/test_tracking_api.py` (55) | Rewrite or remove: it checks names and argument order against the build plan's text (`tests/test_tracking_api.py:1-12`). | The re-exports in `tracking.py:45-53`. |

## 3. Proposed layout
```
README.md  LICENSE  CONTRIBUTING.md  CHANGELOG.md  CLAUDE.md (general rules)  pyproject.toml  uv.lock  conftest.py
outline_tracker/
  __init__.py  launch.py
  cli/        main, __main__, export, from_tracker, gui, probe, selftest
  core/       schema, schema_docs, fileio, provenance, geometry, tracker_io, session, session_parts, run_folder
  video/      reader, frame_source, convert, probes
  measure/    mask, results, derive, heading, outline, qc, contact
  tracking/   job, plan, fine, guard, ids, edit, corrections, flags
  segmenter/  base, hf, edgetam_convert, fake
  export/     pipeline, tables, log, overlay
  workflows/  from_tracker, from_tracker_session, selftest
  synthetic/  clips, shapes
  gui/        app, main_window, menus, dialogs, about, theme, session_controller, estimate, worker*, view/, widgets/, panels/
tests/        core/ video/ measure/ tracking/ segmenter/ export/ workflows/ synthetic/ cli/ gui/ slow/ repo/ port/ support/  conftest.py
docs/         user/ (quickstart, commands, troubleshooting)  developer/ (architecture with the module index, testing, adding a panel, adding a model)
              OUTPUTS.md (generated; path kept)  validation.md  design/ (SPEC-v0.1.md, decisions.md)
```
- Allowed direction: `cli -> workflows -> export -> tracking -> measure, segmenter -> video -> core`; `gui` may import all but `cli`; `synthetic` sits beside `tracking`. Every new `__init__.py` holds a docstring only, as `gui/__init__.py` and `segmenter/__init__.py` do today: no re-exports.
- The flat layout (package folder at the root, decision X1, `docs/PLAN.md:413`) stays. A `src/` folder would add edits (`tests/test_repo_rules.py:16`, `tests/gui/test_finish.py:24`, `pyproject.toml:43`) for little gain.
- Tests, old to new: the subject groups of 1.3 become folders of the same names; `tests/gui/` and `tests/slow/` stay; `test_repo_rules.py` and `test_readme.py` go to `tests/repo/`; `test_port_fidelity.py`, `test_port_equivalence.py`, `test_measure_port_guard.py` and `reference/` go to `tests/port/`; 23 helper modules go to `tests/support/`; `tests/conftest.py` stays where it is. File names stay as they are, so basenames stay unique.
- After the move, small changes that make the layers true (each one its own commit, test first): the two fps constants of 1.5.1 down to `core/`; the four names of 1.5.2 into `export/`; `load_segmenter` into `segmenter/` (1.5.3); `Calibration` into `core/geometry.py` and the rest of `tracker_io.py` up to `workflows/` (1.5.4; the port tests of 4.8 name these by module); the shared GUI parts into one module (5.3).
- Docs, old to new: `docs/DEVELOPER.md` -> `docs/developer/*.md`; README sections "If the app does not open", "Troubleshooting", "Getting the original video off your phone" -> `docs/user/`; `docs/VALIDATION.md` -> `docs/validation.md`; `SPEC.md` -> `docs/design/SPEC-v0.1.md`; `docs/PLAN.md` section 4 -> `docs/design/decisions.md`, the rest to the git history. The Quickstart stays in the README while Help > Quickstart opens it (`gui/menus.py:28`).

## 4. What a move breaks, and how to keep it small
1. Import lines: 274 in 65 package files, 440 in 133 test files (`git grep` counts). One old-to-new table rewrites them. A line like `from outline_tracker import video` becomes `from outline_tracker.video import reader as video`, so test bodies stay unchanged.
2. Six module names become package names: `cli`, `export`, `measure`, `tracking`, `synthetic`, `video`. A module and a package of one name cannot exist together, so each is converted in the same commit.
3. Entry point: `pyproject.toml:33` (`outline_tracker.launch:main`) is unchanged if `launch.py` stays. `pyproject.toml:43` names the package folder only; that hatchling then ships the subpackages was not verified by a build. CI checks the installed command on Windows (`.github/workflows/tests.yml:59-68`).
4. `python -m` paths: `outline_tracker.cli` (`tests/test_repo_rules.py:64`, `tests/test_cli.py:178`) needs `cli/__main__.py`. `outline_tracker.schema_docs` (`tests/test_schema_docs.py:608, 618`) is also written into line 1 of `docs/OUTPUTS.md` (`schema_docs.py:375-376`), and the file is compared byte for byte (`tests/test_schema_docs.py:602`): regenerate it in the same commit. `outline_tracker.launch` (`tests/gui/test_launch.py:118`) is unchanged.
5. `tests/test_repo_rules.py` pins: `--version` prints `outline-tracker 0.1.0` and loads no Qt and no torch (62-69); the ten reference files by SHA-256, nothing added beside them, `tests/conftest.py` equal to the template's (76-85); the template's tests give exactly "27 passed" (88-97); Qt only where the first path part is `gui`, model libraries only under `segmenter` or in `gui/app.py`, `shrimp` nowhere (125-147); a file `cli.py` exists (145); no module outside `gui/` and `segmenter/` loads a heavy library when imported (207-221); no tracked video, npz or weights file and no home-folder path in any tracked file (234-250). To edit for the move: line 145. Lines 130-131 only if `gui` or `segmenter` is renamed.
6. `tests/gui/test_finish.py:128-138`: `docs/DEVELOPER.md` must name every `.py` file of the package by its path in backticks, the eight new `__init__.py` files too; line 137 expects `cli.py`, `gui/worker.py`, `gui/worker_engine.py`, `gui/panels/track_panel.py`. The check sees a name only: a stale description passes.
7. `tests/gui/test_import_order.py:128-129` expects five module names, `outline_tracker.gui.panel` among them (step 2 only).
8. `tests/test_port_fidelity.py`: module names as strings in its tables (lines 43-206); the import line of each ported test is fixed text (210-249); ported tests are found as `tests/<name>` (39, 211-320). `tests/test_measure_port_guard.py:24` names `outline_tracker.measure`. This test also stands in the way of the later renaming: it compares the source text of ported functions with the reference (lines 1-17), and pins names such as `EXTRA_PER_SHRIMP` (176) and test names with "shrimp" (239-240, 294). A reworded ported function needs a row in `ADAPTED` (136-168), or the port rule is retired first.
9. Strings: the logger name `"outline_tracker.gui.worker"` is a literal (`gui/worker_engine.py:34`; `tests/gui/test_worker.py:38`) and survives. `tests/gui/test_worker.py:348` patches by a dotted string.
10. Tests patch module objects (by grep: 15 times `video`, 7 times `tracking`, 23 times `fileio.os` or `fileio.time`). This keeps working when a file is only moved. It breaks when a module becomes a re-exporting front (the build ledger notes this for the split of `tracking_edit.py`). So: no fronts.
11. `tests/test_readme.py` pins: `README.md` at the root (43), the link `(docs/OUTPUTS.md)` (314), the reference file (44), its imports (36-40).
12. pytest (`pyproject.toml:46-50`): `pythonpath = ["tests", "tests/reference"]`, `norecursedirs` with `reference`. Helpers in `tests/gui/` resolve only because pytest puts a test file's own folder on `sys.path`. After the move: `tests/support` and the new place of the reference copy on `pythonpath`; `tests/conftest.py` stays (pinned by hash; imported by bare name in 5 files, e.g. `tests/test_tracker_io.py:11`); no other `conftest.py` below `tests/` (`conftest.py:24-25`). How pytest resolves the bare name `conftest` from subfolders: not verified.
13. CI: the workflow names no package path. `paths-ignore: ["**.md", "docs/**"]` (`tests.yml:7`) means a docs-only push runs no tests, although fast tests read `README.md`, `SPEC.md`, `docs/DEVELOPER.md` and `docs/OUTPUTS.md` (points 6 and 11, and section 2). Push a move of docs together with a code change, or drop that line.
14. Text only, not enforced: 86 lines in 48 package files and 84 lines in tests name a `.py` path in prose (grep counts). `provenance.py:36` takes the folder of its own file to find the checkout's commit; still inside the checkout after a move (not verified by a run).

Recommendation. Move in one step, with `git mv` and no behavior change, before the renaming of class-specific words.
- Before: the release is tagged, `lane-j` is merged, no branch is open. The owner decides on the 15 superseded tests (delete first, so they are not moved) and on the future of `tests/reference/` (a decision only).
- Commit 1: the package (the rows of 1.2 without "(2)"), every import line in package and tests, and the pinned strings of points 4 to 8. Nothing else: no function renamed, no assertion changed. Check: the fast suite gives the same counts as before the commit.
- Commit 2: tests into subject folders, helpers into `tests/support/`, pytest settings. Commit 3: docs. Later and optional: the "(2)" rows, after the shared GUI parts are in one module (5.3).
- Why first: a file that is only moved stays almost identical, so `git log --follow` and blame keep working; a move mixed with rewording may fall under git's similarity limit. The check is simple (same tests, same counts). The renaming, the new backends and the help texts then land at final paths, and the module index is written once. The renaming will itself rename files (`dish_panel.py`), which is easier to review in topic folders.
- Why not earlier: the move touches more than 200 files (82 package files, 133 test files with import lines) and would conflict with every open branch.

## 5. Smaller navigation aids
1. Module index. `docs/DEVELOPER.md:9-96` is kept by hand; the test checks names only (4.6). Generate the index from the first docstring line of each module, with a test that fails when it is out of date, as `docs/OUTPUTS.md` is made from `schema.py`. Each subpackage's `__init__.py` docstring lists its modules.
2. Names. Prefix families become folders (1.2). Names that mislead today: `gui/panel.py` beside `gui/panels/`; `overlay.py` (the mp4) and `gui/overlays.py` (the screen); `tracker_io.py` (the program Tracker) beside `tracking*.py`; `probes.py` and the command module `cli_probe.py`; two functions `size_text` with different output (`cli_export.py:84`, `gui/panels/video_panel.py:62`); test files named after build steps, not subjects (`tests/gui/test_finish.py`, `finish_helpers.py`, `last_controls_helpers.py`, and `test_joins.py` on `lane-j`); 55 test files cite a build task id in their text.
3. Shared GUI parts. Two different classes named `Message`: `gui/panels/calibration_panel.py:54` (a `QFrame`) and `gui/panels/video_panel.py:89` (a `QLabel`). `calibration_panel.py:29-182` also holds `NumberBox`, `button`, `group_label`, `value_label`, `row`, `column`, `follow_tool` and layout constants, imported by `dish_panel.py:19`, `export_panel.py:49`, `review_panel.py:36`, `track_panel.py:55`. `video_panel.py:42-154` holds `ElidedLabel`, `guard_wheel`, `field_rows`, imported by `time_panel.py:33`, `track_panel.py:56`, `export_panel.py:51`, `stopwatch_dialog.py:28`. `PRIMARY_HEIGHT` comes from `track_panel.py:64` (`export_panel.py:50`). The constants `SPACING`, `CONTROL_HEIGHT`, `LABEL_WIDTH`, `PRIMARY_HEIGHT` are defined again in `objects_panel.py:46-47`, `review_panel.py:44`, `main_window.py:40`, `panel.py:26`. `lane-j` moves the `video_panel` parts to `gui/panel_parts.py`; the `calibration_panel` parts are not in it. Target: one module `gui/widgets/panel_parts.py` for all of them, one `Message`, and `settings()` out of `time_panel.py:49`.
4. Package files over 380 lines (the limit is "about 400": `SPEC.md:645`, `docs/DEVELOPER.md:142`): `gui/prompts.py` 430, `segmenter/hf.py` 419, `schema_docs.py` 419, `schema.py` 418, `gui/panels/review_panel.py` 415, `gui/panels/video_panel.py` 403, `cli.py` 400, `geometry.py` 400, `synthetic.py` 400, `gui/panels/export_panel.py` 400, `tracking.py` 397, `gui/tools.py` 390, `gui/panels/track_panel.py` 389. Just under: `gui/review_table.py` 379. Natural cuts: the four inline commands out of `cli.py` (1.2); the manifest functions out of `geometry.py:349-400`; `Calibration` out of `tracker_io.py:90`.
5. Test files over 380 lines (whether the limit holds for tests is not stated): 28 files. In `tests/`: `test_hf_helpers.py` 802, `test_session.py` 770, `test_geometry.py` 751, `test_port_fidelity.py` 679, `test_schema.py` 639, `test_schema_docs.py` 639, `test_corrections.py` 577, `test_readme.py` 545, `test_synthetic.py` 467, `test_tracking_coarse.py` 407, `test_corrections_gaps.py` 392, `test_tracking_fine.py` 389, `test_cli_export.py` 386, `test_tracking_job.py` 384, `test_fileio.py` 384. In `tests/gui/`: `test_session_panels.py` 547, `test_tracking_panel.py` 523, `test_session_saving.py` 523, `test_model_choice.py` 509, `test_tools.py` 507, `test_export_panel.py` 499, `test_worker_jobs.py` 495, `test_review.py` 475, `test_prompts.py` 409, `test_worker.py` 400, `test_controller.py` 393. In `tests/slow/`: `test_real_model.py` 527, `test_regression_reference.py` 435.
6. Documents over 380 lines: `docs/PLAN.md` 1744, `SPEC.md` 938, `docs/VALIDATION.md` 626.
