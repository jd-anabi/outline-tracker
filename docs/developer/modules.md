<!-- Generated from the docstrings of outline_tracker/ by "uv run python tests/support/module_index.py docs/developer/modules.md". Do not edit by hand. -->

# Outline Tracker: the modules

Every module of the package `outline_tracker/` with the first line of its docstring, in the order of the layers, the base first; the paths are relative to `outline_tracker/`.

## `outline_tracker/`

Outline Tracker: track objects in videos with EdgeTAM; export positions and outline shapes. Holds the version.

| module | what it holds |
|---|---|
| `__init__.py` | Outline Tracker: track objects in videos with EdgeTAM; export positions and outline shapes. Holds the version. |
| `launch.py` | What the installed command `outline-tracker` starts (SPEC 11). |

## `core/`

The base of the tool: schema, safe file writes, run facts, geometry, calibration, session and run folder.

| module | what it holds |
|---|---|
| `core/__init__.py` | The base of the tool: schema, safe file writes, run facts, geometry, calibration, session and run folder. |
| `core/csv_format.py` | Write the rows of a table as CSV text: the header line, one line for a row, a whole file (SPEC 8, X11). |
| `core/fileio.py` | Atomic file writes with the Windows lock retry, the hash that identifies a video, and relative paths. |
| `core/geometry.py` | Image <-> world transform, calibration stick, tape check, stopwatch, circle fit, frame grid, manifest. |
| `core/provenance.py` | Which tool wrote a file, and on what machine: the version line and the facts of run.log (SPEC 8.9). |
| `core/run_folder.py` | Where a run's files go (SPEC 8.1), and the check that keeps the tool out of a folder of Tracker files. |
| `core/schema.py` | The single source of truth for every output file of SPEC 8: columns, flags, file names, formats. |
| `core/schema_docs.py` | The two texts that describe a run folder, README.txt and docs/OUTPUTS.md, built from the schema tables. |
| `core/schema_prose.py` | The prose that README.txt and docs/OUTPUTS.md share, and the helpers that make it from the schema tables. |
| `core/session.py` | session.json <-> dataclasses (SPEC 8.10, schema version 1), and finding the video again (SPEC 8.1). |
| `core/session_parts.py` | The records of session.json below the whole session (SPEC 8.10, schema version 1). |
| `core/tracker_io.py` | Tracker's files and calibration, ported from the course's shrimp.segment (unchanged behavior). |

## `video/`

Videos: frames and times read safely, exact random access, a copy for Tracker, brightness probes.

| module | what it holds |
|---|---|
| `video/__init__.py` | Videos: frames and times read safely, exact random access, a copy for Tracker, brightness probes. |
| `video/convert.py` | Make a copy of a phone video that Tracker can open. Ported from the course's shrimp.convert. |
| `video/frame_source.py` | Exact random access to the frames of a video, for the display (SPEC 3.5, 6.5, 13.6; decision X4). |
| `video/probes.py` | Brightness probes: the mean color inside named rectangles on every frame (SPEC 4.6, 8.7). |
| `video/reader.py` | Reading phone videos safely. Ported from the course's shrimp.video (unchanged behavior). |

## `measure/`

Measurement: a mask to its record in results.npz, the store of those records, and what is derived from them.

| module | what it holds |
|---|---|
| `measure/__init__.py` | Measurement: a mask to its record in results.npz, the store of those records, and what is derived from them. |
| `measure/contact.py` | How close the outlines of two tracks come: the geometry behind the CONTACT flag (SPEC 9; X6). |
| `measure/derive.py` | World quantities from the stored pixel records and the current calibration (SPEC 7). |
| `measure/heading.py` | The head direction of a track, frame by frame (SPEC 7.3; decision X17). |
| `measure/mask.py` | Measuring a mask: where an object is, and its pixel-space record for results.npz (SPEC 7, 8.12). |
| `measure/outline.py` | Geometry of an outline polygon: rays, the radial profile, the hull, resampling (SPEC 7.4, 7.6, 7.7). |
| `measure/qc.py` | Quality flags per track and frame (SPEC 9), and their summary for run.log (SPEC 8.9). |
| `measure/results.py` | The store behind results.npz: the pixel-space records of every track (SPEC 8.12; X5, X15). |

## `segmenter/`

Segmenters: what turns a frame and a few clicks into one mask per object (SPEC 6.2, 12).

| module | what it holds |
|---|---|
| `segmenter/__init__.py` | Segmenters: what turns a frame and a few clicks into one mask per object (SPEC 6.2, 12). |
| `segmenter/base.py` | The segmenter protocol and its two records, `ObjectPrompt` and `MaskResult`. |
| `segmenter/edgetam_convert.py` | EdgeTAM: Meta's original checkpoint converted for Hugging Face transformers. |
| `segmenter/fake.py` | Stand-in segmenters for the tests, `ThresholdFake` and `ExactFake` (SPEC 12, 13.2): no model, no torch. |
| `segmenter/hf.py` | SAM 2.1 and EdgeTAM through Hugging Face transformers, one frame at a time; `MODELS` lists them (SPEC 6.2). |
| `segmenter/hf_prompts.py` | The prompts of a run for the Hugging Face processor: one object's points, and all objects put into a session. |
| `segmenter/load.py` | The real model's factory: `load_segmenter(model, device)` makes the Hugging Face segmenter (SPEC 6.2). |

## `tracking/`

Tracking without Qt: jobs, their plan, fine mode, the check at the start of a job, edits and corrections.

| module | what it holds |
|---|---|
| `tracking/__init__.py` | Tracking without Qt: jobs, their plan, fine mode, the check at the start of a job, edits and corrections. |
| `tracking/corrections.py` | The three corrections of SPEC 6.6, without Qt: Re-track from here, End track here, Continue as new track. |
| `tracking/edit.py` | Edits of a session and its results without Qt (SPEC 5, 6.6): what the GUI's panels call. |
| `tracking/fine.py` | Fine mode: a W x W crop that follows one object (SPEC 6.3). |
| `tracking/flags.py` | The flags table of a run folder, for the GUI's review panel (SPEC 9). |
| `tracking/guard.py` | The guard at the start of a job (SPEC 3.5, decision X8): checks that a frame is the frame that was clicked on. |
| `tracking/ids.py` | The names and colors of tracks (SPEC 3.4; decision X14). |
| `tracking/job.py` | Tracking jobs: `run_job`, from a session with prompts to the pixel-space records of results.npz (SPEC 6). |
| `tracking/plan.py` | Planning a tracking job (SPEC 6.1, 6.2): the objects tracked together, their frames, their part of the frame. |

## `synthetic/`

Synthetic test clips with exact ground truth, and the shapes and paths they are drawn from.

| module | what it holds |
|---|---|
| `synthetic/__init__.py` | Synthetic test clips with exact ground truth, and the shapes and paths they are drawn from. |
| `synthetic/clips.py` | Synthetic test clips with exact ground truth (SPEC 13.2), and last week's selftest clip. |
| `synthetic/shapes.py` | Shapes and paths of the synthetic clips: implicit functions and poses (decision X16). |

## `export/`

Export: every output file of a run folder, made from session.json and results.npz, and overlay.mp4.

| module | what it holds |
|---|---|
| `export/__init__.py` | Export: every output file of a run folder, made from session.json and results.npz, and overlay.mp4. |
| `export/log.py` | The block that an export appends to run.log (SPEC 8.9). |
| `export/overlay.py` | overlay.mp4: the clip with each track's outline, centroid dot and id drawn on it (SPEC 8.8, P0 content). |
| `export/pipeline.py` | Export: every output file of a run folder except the overlay's frames (SPEC 8.1-8.6, 8.9, 8.11). |
| `export/tables.py` | The tables of an export: positions.csv, shapes.csv, radial.csv, outlines.npz and the Tracker-format folder. |

## `workflows/`

Workflows that chain the parts: from a Tracker export to every output file, and the selftest.

| module | what it holds |
|---|---|
| `workflows/__init__.py` | Workflows that chain the parts: from a Tracker export to every output file, and the selftest. |
| `workflows/from_tracker.py` | from-tracker: last week's workflow with this week's outputs (SPEC 11). The fallback if the app slips. |
| `workflows/from_tracker_session.py` | What `from_tracker` settles before anything is tracked: fps_true, the name, the run folder, the session. |
| `workflows/selftest.py` | selftest: check the installation and time the model on this computer (SPEC 11, 13.4). |

## `gui/`

The window (SPEC 10): PySide6 and pyqtgraph are imported only in this package.

| module | what it holds |
|---|---|
| `gui/__init__.py` | The window (SPEC 10): PySide6 and pyqtgraph are imported only in this package. |
| `gui/about.py` | Help > About (SPEC 10.1): which tool this is, the versions of what it runs on, and where the model is stored. |
| `gui/app.py` | The window's entry point (SPEC 10.2): `main` imports torch first, then Qt, and starts the application. |
| `gui/dialogs.py` | The dialogs of the window (SPEC 10.2): a message, a question, and the choice of a file or of a folder. |
| `gui/estimate.py` | How long tracking will take, and how the time is worded (SPEC 6.4, 6.5, 10.1; panel 7). |
| `gui/main_window.py` | The main window (SPEC 10.1): the video, the bar above it, navigation below, a dock of nine panels. |
| `gui/menus.py` | The File and Help menus of the window (SPEC 10.1), and which of their items are off when. |
| `gui/session_controller.py` | The owner of the open video and of its session, and the one writer of session.json (SPEC 8.1, 8.10, 10.1). |
| `gui/settings.py` | `settings()` gives the application's `QSettings`, kept between sittings (the manifest's place, Fill). |
| `gui/theme.py` | The look of the window: colour tokens for light and dark, one palette and one style sheet. |
| `gui/worker.py` | The `Worker`, the GUI thread's handle of the worker thread (SPEC 5, 6.4, 10.2; decision X7). |
| `gui/worker_engine.py` | The worker thread's engine (SPEC 5, 6.4, 10.2; decision X7): the object that lives in it and calls the model. |
| `gui/worker_jobs.py` | `Jobs`: tracking in the worker thread (SPEC 6.1, 6.4, 10.2; decision X7), and the one lock, `Jobs.writing()`. |

## `gui/view/`

The picture of the window: video view, tools and clicks, what is drawn on it, bottom bar and status bar.

| module | what it holds |
|---|---|
| `gui/view/__init__.py` | The picture of the window: video view, tools and clicks, what is drawn on it, bottom bar and status bar. |
| `gui/view/click_rules.py` | The rules of a click on the video (SPEC 3.5, 5): which click is negative, which frame takes a click. |
| `gui/view/navigation.py` | The bottom bar (SPEC 3.3, 3.4, 10.1, 10.2): the slider on the frame grid, the frame buttons, play and pause. |
| `gui/view/overlays.py` | Results on the picture (SPEC 6.3, 7.3, 10.1): stored outlines, centroids, ids, head marks, the mask fill. |
| `gui/view/prompt_drawing.py` | What the point tools draw (SPEC 5, 10.1): the marks of the clicks, the outline, the id and the fine window. |
| `gui/view/prompts.py` | Clicks on objects: the point tools Positive, Negative and Head, undo, and the outline shown after a click. |
| `gui/view/status_bar.py` | The read-outs at the right of the status bar (SPEC 3.1, 3.2, 4.6, 10.1): pixel, position, gray value, device. |
| `gui/view/tool_items.py` | The graphics of the calibration tools on the picture (SPEC 10.1; the design note's video view). |
| `gui/view/tools.py` | The calibration tools of the video view (SPEC 4.2 to 4.5, 10.1, 10.2): Stick, Tape, Circle, Axes. |
| `gui/view/video_view.py` | The video view (SPEC 3.1, 10.1, 10.2): one frame, zoom and pan, and clicks in video pixels. |

## `gui/widgets/`

The widgets that the panels of the dock share: the frame of a numbered panel and its small parts.

| module | what it holds |
|---|---|
| `gui/widgets/__init__.py` | The widgets that the panels of the dock share: the frame of a numbered panel and its small parts. |
| `gui/widgets/panel_frame.py` | One numbered panel of the dock (SPEC 10.1): number, title, state, hint line, body. |
| `gui/widgets/panel_parts.py` | The small parts that the panels of the dock share, so that no panel imports another panel. |

## `gui/panels/`

The nine panels (SPEC 10.1): `PANEL_MODULES` and `build_bodies`, one module per panel, found by its name.

| module | what it holds |
|---|---|
| `gui/panels/__init__.py` | The nine panels (SPEC 10.1): `PANEL_MODULES` and `build_bodies`, one module per panel, found by its name. |
| `gui/panels/calibration_panel.py` | Panel 3, Calibration (SPEC 4.2, 4.3, 10.1): the Stick tool and its length, the scale, and the Tape check. |
| `gui/panels/dish_panel.py` | Panel 4, Dish and axes (SPEC 4.4, 4.5, 10.1): Circle, Origin to Center, Axes, "Crop to dish for tracking". |
| `gui/panels/export_panel.py` | Panel 9, Export (SPEC 8.1, 10.1, 10.2): the run folder, Export all, what it wrote, Open folder. |
| `gui/panels/objects_panel.py` | Panel 6, Objects (SPEC 5, 10.1): the table of objects, Add, Remove, mode, fine window, the point tools. |
| `gui/panels/review_corrections.py` | The three corrections of panel 8 (SPEC 6.6): Re-track from here, End track here and Continue as new track. |
| `gui/panels/review_panel.py` | Panel 8, Review and fix (SPEC 6.6, 9, 10.1): the flags table, Previous flag, Next flag and three corrections. |
| `gui/panels/review_table.py` | The flags table of panel 8: its rows, how they are shown, and how they are listed in the worker thread. |
| `gui/panels/stopwatch_dialog.py` | The Stopwatch dialog of panel 2 (SPEC 4.1): fps_true from a stopwatch that was filmed. |
| `gui/panels/time_panel.py` | Panel 2, "Time" (SPEC 2 step 4, 3.3, 4.1, 10.1): fps_true with its source, the Stopwatch, the manifest. |
| `gui/panels/track_panel.py` | Panel 7, Track (SPEC 6.1, 6.4, 10.1, 10.2): the model and device, the estimate, Track, progress and Cancel. |
| `gui/panels/video_panel.py` | Panel 1, "Student and video": the name, Open video, Open session, the file's facts, the clip. |

## `cli/`

The command line (SPEC 11): the parser in `main`, and one module for each of the larger commands.

| module | what it holds |
|---|---|
| `cli/__init__.py` | The command line (SPEC 11): the parser in `main`, and one module for each of the larger commands. |
| `cli/__main__.py` | `python -m outline_tracker.cli`: runs `main()` of `outline_tracker.cli.main`. |
| `cli/check.py` | The `check` command (SPEC 11): check a video before you analyze it, with `--seek` also its frame jumps. |
| `cli/compare_tracks.py` | The hidden `compare-tracks` command (decision X10): compare the track files of two folders, in px. |
| `cli/convert.py` | The `convert` command (SPEC 11): make a copy of a phone video that Tracker can open. |
| `cli/errors.py` | The `ERROR: ...` line that convert, check, probe, synth and compare-tracks print on stderr when they stop. |
| `cli/export.py` | The `export` command (SPEC 8, 11): write every output file of a run folder again. |
| `cli/from_tracker.py` | The `from-tracker` command (SPEC 11, decision X19): its options, and `from_tracker` behind them. |
| `cli/gui.py` | The `gui` command (SPEC 10, 11): open the window, with a video or a session if one is named. |
| `cli/main.py` | Command line entry point: `outline-tracker` and its subcommands (SPEC 11). |
| `cli/probe.py` | The `probe` command (SPEC 4.6, 8.7, 11): brightness probes only, with no model. |
| `cli/selftest.py` | The `selftest` command (SPEC 11, 13.4): its two options, and `selftest.selftest` behind them. |
| `cli/synth.py` | The hidden `synth` command: write one synthetic test clip, for checking an installation by hand. |
