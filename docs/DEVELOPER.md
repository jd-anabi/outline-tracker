# Developer notes

For whoever changes the code. Students read [README.md](../README.md); the files a run writes are described in [OUTPUTS.md](OUTPUTS.md); the plan and its open questions are in [PLAN.md](PLAN.md); the measured results of the real model are in [VALIDATION.md](VALIDATION.md). The specification is `SPEC.md` at the root.

## Where what is

The package is `outline_tracker/`. The core has no window: everything outside `gui/` runs from the command line and never imports Qt. Paths below are relative to `outline_tracker/`.

### Core

| module | what it holds |
|---|---|
| `__init__.py` | the version |
| `launch.py` | what the installed command `outline-tracker` starts |
| `cli.py` | the entry point and its subcommands |
| `cli_export.py` | the `export` command: write every file of a run folder again |
| `cli_from_tracker.py` | the `from-tracker` command and its options |
| `cli_gui.py` | the `gui` command: open the window, with a video or a session |
| `cli_probe.py` | the `probe` command: brightness probes, no model |
| `cli_selftest.py` | the `selftest` command and its two options |
| `schema.py` | the single source of every output file: columns, flags, file names, formats |
| `schema_docs.py` | README.txt and docs/OUTPUTS.md, built from the tables of `schema.py` |
| `geometry.py` | image to world transform, stick, tape check, stopwatch, circle fit, frame grid, manifest |
| `tracker_io.py` | Tracker's files and calibration, ported from last week's script |
| `fileio.py` | atomic writes with the Windows lock retry, the hash that identifies a video, relative paths |
| `session.py` | session.json to dataclasses and back; finding the video again |
| `session_parts.py` | the records of session.json below the whole session |
| `run_folder.py` | where a run's files go, and the check that keeps the tool out of a folder of Tracker files |
| `video.py` | reading phone videos safely, ported from last week's script |
| `frame_source.py` | exact random access to the frames of a video (the display) |
| `convert.py` | a copy of a phone video that Tracker can open, ported |
| `provenance.py` | which tool wrote a file, on what machine: the facts of run.log |
| `synthetic.py` | synthetic test clips with exact ground truth, and the selftest clip |
| `synthetic_shapes.py` | shapes and paths of the synthetic clips as implicit functions |
| `measure.py` | a mask to its pixel-space record for results.npz |
| `derive.py` | world quantities from the stored records and the current calibration |
| `derive_heading.py` | the head direction of a track, frame by frame |
| `derive_outline.py` | geometry of an outline polygon: rays, radial profile, hull, resampling |
| `qc.py` | quality flags per track and frame, and their summary for run.log |
| `qc_contact.py` | how close two outlines come: the geometry behind the CONTACT flag |
| `results.py` | the store behind results.npz: the pixel-space records of every track |
| `tracking.py` | tracking jobs: `run_job`, from a session with prompts to results.npz |
| `tracking_plan.py` | which objects are tracked together, on which frames, on what part of the frame |
| `tracking_fine.py` | fine mode: a square crop that follows one object |
| `tracking_guard.py` | the check at the start of a job that a frame is the frame that was clicked on |
| `tracking_ids.py` | the names and colors of tracks |
| `tracking_edit.py` | edits of a session and its results without Qt: what the panels call |
| `tracking_corrections.py` | Re-track from here, End track here, Continue as new track, without Qt |
| `tracking_flags.py` | the flags table of a run folder, for the review panel |
| `export.py` | every output file of a run folder except the overlay's frames |
| `export_tables.py` | the rows of positions.csv, shapes.csv, radial.csv, outlines.npz and the Tracker-format folder |
| `export_log.py` | the block that an export appends to run.log |
| `overlay.py` | overlay.mp4: the clip with outlines, centroid dots and ids |
| `probes.py` | brightness probes: the mean color inside named rectangles on every frame |
| `from_tracker.py` | last week's workflow with this week's outputs; `load_segmenter`, the real model's factory |
| `from_tracker_session.py` | what `from_tracker` settles before tracking: fps_true, the name, the run folder, the session |
| `selftest.py` | check the installation and time the model on this computer |
| `segmenter/__init__.py` | what a segmenter is: a frame and a few clicks to one mask per object |
| `segmenter/base.py` | the segmenter protocol and its two records, `ObjectPrompt` and `MaskResult` |
| `segmenter/hf.py` | SAM 2.1 and EdgeTAM through Hugging Face transformers, one frame at a time; `MODELS` |
| `segmenter/edgetam_convert.py` | EdgeTAM's original checkpoint converted for transformers |
| `segmenter/fake.py` | stand-in segmenters for the tests: `ThresholdFake`, `ExactFake`; no torch |

### The window

| module | what it holds |
|---|---|
| `gui/__init__.py` | the package of the window: the only place that imports PySide6 and pyqtgraph |
| `gui/app.py` | the entry point `main`: imports torch first, then Qt, then makes the application |
| `gui/main_window.py` | the window: the video, the bar above it, the bottom bar, the dock with nine panels |
| `gui/panel.py` | one numbered panel: number, title, state, hint line, body |
| `gui/theme.py` | color tokens for light and dark, one palette, one style sheet |
| `gui/menus.py` | the File and Help menus, and which items are off when |
| `gui/dialogs.py` | every dialog: a message, a question before something that cannot be undone, file and folder choice |
| `gui/about.py` | Help > About: the versions |
| `gui/session_controller.py` | the owner of the open video and its session; the one writer of session.json |
| `gui/video_view.py` | the video view: one frame, zoom, pan, clicks in video pixels |
| `gui/navigation.py` | the bottom bar: the slider on the frame grid, the frame buttons, play and pause |
| `gui/status_bar.py` | the read-outs of the status bar: pixel, position, gray value, device |
| `gui/tools.py` | the calibration tools of the view: Stick, Tape, Circle, Axes |
| `gui/tool_items.py` | the graphics of those tools on the picture |
| `gui/click_rules.py` | the rules of a click on the video: which click is negative, which frame takes a click |
| `gui/prompts.py` | the point tools Positive, Negative, Head, undo, and the outline shown after a click |
| `gui/overlays.py` | results on the picture: stored outlines, centroids, ids, head marks, the mask fill |
| `gui/estimate.py` | how long tracking will take, and how the time is worded |
| `gui/stopwatch_dialog.py` | the Stopwatch dialog of panel 2 |
| `gui/worker.py` | the `Worker`: the GUI thread's handle of the worker thread |
| `gui/worker_engine.py` | the engine: the object that lives in the worker thread and calls the model |
| `gui/worker_jobs.py` | `Jobs`: tracking in the worker thread, and what a job reports back |
| `gui/panels/__init__.py` | `PANEL_MODULES` and `build_bodies`: one module per panel, found by its name |
| `gui/panels/video_panel.py` | panel 1: the name, Open video, Open session, the file's facts, the clip |
| `gui/panels/time_panel.py` | panel 2: fps_true with its source, Stopwatch…, the manifest; `settings()` |
| `gui/panels/calibration_panel.py` | panel 3: Stick and its length, the scale, the Tape check |
| `gui/panels/dish_panel.py` | panel 4: Circle, Origin to Center, Axes, "Crop to dish for tracking" |
| `gui/panels/objects_panel.py` | panel 6: the table of objects, Add, Remove, mode, fine window, the point tools |
| `gui/panels/track_panel.py` | panel 7: model, device, the estimate, Track, progress, Cancel |

Panel 8 (`gui/panels/review_panel.py`, with its table in `gui/panels/review_table.py`) and panel 9 (`gui/panels/export_panel.py`) are tasks of their own. Panel 5 has no module (`probes_panel` in `PANEL_MODULES`): it shows its hint line alone, and the `probe` command does its work.

## Tests

The fast tests need no model and no internet. Run them after every change:

```shell
uv run pytest -m "not slow"
```

The slow tests run the real model on short synthetic clips. They take minutes, need the model on this computer, and are not part of CI. Never run the model on a long real video for a test.

```shell
uv run pytest -m slow
```

The tests of the window alone:

```shell
uv run pytest tests/gui -q
```

How the tests are laid out:

- One test file per module in `tests/`; the window's tests in `tests/gui/`; the real model's in `tests/slow/`.
- `tests/conftest.py` is last week's file, unchanged. Shared fixtures are in `tests/helpers.py`. No other `conftest.py` below `tests/`.
- `tests/reference/` holds last week's code, unmodified. The package never imports it.
- The window's tests run offscreen with stand-in segmenters (`segmenter/fake.py`), so none of them imports torch. Every test gets its window from the `window` fixture, which closes it and stops the worker thread.
- A test that depends on what a thread is doing parks the stand-in on a gate (`tests/gui/prompt_helpers.py`, `Gate`) and waits for a condition. No test sleeps.
- Dialogs are replaced by a recorder (`tests/gui/gui_helpers.py`, `record_dialogs`), so no test waits for a click.

## Rules

These are the rules of `CLAUDE.md` that every change has to keep, and at the end two of SPEC 12.

- Test first. Expected values come from geometry, analytic shapes or synthetic ground truth, never from running the code and copying what it gives.
- Never weaken, skip or delete a test to make it pass. A test that seems wrong is marked `xfail(strict=True, reason=...)` and raised under "Questions for J" in docs/PLAN.md.
- Conventions (SPEC 3): pixel coordinates are Tracker's, with pixel centers at +0.5; world units are mm in the session's axes, y up; `t_s = frame / fps_true`; frames live on the clip's grid. Units go in column names.
- Ported code keeps its behavior.
- Nothing outside `outline_tracker/gui` imports Qt. torch and transformers are imported only inside `outline_tracker/segmenter`, and only when a model is loaded. The one exception is `gui/app.py`, which imports torch before PySide6: on Windows torch does not load after Qt.
- Never load a whole video into memory. No full-frame float array per object lives longer than the current frame.
- No new dependency without asking in docs/PLAN.md first.
- Never commit videos, results.npz or model weights. Never force-push.
- The repository is public: no answer keys, no student data, no rosters, no personal paths.
- Every public function says its units and its coordinate frame in its docstring. Split a file that grows past about 400 lines.

## Adding a panel

1. Write `outline_tracker/gui/panels/<name>.py` with one function, `build(window)`, that returns the widget with the panel's controls. `PANEL_MODULES` in `gui/panels/__init__.py` maps the panel's number to the module's name; nothing else has to be edited.
2. The window gives the module what it works with: `window.controller` (the open video and its session), `window.view` (the picture: tools, clicks, graphics), `window.navigation` (the bottom bar), `window.panels` (the nine `Panel`s) and `window.menus`.
3. To change the session: change `controller.session`, then call `controller.touch()`. To show the session: listen to `controller.session_changed` and `controller.video_opened`.
4. The panel's own state and hint line: `window.panels[n - 1].set_state("todo" | "done" | "attention")` and `.set_hint(text)`. The hint line says the next step, and why the main button is off.
5. Give the panel public functions with plain arguments, and connect the buttons to them: the tests call the functions. A dialog is asked for only through `gui/dialogs.py` (`message`, `confirm`, `open_file`, `choose_folder`), from a button's own small function.
6. A part that changes what a tracking job works on (the objects, the clip, the video, the model) is off while `jobs_of(window).running` is true, and says so in its tooltip. `Jobs.started` and `Jobs.finished` say when.
7. No Qt object is made when the module is imported.

## The worker thread and the GUI thread

There is one worker thread per window. The model is loaded in it and only it calls the model: an outline after a click, or one tracking job, one at a time.

- `gui/worker_engine.py`, `_Engine`: the object that lives in the worker thread.
- `gui/worker.py`, `Worker`: its handle in the GUI thread. Its methods return at once (`start`, `load`, `request_preview`, `run`, `stop`). Its signals are always emitted in the GUI thread.
- `gui/worker_jobs.py`, `Jobs`: starts `tracking.run_job` in the worker thread with a copy of the session, and takes over what the job reports.

A signal of the worker or of a job is connected only to a method of an object that lives in the GUI thread, never to a lambda: a lambda would run in the worker thread.

An outline is made by the model that was asked for last. The `Worker` gives the engine a request only while its state is "ready". A request made while a model loads waits in the `Worker`, in the GUI thread, and so does one that is still awaited when another model is asked for; it is handed over when that model is ready. When a model begins to load, `gui/prompts.py` takes the outlines of the model before off the picture and asks for the frame shown again, and panel 7 forgets the time per frame.

Who writes which file:

| file | written by |
|---|---|
| session.json | the GUI thread only (`SessionController`). What a job changes in the session crosses as the value of a signal and is applied and saved there. |
| results.npz | the worker thread while a job runs (`tracking.run_job`). Between jobs, the GUI thread's edits that delete records write it (Remove), which is one reason why they are off during a run. |
| run.log | the GUI thread, when a job has ended, with the lines the job said and the trace of a failure; an export appends its own block. |
| the settings (the manifest's place, Fill) | the GUI thread (`gui/panels/time_panel.py`, `settings()`). |

Cancel is a `threading.Event` that `run_job` asks before every frame. Closing the window sets the worker's `stopping`, waits for the thread, and then saves the session as the job left it.

## Adding a model

A segmenter is any object with the methods of the protocol in `segmenter/base.py`; the stand-ins in `segmenter/fake.py` are the smallest examples. The real models are the keys of `MODELS` in `segmenter/hf.py`, made by `from_tracker.load_segmenter(model, device)`. A model that panel 7 should offer also gets its name in `MODEL_NAMES` in `gui/panels/track_panel.py`. A new model needs a slow test on a synthetic clip before it is offered, and its numbers in docs/VALIDATION.md.
