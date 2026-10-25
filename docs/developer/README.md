# Developer notes

For whoever changes the code. Students read [README.md](../../README.md); the files a run writes are described in [OUTPUTS.md](../OUTPUTS.md); the plan is [ROADMAP.md](../ROADMAP.md), and [decisions.md](../design/decisions.md) holds the decisions of the first build; the measured results of the real model are in [VALIDATION.md](../VALIDATION.md). The specification of version 0.1.0 is [SPEC-v0.1.md](../design/SPEC-v0.1.md).

## Where what is

The package is `outline_tracker/`. The core has no window: everything outside `gui/` runs from the command line and never imports Qt. A path of a module, such as `gui/panels/`, is relative to `outline_tracker/`; other paths are relative to the repository.

Which module holds what is in [modules.md](modules.md): one row for each module, with the first line of its docstring. The page is generated from those first lines, and a test fails when it is out of date; write it again with `uv run python tests/support/module_index.py docs/developer/modules.md`. A new module therefore starts with a docstring whose first line is a sentence of its own.

The subpackages are layers, from the bottom: `core`; `video`; `measure` and `segmenter`; `tracking` and `synthetic`; `export`; `workflows`; `gui`; `cli` (its `gui` command opens the window); and `launch.py`. A module imports modules of its own subpackage and of lower layers only, and two subpackages of one layer do not import each other. In `gui/`, a panel (a module that `PANEL_MODULES` names) imports no other panel, and no module that is not a panel imports one; a part of one panel, such as `gui/panels/review_table.py`, lies beside it. `tests/repo/test_layers.py` holds the rule; its list `EXCEPTIONS` names the imports that still break it, each with the workstream that removes it.

Panel 5 has no module (`probes_panel` in `PANEL_MODULES`): it shows its hint line alone, and the `probe` command does its work.

## Tests

The fast tests need no model and no internet. Run them after every change:

```shell
uv run pytest -m "not slow"
```

The slow tests need torch. Most of them run the real model on short synthetic clips: they take minutes, need the model on this computer, carry the mark `weights`, and are not part of CI. Never run the model on a long real video for a test.

```shell
uv run pytest -m slow
```

The slow tests without that mark need no weights and take seconds. CI runs them in every job (Ubuntu, Windows and macOS), after the fast tests, offline and with a model folder that holds nothing:

```shell
uv run pytest -m "slow and not weights" tests/slow
```

`tests/repo/test_slow_selection.py` lists them, ten today. One of the ten needs the Apple GPU and is skipped on Ubuntu and Windows (measured: macOS 10 passed; Ubuntu and Windows 9 passed, 1 skipped). A new slow test that loads the model gets the mark `weights`; one that does not is added to that list.

The tests of the window alone:

```shell
uv run pytest tests/gui -q
```

How the tests are laid out:

- The tests of a module are in the folder of `tests/` named like its subpackage (`tests/core/`, `tests/video/`, `tests/measure/`, `tests/tracking/`, `tests/segmenter/`, `tests/export/`, `tests/workflows/`, `tests/synthetic/`, `tests/cli/`); the window's tests are in `tests/gui/`, the real model's in `tests/slow/`, and the rules of the repository in `tests/repo/`.
- Shared fixtures are in `conftest.py` files, where pytest finds them by name: `tests/conftest.py` for the tests of every folder (the synthetic clips, the `window`), `tests/core/conftest.py` for the tests of the two texts that `core/schema_docs.py` builds (`schema_docs`, `readme`, `outputs`), `tests/tracking/conftest.py` for the tests of the corrections (`tracked`), `tests/gui/conftest.py` for the window's tests, `tests/slow/conftest.py` for the tests with the real model (the loaded model). Helpers that tests import by name are in the helper modules of `tests/support/` (`tests/support/helpers.py`, `tests/support/gui_helpers.py` and others), which is on pytest's `pythonpath`; `tests/support/layout_helpers.py` holds the one table of the paths that tests name, so that a move of files changes that table and no test. No test imports a fixture, and none imports a `conftest.py` (`tests/repo/test_repo_rules_test_code.py`).
- `tests/data/` holds frozen reference files: what the code gave in a run that an independent check confirmed, each with a header that says how it was made (commit, machine, system, library versions). A test never rewrites them. The command in a file's header is the one that made it: it set `OUTLINE_TRACKER_FREEZE=1`, the switch without which nothing is written, and the package had to be as committed (`tests/support/frozen_helpers.py`). The files that are there now cannot be made again at this commit: the check that confirmed them was last week's code, which has left the repository, and the tests that wrote them went with it (`docs/VALIDATION.md`, section 7, says how a freeze is repeated from an earlier commit). The golden Tracker-format files (`tests/data/tracker_format/`) are compared byte for byte only where the decoder is the one that their `HEADER.txt` names; elsewhere a run's files are checked for their fixed text and against the true centers of the clip's disks.
- The window's tests run offscreen with stand-in segmenters (`segmenter/fake.py`), so none of them imports torch. Every test gets its window from the `window` fixture, which closes it and stops the worker thread.
- A test that depends on what a thread is doing parks the stand-in on a gate (`tests/support/prompt_helpers.py`, `Gate`) and waits for a condition. No test waits by a delay (`tests/repo/test_repo_rules_test_code.py`), with one exception: the steps of a mouse drag are 20 ms apart, because pyqtgraph drops faster moves (`tests/support/gui_helpers.py`, `drag`).
- Dialogs are replaced by a recorder (`tests/support/gui_helpers.py`, `record_dialogs`), so no test waits for a click.

## Rules

The rules that every change has to keep are in [CLAUDE.md](../../CLAUDE.md), in short, and in
section 2 of [ROADMAP.md](../ROADMAP.md), in full. They are not repeated here. One more, from SPEC 12:

- Every public function says its units and its coordinate frame in its docstring. No `.py` file of the package or of the tests is over 400 lines: `tests/repo/test_file_sizes.py` fails one that is, unless its list `OVER_400` names the file with the reason.

## Adding a panel

1. Write `outline_tracker/gui/panels/<name>.py` with one function, `build(window)`, that returns the widget with the panel's controls. `PANEL_MODULES` in `gui/panels/__init__.py` maps the panel's number to the module's name; nothing else has to be edited.
2. The window gives the module what it works with: `window.controller` (the open video and its session), `window.view` (the picture: tools, clicks, graphics), `window.navigation` (the bottom bar), `window.panels` (the nine `Panel`s) and `window.menus`.
3. To change the session: change `controller.session`, then call `controller.touch()`. To show the session: listen to `controller.session_changed` and `controller.video_opened`.
4. The panel's own state and hint line: `window.panels[n - 1].set_state("todo" | "done" | "attention")` and `.set_hint(text)`. The hint line says the next step, and why the main button is off.
5. Give the panel public functions with plain arguments, and connect the buttons to them: the tests call the functions. A dialog is asked for only through `gui/dialogs.py` (`message`, `confirm`, `open_file`, `choose_folder`), from a button's own small function.
6. A part that changes what a tracking job or an export works on (the objects, the clip, the video, the run folder, the model, the results) is off while `jobs_of(window).writing()` gives a sentence, and shows that sentence as its tooltip: "Tracking is running." or "An export is running." `Jobs.writing_changed` says when. Ask this one function, never another panel's attribute.
7. No Qt object is made when the module is imported.

## The worker thread and the GUI thread

There is one worker thread per window (decision X7 in docs/design/decisions.md). It owns the loaded model and only it calls the model: an outline after a click, or one tracking job, one at a time. The same thread runs the two tasks that need no model, Export all and the listing of the flags table: `Worker.run(task, needs_model=False)` takes them whatever the model's state, so they also work while a model is still loading and when no model could be loaded.

A model is made in a thread of its own, which calls the factory, does nothing else, and ends when the factory has returned (`_Making` in `gui/worker_engine.py`; the same 64 MiB of stack as the worker thread). The worker thread takes the model over and is free until then. What keeps X7's reasons: the model before is closed before the next one is made, a second load waits until what the first one made is closed (one copy of the weights), and no thread but the worker thread ever calls a model (no concurrent use). The step before loading, which sets torch's thread limit, runs in the worker thread. A model that is made while a task runs is taken over when that task has returned, and the state stays "loading" until then.

- `gui/worker_engine.py`, `_Engine`: the object that lives in the worker thread. `_Making`: the thread that makes one model and ends.
- `gui/worker.py`, `Worker`: its handle in the GUI thread. Its methods return at once (`start`, `load`, `request_preview`, `run`, `stop`). Its signals are always emitted in the GUI thread.
- `gui/worker_jobs.py`, `Jobs`: starts `run_job` (`tracking/job.py`) in the worker thread with a copy of the session, and takes over what the job reports. `Jobs.writing()` is the one lock: it says whether, and why, the worker is busy with a task that writes files (a tracking run, or the export that panel 9 reports with `set_exporting`).

A signal of the worker or of a job is connected only to a method of an object that lives in the GUI thread, never to a lambda: a lambda would run in the worker thread.

An outline is made by the model that was asked for last. The `Worker` gives the engine a request only while its state is "ready". A request made while a model loads waits in the `Worker`, in the GUI thread, and so does one that is still awaited when another model is asked for; it is handed over when that model is ready. When a model begins to load, `gui/view/prompts.py` takes the outlines of the model before off the picture and asks for the frame shown again, and panel 7 forgets the time per frame.

Who writes which file:

| file | written by |
|---|---|
| session.json | the GUI thread only (`SessionController`). What a job changes in the session crosses as the value of a signal and is applied and saved there. A closing window writes it only when a save is due: its time tells panel 9 whether the exported files are older than the session. |
| results.npz | the worker thread while a job runs (`run_job`, `tracking/job.py`). Between jobs, the GUI thread's edits that delete records write it (Remove, and the corrections of panel 8), which is one reason why they are off during a run and during an export. |
| positions.csv and the other exported files | the worker thread during Export all (`export_all`, `export/pipeline.py`), from session.json and results.npz as they are on disk. |
| run.log | the GUI thread, when a job has ended, with the lines the job said and the trace of a failure; an export appends its own block. |
| the settings (the manifest's place, Fill) | the GUI thread (`gui/settings.py`, `settings()`). |

Cancel is a `threading.Event` that `run_job` asks before every frame. Closing the window sets the worker's `stopping`, waits for the thread, and then saves the session as the job left it.

## Adding a model

A segmenter is any object with the methods of the protocol in `segmenter/base.py`; the stand-ins in `segmenter/fake.py` are the smallest examples. The real models are the keys of `MODELS` in `segmenter/hf.py`, made by `load_segmenter(model, device)` in `segmenter/load.py`. A model that panel 7 should offer also gets its name in `MODEL_NAMES` in `gui/panels/track_panel.py`. A new model needs a slow test on a synthetic clip before it is offered, and its numbers in docs/VALIDATION.md.
