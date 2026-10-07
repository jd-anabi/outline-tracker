# Roadmap inventory: open-source readiness, and what the plan left unbuilt

> Inventory for `docs/ROADMAP.md`: open-source readiness. Read-only findings at commit `f73d29f` (2026-10-07).
> File and line pointers are true for that commit. "The ledger", "the build ledger" and "the
> scratch folder" are private working notes of the first build; they are not in the repository.
> The design note of the window is `docs/design/gui_design.md`.

State read: `main` at `f73d29f` (2026-10-07). Read only; nothing was run except greps, `gh` reads, two web reads (PyPI, Hugging Face Hub API) and metadata reads from `.venv`. Paths are relative to the repository. "Not verified" marks what was not checked. Section 3 lists options; it decides nothing.

## 1. What exists and what is missing

### 1.1 Project files

| item | state today | pointer |
|---|---|---|
| LICENSE | Missing. GitHub reports no license. Open question 12: default Apache-2.0 with a NOTICE, copyright holder "The outline-tracker authors" unless the owner names one. SPEC calls it the owner's choice. | `git ls-files`; `gh repo view` (licenseInfo null); docs/PLAN.md:187-189; SPEC.md:842 |
| NOTICE / third-party credits | Missing. The EdgeTAM key table is taken from transformers' conversion script (Apache-2.0); the credit is only in a docstring. | outline_tracker/segmenter/edgetam_convert.py:5-8 |
| CONTRIBUTING | Missing. docs/DEVELOPER.md has a module map, test commands, rules, "Adding a panel", "Adding a model". Its rules point to CLAUDE.md and to "Questions for J". | docs/DEVELOPER.md:100-152, 177-179 |
| CODE_OF_CONDUCT, SECURITY, CITATION.cff, issue forms, PR template, CODEOWNERS, Dependabot or Renovate | All missing. `.github/` holds one file, the workflow. README has no "how to cite" although SPEC 16 asks for it. | `git ls-files`; SPEC.md:854; grep "cite" in README.md: none |
| CHANGELOG.md | Exists, 67 lines. One section, "0.1.0 (not released yet)", one "Added" list in build order, worded for the class ("last week's"). No dates, no links. Says versions are tags `vX.Y.Z`. | CHANGELOG.md:1-8 |
| Package metadata | Only name, version, description, readme, requires-python, dependencies, one script. No license, authors, classifiers, urls, keywords. | pyproject.toml:1-33 |
| Build documents in the repo | SPEC.md (938 lines), CLAUDE.md, docs/PLAN.md (1,744 lines) are tracked. They describe the class and the two-day build. | `git ls-files` |
| tests/reference | Byte copies of ten files of the course template. That template's license: not verified. | tests/reference/README.md:1-10 |
| GitHub page | Private. No description, topics or homepage. Issues on, Discussions off, Wiki off. | `gh repo view` (2026-10-07) |

### 1.2 Version, releases, install

- The version is written in two places, both `0.1.0`: pyproject.toml:3 and outline_tracker/__init__.py:3. One test asserts the literal `0.1.0` (tests/test_repo_rules.py:66). Whether a test holds the two equal: not verified.
- `--version`, run.log and outlines.npz carry `outline-tracker 0.1.0 (commit abc1234)`. The commit comes from `direct_url.json` (git install) or from a git checkout; otherwise it prints `(commit unknown)` (outline_tracker/provenance.py:3-15, 47-52). A wheel from a package index would print "unknown".
- No tag, locally or on origin. No GitHub release (`git tag -l`, `git ls-remote --tags origin`, `gh release list`: all empty).
- Install today: `uv tool install git+https://github.com/jd-anabi/outline-tracker` (README.md:12). It names no tag and no Python version, so it installs the head of `main`. SPEC and plan use `--python 3.12 ...@v0.1.0` (SPEC.md:828; docs/PLAN.md:1702). The user needs `uv` and `git`; while the repo is private, also a GitHub sign-in (docs/PLAN.md:1602-1611).
- Package index: `https://pypi.org/pypi/outline-tracker/json` answered 404 on 2026-10-07, so it is not on PyPI and the name was free then. conda-forge: not verified.
- Update is a reinstall with `--force` (README.md:27-31). The tool has no update notice: the only URL in the package is the Help link (outline_tracker/gui/menus.py:28). Uninstall (README.md:33-37) leaves the model folders behind; the README does not say so. They are `~/.cache/shrimp-models/edgetam` and the Hugging Face cache (outline_tracker/gui/about.py:36-41).

### 1.3 Dependencies, Python versions, platforms

- 14 direct dependencies are pinned with `==`; torch and torchvision have a range with the tested version as upper bound (pyproject.toml:11-29). The stated reason: `uv tool install` ignores uv.lock, so the pins are what users get (pyproject.toml:8-10).
- uv.lock (63 packages) is used by CI and developers only (`uv sync --locked`, tests.yml:38, 53).
- What the owner needs to weigh for a published tool: exact pins give every user the tested set; they hold back security fixes until a new release; they clash if someone installs into a shared environment (a tool environment from `uv tool` or `pipx` avoids that); indirect dependencies still float (docs/PLAN.md:184-186 says two changed in one day). Alternatives named in the plan: `--exclude-newer` in the install command (question 11). Others: ranges plus a published constraints file; frozen app bundles.
- Python: `requires-python = ">=3.11"` (pyproject.toml:6); `.python-version` is 3.12; CI runs 3.12 only (tests.yml:35, 50). 3.11, 3.13 and 3.14 are untested; the lock has markers for them (uv.lock:4-17).
- Platforms the README names: Windows 11 and Macs with an Apple chip; Intel Macs are not supported (README.md:5). Linux is not named for users; CI runs there with CPU-only torch (pyproject.toml:58-66). What a Linux user with an NVIDIA card gets from the install command: not verified. `--device cuda` exists (outline_tracker/segmenter/hf.py:242-243; gui/panels/track_panel.py:69, 103) and has never been run (docs/VALIDATION.md:530).
- macOS 13: torch 2.14.1 has wheels only for macOS 14 and later; the range lets a macOS 13 laptop fall back to 2.11.0 (pyproject.toml:21-25; docs/PLAN.md:149-153). No test ran on macOS 13: not verified that it works.
- Licenses of the dependencies, read from the installed metadata in `.venv`: PySide6-Essentials is `LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only`; torch, transformers, timm, huggingface_hub, safetensors, OpenCV are Apache-2.0; numpy, scipy, pandas, scikit-image, torchvision, imageio-ffmpeg are BSD; pyqtgraph is MIT. imageio-ffmpeg ships an ffmpeg program; that program's license: not verified. Both points matter for an app bundle.

### 1.4 CI and quality gates

- One workflow, two jobs: `ubuntu-latest` and `windows-latest`, Python 3.12, uv 0.12.23, `uv sync --locked`, then `uv run pytest -m "not slow"` with Qt offscreen (.github/workflows/tests.yml:19-40, 42-70). The Windows job also checks the import order torch then Qt, the entry point, and the users' install path `uv tool install --python 3.12 .` (tests.yml:54-68).
- It runs on a push to `main` (not for `.md` or `docs/`), on every pull request, and by hand (tests.yml:4-9). The token is read-only (tests.yml:11-12).
- The last run on `main` was green (`gh run list`, 2026-10-07 22:15 UTC, 10 min 33 s). While the repo is private, minutes are metered and Windows counts double (docs/PLAN.md:179-183).
- Not in CI: a macOS job; any real-model test (tests.yml:3); lint; format check; type check; coverage; a wheel or sdist build; a Python version matrix; a scheduled run; a release or publish workflow; a dependency audit. Actions are pinned by tag, not by commit hash (tests.yml:27, 32).
- pyproject.toml has no section for ruff, black, mypy, pyright or coverage. The dev group is pytest and pytest-qt only (pyproject.toml:36). There is no `.pre-commit-config.yaml`.
- Gates that exist as tests: import boundaries (no Qt outside `gui`, torch only in `segmenter`), "the core loads no torch", "no personal paths or big files", "reference copies unmodified" (tests/test_repo_rules.py:76, 143, 207, 234); the README is checked against the window's texts (tests/test_readme.py; details not verified).
- 23 lines with `xfail` in tests outside tests/reference (grep). Most wait for the owner's "delete it" (section 2.4).

### 1.5 Documentation, pictures, example data

- README.md (230 lines) speaks to the class: "you have both from last week" (line 5), "send the whole text of the window to your instructor" (55), a "last week / this week" table (97-107), `data/manifest.csv` (109). It has no picture, no badge, no license line, no citation, no link for contributors.
- SPEC 16 asks the README for "the three things that go wrong" and "how to cite SAM 2 and EdgeTAM" (SPEC.md:852-854). Neither is there.
- docs/: DEVELOPER.md (179 lines), OUTPUTS.md (283, the contract for the output files), VALIDATION.md (626, measured results), PLAN.md. No documentation site (no mkdocs.yml, no Sphinx conf). No screenshots folder. No image file is tracked at all (`git ls-files`).
- Help in the app: Help > Quickstart opens `https://github.com/jd-anabi/outline-tracker#quickstart` in the browser (outline_tracker/gui/menus.py:28, 107-110). That fails while the repo is private and needs the internet. Help > About lists versions and the model folders (outline_tracker/gui/about.py:27-41).
- 50 `setToolTip` calls in `outline_tracker/gui`; no `setWhatsThis`, no accessible names, no tab order calls (grep: 0).
- Example data: no video in the repo (.gitignore:2-13). A hidden command writes a synthetic clip: `outline-tracker synth {dish,closeup} OUT.mp4` (outline_tracker/cli.py:125-136). It is not in `--help` and not in the README. No application icon (grep `QIcon`, `setWindowIcon`: none).

### 1.6 The model: what is downloaded, from where, where it is kept

- EdgeTAM (default): `hf_hub_download("facebook/EdgeTAM", "edgetam.pt")`, converted once, saved to `~/.cache/shrimp-models/edgetam`, or `$SHRIMP_MODEL_CACHE/edgetam` (outline_tracker/segmenter/edgetam_convert.py:161-178). Folder and variable carry the course's name.
- SAM 2.1 tiny and small: `Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny" | "...-small")`, kept in the Hugging Face cache (outline_tracker/segmenter/hf.py:53-57, 99-101).
- Size: about 56 MB for EdgeTAM (README.md:45). The first run needs the internet. EdgeTAM also fetches one small settings file of its backbone even when the weights are on disk (hf.py:72-74).
- Licenses: the Hub's model cards tag all three repositories `apache-2.0`, not gated (Hub API, read 2026-10-07). The license texts were not read. SPEC says Apache-2.0 "matches EdgeTAM and transformers" (SPEC.md:842).
- Trust points for a SECURITY policy. (a) Neither download pins a revision (edgetam_convert.py:167; hf.py:101). (b) The downloaded `.pt` is read with `torch.load(..., weights_only=False)` (edgetam_convert.py:168); that can run code from the file. (c) The SHA-256 of the weights is written into every run record (hf.py:127-130, 236) but is compared with no expected value (grep: none).
- No offline switch and no setting for the model folder (grep `HF_HUB_OFFLINE`, `local_files_only`: none). Whether the window shows download progress: not verified. Every `OSError` while loading is reworded to "the first run needs internet once" (hf.py:69-90). A full disk or a permission error then reads as a network problem.

### 1.7 Errors, logs, settings, crashes

- Command line: an error is one `ERROR: ...` line on stderr, without a traceback (outline_tracker/cli.py:157-159; cli_export.py:126). There is no `--debug` or `--verbose` flag to get the trace (grep).
- A failed tracking run reports the reason and the traceback through the job's `log` callback (outline_tracker/tracking.py:326-329). For `from-tracker` that callback is `print` (from_tracker.py:107, 201). Whether the trace also reaches run.log there: not verified.
- Window: a failure in the worker is logged with its trace to the Python logger `outline_tracker.gui.worker`, which writes to stderr (gui/worker_engine.py:33-34, 67-71). The newest 20 traces are kept (gui/worker.py:63, 201-202) and appended to `<run folder>/run.log` under "Errors noted in the window" (gui/worker_jobs.py:314-330). Dialogs show plain text and no trace (gui/dialogs.py:25-32; worker_engine.py:60-64).
- Gaps. (a) There is no log file outside a run folder: a failure before a video is open, such as a model that does not load, is on stderr only. (b) A user who starts the app without a terminal sees no stderr. (c) No `sys.excepthook`, no Qt message handler, no `faulthandler` in the app (grep: none; pytest alone sets `faulthandler_timeout`, pyproject.toml:53). What happens to an exception in a GUI-thread slot: not verified by a run.
- What a user can send today: `run.log`. Each export block holds the tool version and commit, the machine, library versions, device, model, video, calibration, runs, corrections, the QC summary and the file list (outline_tracker/export_log.py:29-67). It holds no user or computer name by design (provenance.py:51-54), but the run folder's name holds the name the user typed.
- There is no diagnostics command or "copy report" button. `--version` prints one line; About shows versions (gui/about.py). Logging: one `logging` logger in the whole package (gui/worker_engine.py:34). Everything else prints. No levels, no file handler.
- Settings: an INI file through `QSettings("outline-tracker", "outline-tracker")` (gui/panels/time_panel.py:49-52). Two keys: the manifest's place (time_panel.py:126) and Fill (gui/overlays.py:149, 176). No settings dialog. Model and device are chosen in panel 7 (CHANGELOG.md:65-66).
- File versions: `session.json` has `schema_version` 1 (outline_tracker/schema.py:30) and `results.npz` a format version (results.py:40-53, 199-205). A file with another version is refused with a message (session.py:171-173). There is no migration code (grep "migrat": none).

## 2. Features of the spec and the plan that are not built

### 2.1 P1 = Phase D (docs/PLAN.md:1648-1683; every box is unticked)

| task | what it is | what exists |
|---|---|---|
| D1 GUI probe tool | draw a named box, Measure brightness in the window | Panel 5 has no module and shows its hint line only (gui/panels/__init__.py:18-19; docs/DEVELOPER.md:98). The core and the `probe` command exist (probes.py, cli_probe.py). |
| D2 Live view | the latest tracked frame, at most twice a second | Not built. The job's `frame_result` callback does nothing in the window (gui/worker_jobs.py:105). Outlines show when the user steps (CHANGELOG.md:51-53). |
| D3 Draggable handles | move stick, tape, circle points, origin by dragging | Not built (grep: no movable items). Click to place, with Redo (gui/tools.py:119, 337). |
| D4 Flag timeline strip | colored marks under the slider | Only an empty 6 px slot (gui/navigation.py:3, 29, 125-126). |
| D5 Overlay extras | trail, head tick, circle, axes, scale bar, full resolution, zoom video per fine track | Not built. overlay.py holds the "P0 content": outline, dot, id, time stamp, 960 px wide (overlay.py:1-9). |
| D6 Calibration export and import | `calibration.json` shared by a group | Not built. One mention in generated text (schema_docs.py:153). |
| D7 Resume after cancel | continue from the first untracked frame with an automatic click | Not built (grep "resume": none). "Re-track from here" by hand exists (gui/panels/review_panel.py). |
| D8 `run SESSION.json` | all pending runs without a window | Not built: no such subcommand (cli.py:65-153). |
| D9 GUI "Convert for tracking" | convert a `.MOV` from the window | Not built. The window says to run `outline-tracker convert` (gui/session_controller.py:43). |
| D10 Full flow test and screenshots | GUI test through a `CONTACT` flag; `scripts/screenshots.py` | No `scripts/`, no `docs/screenshots/`. The P0 smoke test exists (tests/gui/test_smoke.py). Whether another test covers the full flow: not verified. |
| D11 `selftest --fine` | selftest with the close-up clip | Not built (no `--fine` in cli_selftest.py). |
| D12 docs/VALIDATION.md | measured results | Exists: six sections, run times, peak memory 1.43 GB for 10 objects (docs/VALIDATION.md:428). The box is unticked. "The `shape_ok` threshold and why": not verified. |

### 2.2 Phase E, release (docs/PLAN.md:1687-1703; every box is unticked)

- E1 Fixes from go/no-go 2: the owner reports the Mac part works. A report from a real Windows screen is not in the plan (docs/PLAN.md:263-268).
- E2 README, CHANGELOG, LICENSE: README and CHANGELOG exist. Missing: LICENSE, NOTICE, how to cite, "the three things that go wrong", the macOS version note.
- E3 Confirm pins and tag: pins are set. The wheel build and install check, and the tag `v0.1.0`, are not done.
- E4 Public install check: not done; the repo is private.

### 2.3 P2 (SPEC.md:786-793): none is built

Backward tracking (runs are forward only, SPEC.md:207). Box prompts (SPEC.md:193; no `input_boxes` in the code). Undo beyond clicks (only `undo_prompt`: tracking_edit.py:286; gui/prompts.py:132-134). Image-sequence input such as TIFF stacks. `.trk` import (only Tracker's exported text is read: tracker_io.py). A probe plot in the window (no plot widget in `gui`). A batch UI for many videos. macOS CI. A standalone installer.

### 2.4 Open questions for the owner

From docs/PLAN.md section 2 (lines 128-201):

- Q1. Folder: settled (the working copy is outside the synced folder). Q2, Q3, Q9: past.
- Q4. macOS 13: the range is in use. Open as "which macOS is the minimum for the general tool".
- Q5. `flags` note: information for the class's analysis template; no decision. Q6. `from-tracker` folder and `--student`: default in use. The naming is the class's.
- Q7. torchvision listed: in use. Q8. Exact pins from the first commit: in use; reopens for a published tool (1.3).
- Q10. Actions minutes: still metered, the repo is private. Q11. `--exclude-newer` in the install command: not used; indirect dependencies float.
- Q12. License: open. No LICENSE file.
- Q13. Go/no-go 2 inputs: the by-hand run on Windows is not reported. Q14. Cutting GUI items: no longer needed; all were built (CHANGELOG.md:57-66).
- Q15. Two readings of the spec (X17: one axis jump flags one frame; long shapes always take the core fallback): open.

From "Raised during the work" (docs/PLAN.md:207-358), still open:

- Fine mode for a lost object: the default says the README carries this advice; the Quickstart on `main` does not (README.md:77).
- Superseded tests marked `xfail(strict=True)` that wait for "delete them": C8b 1, the entry of Wed 14:25 2 (one on Linux only), C8a 2, C2 2, C3 1, C1 1, export with a gap 2, B1 1 (on Linux and Windows only), A16 2, A15 1 (docs/PLAN.md:214-262, 269-280, 307-316, 324-330, 338-344).
- C2: a run folder that already holds a session is not opened by itself. Alternative: offer "Open that session" (docs/PLAN.md:246-249).
- B5 fine mode: is the solidity limit 0.02 for the absolute value or for its variation? Two slow tests are xfail (docs/PLAN.md:281-295).
- B5 small objects: should the size checks use the largest piece of the mask? That changes SPEC 7.8. One slow test is xfail (docs/PLAN.md:296-306).
- Track colors: three pairs are hard to tell apart with red-green color blindness; a safer list is ready; the default keeps X14 (docs/PLAN.md:317-323).
- A16: the stored fine window cannot tell "auto" from "set by hand" (docs/PLAN.md:331-336). An Auto button exists (gui/panels/objects_panel.py:123-125). A07, file length: done in the code (session.py is 265 lines, schema.py 418), but the entry is still listed (docs/PLAN.md:346-349).
- Go/no-go 2 left out the comparison with last week's real outputs (docs/PLAN.md:1536-1538).

## 3. Candidates for the brainstorm

Size: small = up to a day of agent work with review; medium = a few days; large = a week or more, or it needs accounts, money or the owner's hands. "Today" points to section 1 or 2.

### Stability

| item | value | size | today |
|---|---|---|---|
| Crash handler and an app log file outside the run folder | failures before a video is open leave a record | small to medium | 1.7 gaps |
| Diagnostics bundle (a command and a Help item: versions, devices, settings, last log, no video) | one file to attach to a bug report | small to medium | 1.7 |
| Session and results migration between versions | old run folders stay readable after a schema change (the generalization will change names) | medium | 1.7: refusal only |
| Model download hardening: pinned revisions, hash check, no `weights_only=False`; an offline mode and correct download error texts | a stranger can trust the first run; labs without open internet | small to medium | 1.6 |
| By-hand pass on Windows with the real model; a long real video on each system | closes the largest known gaps | large (owner's hands) | section 4 |
| Resume after cancel or crash (D7) | long runs survive an interruption | medium | 2.1 |
| Resolve the xfail tests and the two B5 questions | a clean test report for contributors | small | 2.4 |

### Usability

| item | value | size | today |
|---|---|---|---|
| Draggable handles for the calibration tools (D3) | fix a point without redoing the tool | medium | 2.1 |
| Undo and redo beyond clicks (calibration, objects, corrections) | safe exploration; some edits delete results | large | 2.3; only `undo_prompt` |
| Settings dialog (default model and device, model folder, colors, overlay options) | choices persist and are findable | medium | 1.7: two keys |
| Keyboard access: tab order, accessible names, a shortcut list | usable without a mouse and with a screen reader | medium | 1.5: 0 accessible names |
| Color-blind safe track colors | one list to change; no stored result changes | small | 2.4 |
| Live view (D2) and flag timeline (D4) | see a run going wrong early; find flags at a glance | small to medium each | 2.1 |
| Convert button (D9), probe panel (D1) | no terminal needed for these steps | small, medium | 2.1 |
| Help that works offline and while the repo is private | the Help menu always opens something | small to medium | 1.5 |
| Start screen with a sample clip | a stranger can try the tool in a minute | small | 1.5: hidden `synth` |
| Internationalization | other languages | large | 0 `tr()` calls; tests assert English texts |

### Features

| item | value | size | today |
|---|---|---|---|
| Headless `run SESSION.json` (D8), then a batch over many videos | overnight and cluster runs | medium, then large | 2.1, 2.3 |
| Export formats beyond CSV (Parquet or HDF5 tables, masks as COCO RLE, JSON) | fits other analysis tools | medium each | CSV, npz, mp4 only (README.md:132-141) |
| Quick-look plots of a track (position, speed, area over time) | check a result before leaving the app | medium | no plot widget; pyqtgraph is installed |
| Overlay extras (D5), calibration sharing (D6) | better slides; one calibration per group | medium, small | 2.1 |
| Box prompts, backward tracking | fewer clicks; track before the start frame | medium each | 2.3 |
| Image sequences and TIFF stacks | microscope data | medium | 2.3 |
| Plugin interface for tracking methods (entry points on the `Segmenter` protocol) | others add methods without a fork; base for the owner's model list | medium to large | protocol in segmenter/base.py; models are a dict (hf.py:53-57; DEVELOPER.md:177-179) |
| Documented Python API | use from notebooks; the core is Qt-free | small to medium | not documented |
| CUDA checked and documented | the "orders of magnitude" goal needs it | medium (needs hardware) | 1.3: never run |

### Distribution

| item | value | size | today |
|---|---|---|---|
| Tag and GitHub release `v0.1.0`; install command with a tag | users get a named version | small | 1.2 |
| Publish on PyPI (trusted publishing); later perhaps conda-forge | `uv tool install outline-tracker`, `pipx`; no git needed | small to medium (conda-forge: medium) | needs license, metadata, a pin policy (1.1, 1.3) |
| One source for the version (from the tag) | no drift; a commit in every build | small | 1.2: two places |
| Installers or app bundles (macOS `.app`/`.dmg`, Windows installer) | people who never use a terminal | large | none; torch makes the bundle large (size not measured); LGPL and ffmpeg terms apply (1.3) |
| Signing and notarization | no Gatekeeper or SmartScreen warning | medium, plus yearly fees and accounts | none |
| Update notice (opt-in check of the release list) | users learn of fixes | small | none; first network call outside the model |
| Rename the model cache and its variable, with a move of the old folder | no course name on users' disks | small | 1.6 |
| Statement of supported systems (Linux, macOS minimum, Intel Mac) | no surprise at install | small to write; large to widen | 1.3 |

### Community

| item | value | size | today |
|---|---|---|---|
| LICENSE and NOTICE | nobody may legally use or contribute without it | small (owner decides) | 1.1 |
| CONTRIBUTING (from DEVELOPER.md and the rules) | a stranger knows how to set up, test, and propose a change | small to medium | 1.1 |
| CODE_OF_CONDUCT, SECURITY.md | expected by users and by GitHub's community checklist | small each | 1.1 |
| CITATION.cff, a DOI, and how to cite SAM 2 and EdgeTAM | credit in papers | small | 1.1 |
| Issue forms and a PR template that ask for the diagnostics file | usable bug reports | small | 1.1 |
| README for strangers; class material moved to its own page | the first screen says what the tool is | medium | 1.5 |
| Documentation site with pictures; screenshots script (D10); a demo clip | people see the tool before they install | medium to large | 1.5 |
| Repository description, topics, Discussions | found by search; a place for questions | small | 1.1 |

### Quality gates

| item | value | size | today |
|---|---|---|---|
| macOS CI job (Apple chip runner) | the owner's own platform is tested on every push | small | none; costs more minutes while private |
| Lint and format gate (for example ruff) | one style for outside contributors | small to medium (first clean-up) | none |
| Type check gate | catches interface errors as the method list grows | medium to large | none |
| Coverage report, then a threshold | shows untested code | small | none |
| Nightly real-model tests | the 28 slow test functions run somewhere other than one laptop | medium | none; hosted runners are CPU; whether `mps` tests can run there: not verified |
| Python version matrix | `>=3.11` becomes a tested claim | small | 3.12 only |
| Wheel and sdist build and install check; look at what the sdist contains | a broken package is caught before release | small | not done (E3); sdist content not verified |
| Dependency update automation | pins do not rot | small to set up; steady review work with exact pins | none |
| pre-commit, release workflow, actions pinned by hash, dependency audit | routine hygiene | small each | none |

## 4. Stability evidence today

Fast tests: about 1,080 test functions in 67 files under `tests/` and 546 in 41 files under `tests/gui` (counted in the source, before parametrization); the build ledger's last full run reports 2,682 passed, 1 skipped, 13 xfailed (`.superpowers/sdd/PLAN/progress.md`, entry "Wed 14:55"; not re-run here). They use stand-in segmenters on synthetic clips with known answers and need no torch (docs/DEVELOPER.md:102-127), and they run on Ubuntu and Windows in CI (tests.yml). The window's tests run offscreen with dialogs replaced by a recorder (docs/DEVELOPER.md:125-127), so no test sees a real screen. Slow tests: 28 functions in 8 files under `tests/slow` run the real model on short synthetic clips, on cpu and on the Apple GPU of one Mac; they are not in CI; five are xfail (docs/PLAN.md:281-306; docs/VALIDATION.md sections 1, 2, 4, 5, 6.1). By hand, not as tests: a script drove the whole window with the real model twice, offscreen (docs/VALIDATION.md:564-617), and the owner reports going through all panels on a Mac with the synthetic clips. Known gaps: no person has used the window on a real Windows screen, and Windows has never run the real model (docs/VALIDATION.md:621, 625-626; docs/PLAN.md:263-268); no real video and no long video has been tracked, so opening time and memory over minutes are unknown (docs/VALIDATION.md:622); the three corrections of panel 8 were tested with stand-ins only (docs/VALIDATION.md:623-624); the comparison with last week's real outputs was left out (docs/PLAN.md:1536-1538); `--device cuda` was never run (docs/VALIDATION.md:530); a real file lock on Windows is covered by one test of `atomic_write` in CI (tests/test_fileio.py:251-262), not by a spreadsheet program holding an exported file; macOS has no CI job, so macOS results come from one laptop; Python 3.11, 3.13, 3.14 and macOS 13 are untested.
