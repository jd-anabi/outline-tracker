# Outline Tracker implementation plan

> **For agentic workers:** work through the tasks in order, one at a time, with the cycle in
> "How the work runs". Tick a task's box only when its tests pass and it is pushed. The spec
> (SPEC.md) and the rules (CLAUDE.md) travel with this plan: read both before the first task.

**Goal:** by Thursday Oct 8, 1 pm Pacific, 16 students install `outline-tracker` on their own
laptops, track objects with EdgeTAM, and export positions and outline shapes in the SPEC §8 format.

**Architecture:** a Qt-free core (`outline_tracker`: geometry, measurement, tracking runners,
results store, export) that the command line and the GUI both call. The model sits behind a small
`Segmenter` protocol, so everything except the model itself is tested with stand-in segmenters on
synthetic clips with known answers. The GUI (`outline_tracker/gui`) only collects input and shows
results.

**Tech stack:** Python 3.12 (>= 3.11), uv, numpy, scipy, pandas, opencv-python-headless,
scikit-image, imageio-ffmpeg, torch, transformers 5.18.0 (EdgeTAM / SAM 2.1), PySide6-Essentials,
pyqtgraph, pytest, pytest-qt.

**Spec:** [SPEC.md](../SPEC.md) v1.2. Section numbers below (§) refer to it.

**Status:** plan drafted Tue Oct 6 evening. Waiting for J's approval. Setup task S0 is done.
Read sections 0–2 first; they are the part that needs J tonight.

---

## 0. Before J leaves tonight (5 minutes, J present)

- [ ] **Power.** The Mac is on battery (68% at 20:27) and system sleep is set to 1 minute. Plug it
  in and leave the lid open. Check: `pmset -g batt` says `AC Power`.
- [ ] **Folder.** This repo sits in `~/Desktop`, which iCloud syncs (question 1 below). No virtual
  environment is created until that is settled.
- [ ] **No prompts.** Right after approval the agent runs one command of each kind while J
  watches (`uv sync`, `uv run pytest`, `git commit`, `git push`, `gh run list`, a background test
  run, a write under `~/.cache`), so every permission prompt is answered once. The agent also
  asks the app to keep the Mac awake for the session.

---

## 1. Timeline and tripwires (Pacific)

| when | what |
|---|---|
| Tue night | Phase A (core), Phase B (commands) |
| **Wed ~noon** | **GO/NO-GO 1**: J runs `from-tracker` on a real clip (§14.1) |
| Wed afternoon | Phase C (GUI, P0) |
| **Wed evening** | **GO/NO-GO 2**: J runs the §14.3 checklist on the Mac, then Windows |
| Wed night | Phase D (P1), in the §14.2 order |
| Thu morning | Phase E (fixes, README, confirm pins, tag `v0.1.0`, repo public) |
| Thu 1 pm | students install |

The go/no-go points are checks for J, not stops for the agent. If the core finishes before J is
back, the agent starts Phase C. Anything J reports at a go/no-go point is fixed before new work.

**Size.** Phases A and B are 27 tasks; a rough estimate is 12–18 agent-hours against about
14 hours until Wed noon. Tasks marked **[CP]** are the critical path to go/no-go 1 (positions,
Tracker-format files, overlay, log). Independent modules (geometry, measure, derive, qc on one
side; ports, synthetic clips, tracking on the other) may be built in two parallel git worktrees
and merged into `main` one at a time, each merge with the fast tests green.

Tripwires (the agent applies them without asking and records what it cut under "Questions for J"):

1. **Tue 23:00** — both CI jobs are not green on the A01 skeleton: continue locally, spend at most
   30 minutes per CI problem. The Windows job must be green before go/no-go 2.
2. **Wed 00:30** — the ported model does not reproduce last week's positions (A04 slow test red):
   run last week's selftest from the reference copy to tell "environment" from "port", record
   numbers and images in docs/VALIDATION.md, spend at most 45 minutes, then continue on the
   stand-in segmenters. No tuning (§13.4).
3. **Wed 03:00** — frame exactness (A09) is not green locally and on both CI runners: use plain
   forward decoding from a cached position.
4. **Wed 06:00** — derive and flags (A12, A13) are not done: `from-tracker` (B1) ships first with
   positions.csv, the Tracker-format folder, overlay, session and log; shapes follow.
5. **Wed 08:00** — `from-tracker` does not run end to end on a synthetic clip: apply the §14.2 cut
   order inside P0 (`radial.csv`, then circularity and Feret) and finish only [CP] tasks.
6. **Wed 18:00** — the GUI smoke test (C6) is not green: freeze features, drop the cursor readout,
   no P1 work.
7. **Wed 21:00** — click, track, export does not work in the GUI on a synthetic clip: recommend the
   `from-tracker` fallback for Thursday (§14.1); Phase E documents that path first.

CI is never waited on: the agent looks at `gh run list` at the start of the next task.

---

## 2. Questions for J

Answer in chat. Each has the default the agent uses if there is no answer.

### Needs an answer before the overnight run

1. **Move the working copy out of iCloud?** `~/Desktop` is synced (`FXICloudDriveDesktop = 1`,
   Optimize Storage on), which §14.1 says to avoid: a 1.3 GB, 31,000-file `.venv` and a busy
   `.git` there can stall or corrupt an unattended run.
   **Recommended:** the agent clones the repo to `~/Developer/outline-tracker` and moves this
   session there. Your notes file `outline-tracker-spec.md` stays behind and is never committed.
   Default without an answer: stay here, keep the environment outside the folder
   (`UV_PROJECT_ENVIRONMENT=~/.venvs/outline-tracker`), and re-clone if git reports a conflict copy.
2. **Power and prompts**: the two actions in section 0.

### Not blocking: defaults J can overrule

None of these stops the overnight work. The date says when a different answer is still cheap.

3. **Go/no-go 1 inputs (Wed noon).** No real `_tracker.mp4`, `extra/start.csv` or last week's
   `edgetam/` folder was found on this Mac. You need local copies (not cloud placeholders) of one
   day-1 clip, the start file exported from its `.trk`, the `edgetam/` CSVs made from that start
   file, and fps_true. Default: the commands use shell variables you fill in. If a real
   `_tracker.mp4` is on this Mac tonight and you name it, the agent runs decode-only checks on
   it (frame exactness, timestamp gaps; no model); otherwise that check is part of go/no-go 1.
4. **Students on macOS 13? (Thu morning)** torch 2.14.1 has wheels only for macOS 14 and later; an exact pin
   `torch==2.14.1` (§15) cannot install on macOS 13. Default: `torch>=2.11.0,<=2.14.1` and the
   matching torchvision range. Windows and macOS 14+ then get exactly 2.14.1; a macOS 13 laptop
   gets 2.11.0, which is what `--with torch` gave it last week. Say "macOS 14 only" for exact pins.
5. **Note for the analysis template (no answer needed): `flags` in positions.csv.** §8.2 and
   §8.4 give both files the same §9 codes, so every row of a coarse dish-scale track reads
   `LOWRES;HEADGUESS` (the body is about 14 px, under the 20 px limit, and `from-tracker` has no
   head click). The template should filter on specific codes, not on "flags is empty".
   README.txt says which codes concern positions (LOST, JUMP, SIZE, CONTACT, EDGE) and which
   only shapes.
6. **`from-tracker` output folder and name (before go/no-go 1).** §11 gives it no student name, and §8.1 needs one.
   Default: an optional `--student NAME`; without it, the name of the export's home folder (last
   week's per-student folder, e.g. `ana`); the run folder is `--out`, else
   `<video folder>/<video stem>_outline_<student>/`. It never writes into a folder that already
   holds other CSV files, so last week's `edgetam/` folder cannot be overwritten. Track names
   come from the Tracker export unchanged (the `^[A-Z]+[0-9]*$` rule applies to ids made in the GUI).
7. **torchvision as a listed dependency.** It is not in §15 but is already installed through
   timm; listing it is needed so CI's CPU-only index applies to it. Default: list and pin it.
8. **Dependency versions pinned from the first commit**, not Thursday morning: `uv tool install`
   ignores `uv.lock`, so the Windows check at go/no-go 2 would otherwise test other versions than
   students get. The set is today's resolution, which equals last week's (torch 2.14.1,
   transformers 5.18.0, timm 1.0.30, numpy 2.5.3, opencv-python-headless 5.0.0.93,
   PySide6-Essentials 6.11.2, pyqtgraph 0.14.0, scikit-image 0.26.0, imageio-ffmpeg 0.6.0).
   Thursday's step becomes "confirm the pins". Note: §10.2 describes the Windows import bug for
   torch 2.9.x; with 2.14.1 it is untested until the Windows CI job runs. The import order stays.
9. **Run last week's selftest before the port? (tonight, optional)** It would fetch the model (56 MB from Hugging
   Face, public) and give baseline numbers on cpu and on the Apple GPU within minutes of approval.
   The kickoff allows slow tests "once the Hugging Face segmenter is ported". Default: wait for
   A04 (first two hours).
10. **GitHub Actions minutes.** The repo is private, so minutes are metered (2,000 free per month
    on a Free account; Windows counts double). About 20 billed minutes per code push; the whole
    project may use 1,000–1,800. Default: go ahead, with cancel-in-progress, job timeouts, and no
    CI for documentation-only pushes. If the quota runs out, the Windows check stops until the
    repo is public.
11. **Student install command (Thu).** Direct pins do not freeze indirect dependencies (two
    changed today). Adding `--exclude-newer <time of the tagged lock>` to the §15 command would
    freeze them. Default: the §15 command as written.
12. **License (Thu).** Default: Apache-2.0 as §15 suggests, with a NOTICE crediting the
    transformers conversion script and Meta's EdgeTAM and SAM 2. Which copyright holder name?
13. **Go/no-go 2 inputs (by Wed evening).** A Windows machine with uv and Git where you can sign
    in to GitHub; a group repo with a working `load_tracks` cloned on the Mac; a real close-up
    clip if one exists. Default: the synthetic close-up clip for item 7.

14. **Cutting GUI items beyond the spec's cut order.** §10 marks everything P0 unless stated, but
    §14.2's never-cut list does not include: keys 1–9, Save session as, the mask-fill toggle, the
    GUI model selector, the Stopwatch dialog (typed fps remains), play/pause. Default: if
    Wednesday runs late, after the spec's own cuts these go in that order, each recorded here.
15. **Two readings of the spec to confirm** (both only change a re-export, not the tracking):
    X17 (one axis jump flags one frame as `ORIENT`, not the rest of the track) and the note that
    objects with aspect ratio 5 or more, or appendages longer than about 1.7 body lengths, always
    take the core fallback and carry `ORIENT` unless `core_open_frac` is lowered.

### Raised during the work

_(the agent adds disputed tests and cut decisions here, newest first)_

Checked numerically on Tue evening, before any code (scripts kept in the session's scratch
folder): every tolerance of §13.1, §13.2 and §13.4 is reachable, under these conditions, which
the tasks now state:

| check | measured | tolerance | condition |
|---|---|---|---|
| moments, ellipse a = 40, b = 15 | L1 0.30%, L2 1.04%, axis 0.32° | 2%, 2%, 1° | rasterize at pixel centers (a `cv2`-drawn ellipse gives 4.5%) |
| circle R = 50 perimeter, 256 points | −0.006% | 0.5% | analytic logits (distance-transform logits give +1.7 to 3.0%) |
| radial profile, ellipse | 0.11–0.22% | 1% | analytic center and heading (through the pipeline: 0.7–1.4%, so the expected value there is the exact ray–ellipse intersection from the measured center) |
| L-shape solidity | 0.50% at 120 px across (1.2% at 40 px) | 1% | L at least 120 px across |
| core, sideways rods | axis 0.7°, centroid 0.24 px, fraction 2.0% | 5°, 0.5 px, 5% | rods reach 30 px beyond the body, so the premise (lateral > axial moment) is true |
| `ThresholdFake` through H.264 | 0.18 px (R = 12, crf 10); 0.30 px (R = 6, crf 23) | 0.25 px | disks of radius 12 px, crf 10 |
| circle fit, 6 points, σ = 0.5 px | fails for 0.2% of seeds if evenly spread, 7.5% if random | 1.0 px | points spread evenly, fixed seed |
| §13.4 solidity signal | peak at 9.0 Hz; ideal-grid RMS 0.003 | ±0.5 Hz; 0.02 | true solidity from the analytic outline, not the pixel mask |

---

## 3. How the work runs

For every task, in this order:

1. Write the task's tests first. Expected values come from geometry, analytic shapes or synthetic
   ground truth, never from the code's own output.
2. Run them and see them fail for the right reason.
3. Implement until they pass.
4. Run `uv run pytest -m "not slow"`. It must be green.
5. Commit, push, tick the task's box here, and note in one line what was verified and what is
   uncertain.

Rules that apply to every task (from CLAUDE.md and the spec):

- A test that seems wrong is never changed: mark it `xfail(strict=True, reason=...)`, add it under
  "Questions for J", continue with the next task.
- Slow tests (real model) run only on short synthetic clips: on `cpu`, plus the `mps` runs of
  §13.4. They run in the background, one at a time, while fast work continues. The model is never
  run on long real videos by the agent.
- Ported code keeps its behavior. `tests/reference/` stays byte-identical to the template and is
  never imported by the package. Functions moved verbatim are checked by a source-equality test.
- Nothing outside `outline_tracker/gui` imports Qt. torch and transformers are imported only inside
  `outline_tracker/segmenter`, lazily; the GUI launcher imports torch before PySide6. No fast
  test needs torch, and no test file imports it at module level (pytest-qt loads PySide6 first,
  which is the order §10.2 warns about).
- `torch.set_num_threads` is called only in the GUI worker, never in the core (the 0.01 px
  regression of §13.3 needs the same thread count as the reference).
- Never load a whole video; no full-frame float arrays per object beyond the current frame.
- No new dependency beyond §15 without asking here first.
- Never commit videos, `results.npz` or weights. Never force-push.
- The repo becomes public: no answer keys, student data, rosters or personal paths.
- All pixel coordinates use Tracker's convention (pixel centers at +0.5); world units are mm with
  y up; `t_s = frame / fps_true`; frames live on the clip's grid. Units go in column names.

---

## 4. Decisions where the spec leaves a choice

These are the agent's defaults. J can overrule any of them.

| # | decision | why |
|---|---|---|
| X1 | Package folder `outline_tracker/` at the repo root, as §12 draws it; hatchling build; `requires-python >= 3.11` with numpy and scipy pinned per Python version | matches the spec's paths; the wheel ships only the package |
| X2 | `tests/reference/` is excluded from the normal test run. One test runs the 27 unmodified template tests against the unmodified reference copy in a subprocess | shows last week's code still passes with this week's libraries, so a failing ported test points at the port; avoids name clashes between the two test sets |
| X3 | Ported tests change only their import lines (see the table in section 8). The three end-to-end tests of `test_segment.py` keep their assertions on frames, times and positions, but look for the files in the §8.1 run folder | §8.1 and §8.3 move the output folder; nothing else changes |
| X4 | Exact random access (`FrameSource`) with OpenCV only, no PyAV. Tracking keeps the plain sequential decode. For display, a jump seeks, then **identifies the frame actually delivered** from its timestamp against a table of all frame timestamps (read once with the bundled ffmpeg, no decoding), then steps forward to the wanted frame. If anything does not match, it decodes forward from the start | §3.5 forbids unverified seeking, with reason: measured tonight, plain OpenCV seeking is exact on evenly timed clips (0 of 50 wrong) but silently 1–2 frames off when timestamps have gaps (10 of 50 wrong), and `convert` keeps the phone's timestamps. The verified method: 0 wrong in 1,518 reads on 7 clips, about 20 ms per jump at 1080p |
| X5 | `results.npz` also stores `core_fallback` and `second_fraction` per frame | `ORIENT` and `MULTI` (§9) cannot be computed from the §8.12 keys alone |
| X6 | `CONTACT` threshold per pair of tracks = `max(3 px, 2 · max(cell_i, cell_j))`, from each track's stored grid cell size | §9 says "2 coarse grid cells"; for two fine tracks that leaves 3 px |
| X7 | One worker thread (64 MiB stack) owns the loaded model and runs previews, runs, exports and probe passes one at a time. Preview is disabled while a run is active. After an Apple-GPU fallback the app stays on cpu | one copy of the weights; no concurrent use of the model; a default Qt thread has a 0.5 MiB stack on macOS, where the model has never run |
| X8 | Frame hash = SHA-256 of the decoded RGB frame's bytes (2.5 ms at 1080p), stored with a decoder tag (OpenCV version, platform, machine). Same tag and a different hash aborts the run. A different tag (the folder moved to another computer) recomputes the stored hash here and logs it | decoders on Windows and macOS may differ in the last bit, and §8.1 wants the run folder to be movable |
| X9 | fps_true for `from-tracker` and `probe`: `--fps`; else `data/manifest.csv` in the current folder (last week's rule); else the same file searched upward from the export's folder, then the video's folder (§4.1); else the export's `t` column, recorded as source `tracker-export` with a printed note | keeps last week's behavior and adds §4.1's search |
| X10 | Two hidden commands not in §11: `compare-tracks NEW_DIR OLD_DIR` (RMS difference in px per track on common frames) and `synth {dish,closeup} OUT.mp4` (writes a synthetic clip) | the go/no-go checks need them, also after `uv tool install` on Windows |
| X11 | New CSVs: UTF-8, `\n` line ends on every platform, NaN as an empty cell. The Tracker-format files keep the ported writer's bytes (so CRLF on Windows, as last week), written to a temporary file and renamed | §8.2 fixes the decimals only; §8.3 asks for last week's bytes |
| X12 | `radial.csv` and `outlines.npz` are always written; with no fine track they hold only the header or the `meta` key | §8.1 marks only `probes.csv` as optional and §14.3 wants every P0 file |
| X13 | `score` in `results.npz` is the model's object-presence logit | it is the only score the Hugging Face video model returns |
| X14 | Track colors follow last week's overlay (A is yellow), stored as RGB hex | students' overlays look the same as last week |
| X15 | `results.npz` also keeps each frame's mask crop, bit-packed (about 1 MB per 1,200-frame fine track) | `core_open_frac` is a setting (§8.10); without the masks, changing it would mean tracking again |
| X16 | Synthetic shapes are implicit functions; `ExactFake` returns the analytic signed distance to the shape as logits | §13.2 says "signed distance to the mask boundary"; a distance transform of the pixel mask gives a staircase outline (perimeter +2 to 3%, radial 3 to 5%) that cannot meet §13.1's 0.5% and 1% |
| X17 | Heading reference: an axis jump or a core fallback flags that frame `ORIENT` but still becomes the next reference; only frames with an undefined axis are skipped | read literally, §7.3 flags every frame after one large turn (measured: 470 of 470) |

---

## 5. Review focus

Inputs the spec implies but its test list (§13) does not exercise. Each has a test in the task
that owns the code.

1. **Paths with spaces and non-ASCII characters** (§15 pitfalls), on Windows too: every command and
   the GUI must open, track, export and write the overlay. Test in A08 (`clip_in_odd_folder`
   fixture, used by A15, A18, A19, B1) and run by the Windows CI job.
2. **A video that ends before the frame count the file reports**: tracking, probes and the overlay
   stop at the last decodable frame, keep what they have, and say so. Tests in A15 and A20.
3. **Nonsense typed into a field**: fps_true <= 0, stick length 0, identical stick endpoints,
   collinear circle points, clip end before start, step 0. Each gives a clear error and writes no
   file. Tests in A06 and A07.
4. **An object that is never found** (empty preview mask, lost from the first frame, leaves the
   frame): the run continues, rows are NaN with `LOST`, the fine window falls back to 96 px.
   Tests in A15 and A16.
5. **A run folder that moved, or whose video moved**: the session reopens through the relative
   path, then the absolute path, then asks; a different file (size or hash) is refused with a
   clear message; `export` without the video still writes every CSV and says the overlay was
   skipped. Tests in A07 and B3.

---

## 6. File map

```
pyproject.toml  uv.lock  .gitignore  .python-version  LICENSE  README.md  CHANGELOG.md
.github/workflows/tests.yml
conftest.py                          sets QT_QPA_PLATFORM=offscreen for every test run
outline_tracker/
  __init__.py        __version__ = "0.1.0"
  schema.py          columns, flags, file names, formats, README.txt text        (A05)
  geometry.py        WorldFrame, stick, tape, circle, fps, grid, manifest        (A06)
  tracker_io.py      ported: read/write Tracker files, Calibration, make_plan    (A03)
  fileio.py          atomic writes with Windows lock retry                       (A07)
  session.py         dataclasses <-> session.json, video identity                (A07)
  video.py           ported probe/check_video/iter_frames + FrameSource          (A02, A09)
  provenance.py      tool version and commit, machine facts for run.log          (A18)
  convert.py         ported                                                      (A02)
  synthetic.py       test clips with ground truth; the selftest clip             (A08)
  segmenter/base.py  Segmenter, ObjectPrompt, MaskResult                         (A04)
  segmenter/hf.py    ported TransformersSegmenter + extensions                   (A04)
  segmenter/edgetam_convert.py  ported _edgetam.py                               (A04)
  segmenter/fake.py  ThresholdFake, ExactFake                                    (A10)
  measure.py         mask (+ logits) -> PixelRecord                              (A03, A11)
  derive.py          PixelRecords + calibration -> world quantities              (A12)
  qc.py              flags                                                       (A13)
  results.py         results.npz store                                           (A14)
  tracking.py        runs, coarse and fine runners, corrections, hash guard      (A15-A17)
  export.py          every file of §8 except the overlay                         (A18)
  overlay.py         overlay.mp4                                                 (A19)
  probes.py          brightness probes                                           (A20)
  from_tracker.py    Tracker export -> session -> run -> export                  (B1)
  cli.py             entry point and subcommands                                 (A01, B1-B4)
  gui/               app.py, main_window.py, video_view.py, worker.py, panels/   (A01b, Phase C)
tests/               one test file per module; tests/slow/ for the real model; tests/gui/
tests/reference/     unmodified template files (done, S0)
docs/                PLAN.md, OUTPUTS.md, DEVELOPER.md, VALIDATION.md, screenshots/
```

---

## 7. Tasks

Every "Check" line is a command J can run from the repo folder in Terminal. Unless it says
otherwise, the expected result is: all tests pass, none skipped unexpectedly.

### Setup

- [x] **S0 · Repo, rules, reference copies** (kickoff steps 1–2)
  - CLAUDE.md is SPEC Appendix A verbatim. Private repo `jd-anabi/outline-tracker` created and
    pushed. The ten allowed template files are copied byte for byte from
    `shrimp-tracker-template@4ec8cd7` into `tests/reference/`.
  - Check: `gh repo view jd-anabi/outline-tracker --json visibility` shows `PRIVATE`.

### Phase A: core (Tue night)

- [ ] **A01 · Scaffold, CI, repo guards** [CP] (§12 rules, §13.7, §15)
  - Files: `pyproject.toml`, `uv.lock`, `.python-version` (3.12), `.gitignore`, `conftest.py`
    (repo root: `QT_QPA_PLATFORM=offscreen` by default, so test runs never open windows; on
    Windows it imports torch first when torch is installed, because pytest-qt loads Qt),
    `outline_tracker/__init__.py`, `outline_tracker/cli.py` (`main(argv=None) -> int`,
    `--version`), `.github/workflows/tests.yml`, `README.md` (stub), `CHANGELOG.md`,
    `tests/conftest.py` (the template's helpers, unchanged), `tests/helpers.py` (new helpers),
    `tests/test_repo_rules.py`.
  - Dependencies pinned now (question 8): numpy 2.5.3 (2.4.6 on Python 3.11), scipy 1.18.1
    (1.17.1 on 3.11), pandas 3.0.6, opencv-python-headless 5.0.0.93, imageio-ffmpeg 0.6.0,
    scikit-image 0.26.0, PySide6-Essentials 6.11.2, pyqtgraph 0.14.0, torch 2.11.0–2.14.1,
    torchvision 0.26.0–0.29.1 (questions 4 and 7), transformers 5.18.0, timm 1.0.30,
    huggingface_hub 1.33.0, safetensors 0.8.0; dev: pytest 9.1.1, pytest-qt 4.5.0.
    Linux takes torch and torchvision from the PyTorch CPU index (`sys_platform == 'linux'`).
  - pytest: `testpaths = ["tests"]`, `pythonpath = ["tests/reference"]`,
    `norecursedirs = ["reference", ...]`, `--strict-markers`, marker `slow`,
    `faulthandler_timeout = 120`.
  - Tests first:
    - `test_version_flag`: `main(["--version"])` prints `outline-tracker 0.1.0` and loads neither
      torch nor Qt.
    - `test_reference_copies_unmodified`: SHA-256 of the ten files in `tests/reference/` equal the
      values recorded from template commit `4ec8cd7`.
    - `test_template_tests_pass_on_reference` (X2): `pytest tests/reference/template_tests` in a
      subprocess reports 27 passed.
    - `test_import_boundaries`: an AST scan finds no `PySide6`/`pyqtgraph` import outside
      `gui/`, no `torch`/`transformers`/`timm` import outside `segmenter/` and the GUI launcher,
      and no import of `shrimp` anywhere in `outline_tracker/`.
    - `test_core_does_not_load_torch`: importing every non-GUI, non-segmenter module in a
      subprocess leaves `torch` and `PySide6` out of `sys.modules`.
  - CI (`actions/checkout@v7`, `astral-sh/setup-uv@v10.2.0`, uv 0.12.23, Python 3.12,
    `uv sync --locked`): ubuntu-latest installs the Qt system libraries (`libegl1 libgl1
    libxkbcommon0 libfontconfig1 libdbus-1-3 libx11-6`) and runs the fast tests; windows-latest
    runs `import torch; import PySide6.QtWidgets` (must pass), the reverse order (information
    only), `uv run outline-tracker --version`, `uv tool install --python 3.12 .` followed by
    `outline-tracker --version` (the students' install path), and the fast tests. Older runs of
    the same branch are cancelled, jobs time out after 15 and 25 minutes, and pushes that touch
    only `*.md` or `docs/` do not start CI.
  - Check: `uv sync && uv run outline-tracker --version && uv run pytest -m "not slow" -q`, then
    `gh run list --limit 1` shows the CI run green.

- [ ] **A01b · GUI launcher and an empty window** (§10.2, §13.7; pulled forward from Phase C)
  - Files: `outline_tracker/gui/__init__.py`, `gui/app.py` (`main(argv)`: imports torch inside
    the function, then PySide6, then pyqtgraph; creates the application and an empty
    `MainWindow` with the nine numbered panels as placeholders), `cli.py` (`gui`, and no
    arguments opens the GUI), `tests/gui/test_shell.py`, `tests/gui/test_import_order.py`.
  - Why now: it puts the real entry point and pytest-qt on both CI runners hours before any GUI
    work, which is the only Windows evidence before go/no-go 2.
  - Tests first: the window opens and closes offscreen; in a fresh subprocess, running the
    launcher imports `torch` before any `PySide6` module (recorded with an import hook); the
    `gui` package itself imports without torch; no Qt object is created at import time.
  - Check: `uv run pytest tests/gui -q`; `uv run outline-tracker` opens an empty window with the
    numbered panels; `gh run list --limit 1` shows both CI jobs green.

- [ ] **A02 · Port video, convert, check** [CP] (§0, §11, §12)
  - Files: `outline_tracker/video.py` (the template's functions, unchanged), `convert.py`,
    `cli.py` (`convert VIDEO...`, `check VIDEO...`: the template's `main` bodies and messages;
    errors become `ERROR: …` with exit code 1), `tests/test_video.py` (8 tests),
    `tests/test_convert.py` (2 tests), both ported with only the import line changed,
    `tests/test_port_fidelity.py`.
  - Tests first: the ported tests; `test_port_fidelity`: for every function moved verbatim, its
    source text equals the reference copy's (`inspect.getsource`). Later port tasks add their
    functions to this test.
  - Check: `uv run pytest tests/test_video.py tests/test_convert.py tests/test_port_fidelity.py -q`

- [ ] **A03 · Port Tracker I/O and `mask_center`** [CP] (§0, §8.3, §11)
  - Files: `outline_tracker/tracker_io.py` (`read_tracker_export`, `Calibration`,
    `fit_calibration`, `write_tracker_file`, `Plan`, `make_plan`, `fps_from_export`),
    `geometry.py` (`fps_from_manifest(video, manifest) -> float | None`, ported),
    `measure.py` (`mask_center(mask) -> (u_px, v_px, area_px)`, ported),
    `tests/test_tracker_io.py` (13 of the 14 non-pipeline cases of `test_segment.py`; imports
    only), `tests/test_measure.py` (the `mask_center` case), `tests/test_port_equivalence.py`.
  - Tests first, in addition to the ported ones: for 200 random inputs, `fit_calibration`
    returns the same fields as `shrimp.segment.fit_calibration`; `write_tracker_file` writes the
    same bytes as the reference on this platform (never compared with a literal, since Windows
    writes CRLF); `make_plan` returns the same `Plan`; `mask_center` returns the same triple on
    random masks (§13.1 Centroid); the fidelity test covers the moved functions.
  - Check: `uv run pytest tests/test_tracker_io.py tests/test_port_equivalence.py -q`

- [ ] **A04 · Segmenter protocol and the Hugging Face backend** [CP] (§5, §6.2, §12, §13.3, §13.4)
  - Files: `segmenter/base.py` (`Segmenter`, `ObjectPrompt`, `MaskResult` exactly as §12;
    `crop_to_bbox(mask, logits, pad=8) -> MaskResult`), `segmenter/edgetam_convert.py` (ported
    `_edgetam.py`; same cache path `~/.cache/shrimp-models/edgetam` and `$SHRIMP_MODEL_CACHE`),
    `segmenter/hf.py` (`HFSegmenter(model_key="edgetam", device="auto")` with `start`, `step`,
    `close`, `preview(image, prompts)`, `device`, `model_id`, `weights_sha256`),
    `tests/test_hf_helpers.py`, `tests/slow/test_regression_reference.py`,
    `tests/slow/test_real_model.py`.
  - Kept from `TransformersSegmenter`: streaming session with its own consecutive `frame_idx`
    (0, 1, 2, …, not the video frame number), pruning with `KEEP_FRAMES = 20`, device order
    cuda > mps > cpu, the mps-to-cpu fallback in `start()`.
  - New, checked against the installed transformers 5.18.0 source:
    - prompts: **one** `add_inputs_to_inference_session` call per object with
      `input_points = [[[[x, y], …]]]` and `input_labels = [[[1, 0, …]]]` (Python floats and
      ints), then `session.obj_with_new_inputs = list(ids)`. Objects are not batched: the
      processor pads unequal point counts with −10, which changes how a one-click object is
      decoded. With one click per object the tensors equal last week's;
    - points that fall outside the image given to the model are dropped; an object left without a
      positive point raises a clear error before the model is called;
    - logits through `post_process_masks(..., binarize=False)`, one object at a time, with
      mask = logits > 0 (equal to `binarize=True`); `score` = the object-presence logit;
    - on a `RuntimeError` or `TypeError` in `step()` while on mps: move to cpu, open a new session
      at the current frame with one positive click at each object's last visible centroid, log it;
    - `preview()` uses a throwaway one-frame session on the same loaded model; one lock guards
      every model call;
    - an `OSError` while loading (the EdgeTAM config needs one small download from the Hugging
      Face Hub the first time) becomes "the first run needs internet once".
  - Kept as hidden options, as last week: model `sam2-small`, `--edgetam-checkpoint`.
  - Tests first (fast, no torch): the prompt lists built per object have the nesting above for
    objects with 1, 3 and 2 points; out-of-image points are dropped; the fallback logic, driven
    by a stub that raises `NotImplementedError` on the third frame, re-seeds every object at its
    last visible centroid and finishes on `cpu`; pruning (on plain dicts) keeps at most 21
    results per object.
  - Tests first (slow, no weights, a few seconds): with the real processor and session, the
    stored prompt tensors have shapes (1, 1, P, 2), labels ≥ 0, and `obj_with_new_inputs == ids`.
  - Tests first (slow, real model, synthetic clips only):
    - §13.3: on the selftest clip, same click, same frames, `cpu`, one loaded model given to both
      the reference `TransformersSegmenter` and the new one, same process and thread count:
      positions equal those in the reference's CSV within 0.01 px (in practice the masks must be
      identical: the test shrimp covers about 75 px).
    - §13.4 selftest: max error < 3 px on `cpu`; once on `mps`.
    - §13.4 negative prompt: two touching ellipses, positive click on one, negative on the other:
      the mask overlaps the neighbor by < 5% of the neighbor's area.
    - step fallback with the real model on `mps`: the model call is made to raise once on the
      third frame; the run completes on `cpu` within 3 px (a plain mps run is unlikely to fail by
      itself, so §13.4's mps runs alone would not exercise the fallback).
    - `preview()` equals the first-frame masks of a run with the same prompts on `cpu`.
  - The slow tests start in the background as soon as this task's fast tests pass. Results go to
    docs/VALIDATION.md. A failure is reported with numbers and images, not tuned (§13.4).
  - Check: `uv run pytest tests/test_hf_helpers.py -q` (seconds), and
    `uv run pytest -m slow tests/slow/test_regression_reference.py -q` (a few minutes; the first
    run downloads the model).

- [ ] **A05 · Schema: one source of truth** [CP] (§8, §9, §13.1 Schema)
  - Files: `schema.py`, `tests/test_schema.py`.
  - Produces: `POSITIONS`, `SHAPES`, `RADIAL`, `PROBES` (lists of
    `Column(name, dtype, unit, fmt, meaning)`), `FLAGS` (ordered: LOST, JUMP, SIZE, CONTACT, EDGE,
    MULTI, LOWRES, ORIENT, HEADGUESS), `FILES`, `RESULTS_KEYS`, `RESULTS_VERSION = 1`,
    `SESSION_SCHEMA_VERSION = 1`, `format_row(columns, row) -> str`, `readme_text() -> str`.
  - Text rules for the new CSVs: UTF-8, `\n` line ends on every platform, NaN written as an empty
    cell, `_s` with 7 decimals, `_mm`/`_mm2`/`_rad`/ratios with 6, `_px` with 3, integers as
    integers (lost rows keep `visible`, `n_components`, `shape_ok` = 0).
  - Tests first: column names and order of positions.csv, shapes.csv, radial.csv (`r_000` …
    `r_355`, 72 columns) and probes.csv exactly as §8.2, §8.4, §8.5, §8.7; dtypes; `readme_text()`
    names every column, every flag and every file; a lost row formats with empty float cells and
    reads back with integer columns still integer.
  - Check: `uv run pytest tests/test_schema.py -q`

- [ ] **A06 · Geometry** [CP: the transform] (§3.2, §3.4, §4.1–4.5, §13.1 Transform / Stick, tape, fps / Circle fit /
  Grid snapping)
  - Files: `geometry.py`, `tests/test_geometry.py`.
  - Produces: `WorldFrame(k_mm_per_px, alpha_rad, u0, v0)` with `.J`, `.to_world(u, v)`,
    `.to_px(x, y)`, `.cov_to_world(cov_px)`; `world_frame_from_calibration(cal)` (raises
    `MirroredCalibrationError` when `cal.flip` is false); `calibration_from_world_frame(wf)`;
    `stick_scale(p1, p2, length_mm, sigma_click_px=0.5) -> StickScale(k_mm_per_px,
    rel_uncertainty, length_px, too_short)`; `tape_check(q1, q2, true_mm, k) ->
    TapeCheck(measured_mm, rel_error, ok)`; `fps_from_stopwatch(frame_a, time_a_s, frame_b,
    time_b_s)`; `fit_circle(points_px) -> CircleFit(center_px, radius_px, rms_px)`;
    `grid_frames(start, end, step)`; `snap_to_grid(frame, start, step, end) -> (frame, moved)`;
    `find_manifest(video, export=None, cwd=None) -> Path | None` (X9: current folder, then upward
    from the export, then upward from the video);
    `dish_mm_from_manifest(video, manifest)`.
  - Tests first:
    - round trip `to_px(to_world(u, v)) == (u, v)` for alpha = 0°, 90°, −30°; with alpha = 0, a
      larger v gives a smaller y;
    - for 100 random (k, alpha, u0, v0): equal to `Calibration.to_mm` in flip form and to the
      template's `tracker_map` rescaled by k/0.05 (1e-9 mm); `Calibration -> WorldFrame ->
      Calibration` is the identity; a mirrored fit (three or more non-collinear points with y
      down) is refused;
    - stick: 926 px for 30 mm gives 32.40 µm/px and σ_k/k = √2·0.5/926; `too_short` below 300 px;
    - tape: 20 mm measured as 20.1 mm is +0.5% and passes; 20.3 mm fails;
    - stopwatch: frames 100 and 2500 at 1.00 s and 11.00 s give 240.0; equal readings raise
      `ValueError`;
    - circle: 3 points on a circle give center and radius exactly (1e-6 px); 6 points spread
      evenly around R = 500 with σ = 0.5 px noise (fixed seed) give center and radius within
      1.0 px;
    - grid: start 96, step 2: frame 101 snaps to 100 or 102 with `moved = True`; 100 stays;
    - review focus 3: zero stick length, identical endpoints, collinear circle points, step 0 and
      end before start each raise `ValueError` with a plain message.
  - Check: `uv run pytest tests/test_geometry.py -q`

- [ ] **A07 · Session file and atomic writes** [CP] (§8.1, §8.10, §13.1 Session)
  - Files: `fileio.py`, `session.py`, `tests/test_fileio.py`, `tests/test_session.py`.
  - Produces: `atomic_write(path, write_fn) -> Path` (temporary file in the same folder, closed,
    then `os.replace`; on `PermissionError` retry for about 5 s; then keep the data as
    `<stem>.new<suffix>` and return that path); `sha256_first_64mib(path)`;
    dataclasses `Session`, `VideoRef`, `Clip`, `TimeSettings`, `CalibrationSettings`, `Axes`,
    `Circle`, `Processing`, `ProbeBox`, `Track`, `Prompt`, `RunRecord`, `Correction` with the
    fields and defaults of §8.10; `Session.load(path)`, `Session.save(path)`,
    `Session.world_frame() -> WorldFrame`, `Session.locate_video(run_folder) -> Path`;
    `SessionVersionError`.
  - Two additions to §8.10, both optional keys: `calibration.tracker_fit = {mm_per_px, rms_mm,
    n_points}` for sessions made by `from-tracker` (no stick exists), and
    `time.source = "tracker-export"`. Run entries also record `model_id` and `weights_sha256`
    (hashed inside `segmenter/` when the model loads, so `export` never needs torch).
  - Tests first: the §8.10 example round-trips without loss (load, save, compare as JSON); a
    `tracker_fit` session round-trips; `schema_version` 0 and 2 raise `SessionVersionError` whose
    message names both versions; video paths are stored relative to the run folder and the folder
    can be moved; a video on another Windows drive stores `relpath = null` (tested with
    `ntpath`); `atomic_write` survives `os.replace` failing three times, and after permanent
    failure leaves `<stem>.new<suffix>` and the old file intact; on Windows only, a target held
    open gives the `.new` file; review focus 5 (moved folder, moved video, wrong video refused).
  - Check: `uv run pytest tests/test_fileio.py tests/test_session.py -q`

- [ ] **A08 · Synthetic clips with ground truth** (§13.2)
  - Files: `synthetic.py`, `tests/test_synthetic.py`, fixtures in `tests/helpers.py`.
  - Rule for every synthetic shape (it is what makes the §13 tolerances reachable; see the
    measured table in "Raised during the work"): a shape is an **implicit function** with a
    signed distance d(u, v), positive inside (ellipse: (1 − q)/|∇q| with q = √((x/a)² + (y/b)²);
    thick segment: exact capsule distance; union: max). The ground-truth mask is d > 0 at pixel
    centers (c + ½, r + ½). Frames are rendered from the same function (coverage =
    clip(d + ½, 0, 1)). Sizes passed to `cv2` drawing calls are never used as truth (a `cv2`
    ellipse of nominal 80 × 30 px comes out 6% larger).
  - Produces: `Scene` (size, fps, n_frames, objects, optional dish circle, optional LED box),
    shapes `Disk`, `Ellipse`, `Shrimp` (ellipse body plus two antennae as thick segments attached
    near the front; angle from the head direction β(t) = β0 + B sin 2πft, β0 = 45°, B = 30°),
    paths (straight, arc), `render(scene, path, crf=23) -> GroundTruth` (libx264, GOP 24,
    B-frames, yuv420p, through imageio-ffmpeg), `GroundTruth.mask(track_id, frame)`,
    `GroundTruth.logits(track_id, frame)` (the signed distance), `GroundTruth.table` (per track
    and frame: centroid u, v, area, heading, and the true solidity = polygon area / hull area of
    the analytic outline sampled at 4× resolution, not of the pixel mask).
    Ready-made scenes: `dish_scene()` (32.4 µm/px, body ≈ 14 px, one object near the wall, two
    passing within 1 px, an LED switching on at a known frame), `closeup_scene()` (10 µm/px,
    body ≈ 47 × 20 px, antennae ≈ 3 px wide, 9 Hz), `disk_scene()` (dark disks of radius 12 px,
    rendered with crf 10, for `ThresholdFake`).
  - Also: `selftest_clip(folder)` (the template's selftest clip, moved verbatim) and the hidden
    command `outline-tracker synth {dish,closeup} OUT.mp4 [--seconds S]` (X10).
  - Tests first: a rendered clip decodes with exactly `n_frames` frames; the table's centroid
    equals `mask_center` of the ground-truth mask; the LED box mean jumps at the stated frame and
    not before; the true solidity of `closeup_scene()` (mean removed) has its largest spectral
    peak within 0.5 Hz of 9 Hz, not at 18 Hz; every frame differs from its neighbors (needed by
    A09); a clip written into a folder named `vidéo test ü` opens (review focus 1).
  - Check: `uv run pytest tests/test_synthetic.py -q`, and
    `uv run outline-tracker synth closeup ~/closeup_tracker.mp4 && open ~/closeup_tracker.mp4`
    shows a shrimp-like shape beating its antennae.

- [ ] **A09 · FrameSource: exact random access** (§3.5, §6.5, §13.6)
  - Files: `video.py` (adds `iter_rgb_frames(path, frames)` ported from `segment.py`,
    `frame_timestamps(path)`, `FrameSource`, `frame_hash`), `synthetic.py` (a clip variant with
    forced B-frames and gaps in its timestamps), `tests/test_frame_source.py`.
  - Produces: `FrameSource(path, cache_frames=64)` with `.info`, `.get(k) -> RGB uint8 array`
    (exact), `.close()`; the cache holds at most 64 frames or 400 MB; `frame_hash(rgb) -> str`
    and `decoder_tag()` (X8). Strategy X4: one open capture; up to 64 frames ahead it steps
    forward; otherwise it seeks a few frames early, reads which frame arrived from its timestamp,
    and steps forward to k; any mismatch, a missing table, or a non-FFmpeg backend falls back to
    decoding forward from frame 0. The ported `read_frame` and `iter_frames(start > 0)` stay for
    their own tests but are not used by display, probes or tracking.
    Hidden option `check VIDEO --seek`: 20 random frames through `FrameSource` against the
    sequential decode, and the number of timestamp gaps (used on a real file at go/no-go 1).
  - Tests first, on two clips (evenly timed, and with three timestamp gaps), both with B-frames:
    the fixtures really have packets out of timestamp order and, for the second, gaps; for 20
    seeded random k plus 0, 1, 23, 24, 25, n − 2, n − 1 in shuffled order, `get(k)` is
    bit-identical to frame k of `iter_rgb_frames`; the same with the timestamp table disabled
    (fallback path); plain `cv2` seeking is shown to be wrong on the gapped clip (so the test
    would catch a regression to it); the cache bound holds; `frame_hash` differs for frames that
    differ in one pixel. These run on both CI runners: they are the Windows proof of §13.6.
  - Check: `uv run pytest tests/test_frame_source.py -q`

- [ ] **A10 · Stand-in segmenters** (§12, §13.2)
  - Files: `segmenter/fake.py`, `tests/test_fakes.py`.
  - Produces: `ThresholdFake()` (dark connected components within 20 px of each object's last
    position, as the template's `DiskFinder`; logits from a distance transform, since it is
    tested on positions only), `ExactFake(ground_truth)` with `set_view(frame, offset, size)`
    (the runner tells it which frame and which crop it is looking at). `ExactFake` slices the
    full-frame ground-truth mask and the analytic signed-distance logits with the integer offset;
    it never re-renders in crop coordinates.
  - Tests first: on `disk_scene()` `ThresholdFake` centers are within 0.25 px of truth;
    `ExactFake` returns exactly the ground-truth mask for the full frame and for a shifted crop;
    logits > 0 equals the mask; each `MaskResult` is cropped to the bounding box ± 8 px with the
    right offset; an invisible object gives an empty `MaskResult`.
  - Check: `uv run pytest tests/test_fakes.py -q`

- [ ] **A11 · Measure: mask to pixel-space record** [CP: area and centroid] (§7.1–7.4, §7.8,
  §8.12, §13.1 Moments / Core)
  - Files: `measure.py`, `tests/test_measure.py`.
  - Produces: `PixelRecord` (frame, visible, area_px, u, v, cov_full[3], cov_core[3], core_u,
    core_v, core_frac, core_fallback, core_r_px, outline_px[256, 2], n_components,
    largest_fraction, second_fraction, cell_px, edge, mode, score, mask crop);
    `measure_mask(result, frame, input_box, mode, core_open_frac=0.1) -> PixelRecord`, where
    `input_box = (c0, r0, width, height)` is the image the model saw, in full-frame pixels.
  - Fixed conventions: components are 8-connected everywhere; the opening radius is
    `max(1, floor(0.1 · L1_px + 0.5))` with an explicit disk (x² + y² ≤ r²) on a zero-padded crop;
    the outline is `find_contours(logits, 0, fully_connected="high")` on the largest component
    (other components' logits set negative, one ring of padding equal to minus the absolute edge
    value so a mask cut by the border closes on the border), keeping the closed contour of
    largest area; array (row, col) maps to (u, v) = (col + ½, row + ½) plus the crop offset;
    without logits, `cv2.findContours` with the same +½.
  - Tests first:
    - ellipses a = 40, b = 15 px rasterized at pixel centers, at several angles and sub-pixel
      centers: `4√λ1`, `4√λ2` within 2% of 80 and 30; axis angle within 1°;
    - core: ellipse body 47 × 20 px with two 3 px rods at 90° reaching 30 px beyond the body
      surface. The test first asserts the premise (lateral second moment > axial), then: core
      axis within 5° of the body axis, core centroid within 0.5 px of the body centroid,
      `core_frac` within 5% of body area / total area;
    - an ellipse of aspect ratio 5 loses more than half its area in the opening: the full mask is
      used, `core_fallback` is true, and `core_frac` keeps the opening's fraction;
    - outline from the analytic signed distance of a circle R = 50: 256 points, perimeter within
      0.5% of 2πR, contour mean equal to the true center; without logits the `cv2` boundary is used;
    - a disk cut by the input border gives a closed outline lying on the border and `edge` true;
      `edge` is false one pixel away from the border; `cell_px = max(width, height)/256`;
    - two components (the second 20% of the first): `n_components = 2`, `largest_fraction` and
      `second_fraction` as constructed; two diagonal pixels count as one component; holes are
      ignored;
    - degenerate masks (empty, 1 px, 2 px, 2 × 2, a 1 px line) do not raise: empty gives
      `visible = False`, NaN floats, zero counts; the others give finite area and centroid.
  - Check: `uv run pytest tests/test_measure.py -q`

- [ ] **A12 · Derive: world quantities** (§7.2–7.9, §13.1 Moments / Head continuity / Outline
  geometry / Radial profile / Solidity and wall distance / Resolution)
  - Files: `derive.py`, `tests/test_derive.py`.
  - Produces: `derive_track(arrays, track, world_frame, fps_true, circle, processing) ->
    DerivedTrack` with per-frame columns for positions.csv and shapes.csv, `outline_xy_mm[n, N, 2]`,
    `outline_xieta_mm[n, N, 2]`, `radial_mm[n, 72]`, and the internal heading unit vectors.
  - Fixed conventions: `area_mm2` = k² × pixel count in both CSVs; solidity and circularity use
    the polygon area; perimeter, Feret, wall distance and the radial rays use the stored 256-point
    polygon mapped to world coordinates; the outline is reversed if its world signed area is
    negative; a convex-hull failure (collinear points) gives NaN, not an exception.
  - Heading reference (a deliberate reading of §7.3, listed under "Questions for J"): frames
    whose axis is undefined (core λ2/λ1 > 0.8, zero extent, lost) are flagged `ORIENT` and never
    become the reference; frames flagged for a jump or for the core fallback are flagged but do
    become the reference, so one jump flags one frame instead of the rest of the track.
    θ = θ_ref + the signed angle from the reference axis in (−90°, 90°]; NaN on lost rows.
    Without a head click: the sign that agrees with the core centroid's displacement from the
    first to the 10th visible frame; if that is under 1 px, the axis direction with x ≥ 0.
  - Tests first:
    - with alpha = 25°, the world-frame axis angle of a rotated ellipse is correct (det J = −1);
    - an ellipse rotating through 2π with a head click on frame 0: no sign flip, unwrapped θ within
      2° of truth; the body-with-rods shape beating through ±40°: heading within 5°; without a
      head click every frame is marked for `HEADGUESS` and the head follows the motion over the
      first 10 tracked frames; a 75° turn during a 30-frame round stretch flags that stretch and
      at most one frame after it;
    - outline of the R = 50 circle: N = 128 points, counterclockwise in world coordinates (signed
      area > 0), chord spacing equal within 1%, first point at the head point (farthest crossing
      of the ray from the core centroid along the heading);
    - radial profile: a circle gives constant R within 0.5 px; for the a = 40, b = 15 ellipse with
      the analytic center and heading, `ab/√(b²cos²φ + a²sin²φ)` within 1% at all 72 angles; a
      ray that misses gives NaN;
    - solidity: disk ≥ 0.99; an L-shape 120 px across, at several rotations, within 1% of its
      analytic value; circularity and maximum Feret of a disk;
    - `shape_ok` uses min(px_along_major, cells_along_major) ≥ 20: a plain 14 × 6 px ellipse in a
      96 px fine window (14 px, 37 cells) is not ok;
    - wall distances for points at known radii, NaN without a circle; negative outside.
  - Check: `uv run pytest tests/test_derive.py -q`

- [ ] **A13 · Quality flags** (§9, §13.1 Flags)
  - Files: `qc.py`, `tests/test_qc.py`.
  - Produces: `compute_flags(derived_by_track, arrays_by_track, world_frame, processing) ->
    dict[track_id, list[str]]` (one `;`-joined string per row, codes in §9 order; the same string
    goes to positions.csv and shapes.csv, question 5); `summary_lines(...)` for run.log in the
    style of last week's CHECK lines.
  - Fixed conventions: `JUMP` and `SIZE` exactly as last week's `_flags` (speed between
    consecutive visible frames over the real time difference, flag on the later frame; area
    strictly outside [0.5, 2] × the median of the track's visible frames), per track id (`A` and
    `A2` separately); `CONTACT` (X6) from the minimum distance between the two stored polygons in
    pixels (0 if they overlap), on frames where both are visible, for both tracks; lost rows
    carry `LOST` only (plus `HEADGUESS`), with `shape_ok` 0 and no `LOWRES`.
  - Tests first: constructed tracks trigger each of `LOST`, `JUMP` (> 100 mm/s), `SIZE`,
    `CONTACT`, `EDGE`, `MULTI` (second component ≥ 10% of the largest), `LOWRES`, `ORIENT`
    (λ2/λ1 > 0.8, axis jump beyond 60°, core fallback) on exactly the expected frames and nowhere
    else; `HEADGUESS` on every frame of a track without a head click.
  - Check: `uv run pytest tests/test_qc.py -q`

- [ ] **A14 · Results store** [CP] (§8.12)
  - Files: `results.py`, `tests/test_results.py`.
  - Produces: `ResultsStore` with `put(track_id, record)`, `arrays(track_id) -> TrackArrays`,
    `track_ids`, `replace_from(track_id, frame_k)`, `truncate_after(track_id, frame_k)`,
    `save(path)` (atomic, through a file object), `ResultsStore.load(path)` (file closed after
    loading), a `version` key. Besides the §8.12 keys it keeps `core_fallback`, `second_fraction`,
    `core_r_px` (X5) and the bit-packed mask crops with their offsets (X15).
  - Tests first: save and load give equal arrays for two tracks, including the mask crops;
    `replace_from("A", k)` removes frames ≥ k of A only; `truncate_after("B", k)` keeps frames
    ≤ k; an unknown version raises a clear error; on Windows the file can be saved again right
    after loading.
  - Check: `uv run pytest tests/test_results.py -q`

- [ ] **A15 · Tracking: runs and the coarse runner** [CP] (§3.5, §6.1, §6.2, §6.4, §6.5, §13.2)
  - Files: `tracking.py`, `tests/test_tracking_coarse.py`.
  - Produces: `Job(session, run_folder, video_path, make_segmenter, track_ids=None)`;
    `Callbacks(progress, frame_result, log, finished, should_cancel)`;
    `plan_runs(session, store) -> list[RunPlan(track_ids, start_frame, mode, frames, input_box)]`;
    `run_job(job, callbacks) -> "complete" | "cancelled" | "failed"`; `dish_box(circle, frame_size)`
    (the square of §6.2 with m = 0.03 R, clipped); `FrameHashMismatch`.
  - Tests first (with `ExactFake` on `dish_scene()`):
    - positions equal the ground-truth centroids within 0.01 px with the dish crop on and off
      (frames flagged `EDGE` excluded); `ThresholdFake` on `disk_scene()` within 0.25 px;
    - the dish crop has integer corners (floor and ceil), clipped to the frame; a prompt point
      that falls outside the crop is dropped, and an object with no positive point left raises a
      plain error before the run;
    - coarse objects with the same start frame share one session; a later start frame makes a
      second run;
    - the hash guard aborts with `FrameHashMismatch` when the stored hash of a prompt frame
      differs from the decoded frame;
    - cancel after frame 10 keeps frames ≤ 10, saves, and leaves `"complete": false`;
    - autosave every 200 tracked frames writes `results.npz` and `session.json`;
    - progress reports done, total, s/frame and ETA;
    - review focus 2: a clip whose `end` lies past the last decodable frame finishes at the last
      frame and logs it; review focus 4: an object with an empty mask on every frame gives NaN rows
      and the run still completes; review focus 1: the clip in an odd folder name tracks.
  - Check: `uv run pytest tests/test_tracking_coarse.py -q`

- [ ] **A16 · Tracking: fine mode** (§6.3, §13.2)
  - Files: `tracking.py`, `tests/test_tracking_fine.py`.
  - Produces: `fine_window(preview_mask, factor=3.0) -> int` (clip(⌈3F⌉, 96, 512), F = maximum
    Feret diameter in px; 96 for an empty mask); the fine runner (fixed W, crop centered on the
    previous centroid, integer corner, `cv2.BORDER_REPLICATE` padding, one session per object).
  - Tests first (with `ExactFake` on `closeup_scene()`): positions within 0.01 px with the dish
    crop on and off; `fine_window` for F = 47 px is 141, for F = 14 is 96, for F = 300 is 512; the
    crop keeps its size when the object is at the frame border; `EDGE` when the mask touches the
    crop border; a lost frame keeps the last crop center and is marked lost; a fine object is not
    also tracked in coarse mode.
  - Check: `uv run pytest tests/test_tracking_fine.py -q`

- [ ] **A17 · Corrections and the edit API** (§5, §6.6, §13.2)
  - Files: `tracking.py`, `tests/test_corrections.py`.
  - Produces: `retrack_from(session, store, track_ids, frame_k, prompts)`,
    `end_track(session, store, track_id, frame_k)`,
    `new_piece(session, parent_id, frame_m, prompts) -> str` (next free piece: `A2`, `A3`, …),
    `next_track_id(session) -> str` (`A` … `Z`, `AA` …); each records a §8.10 correction entry.
    Also the Qt-free edit functions the GUI panels call, so the panels stay thin: `add_object`,
    `remove_object`, `add_prompt` (snaps to the grid, stores the frame hash), `undo_prompt`,
    `set_head`, `pending_runs`, `flags_table(run_folder)`.
  - Tests first: after `retrack_from(["A"], k)` only frames ≥ k of A changed: every other array
    in `results.npz` and every other u, v, x, y in positions.csv is identical (flags and θ are
    not compared, since the `SIZE` median and a head guess are whole-track quantities); `end_track("B", k)` leaves B's rows ending at k; `new_piece("A", m)` creates
    `A2` tracked from m, and the export has both `A` and `A2`; off-grid k and m snap to the grid
    with a note; ids match `^[A-Z]+[0-9]*$`.
  - Check: `uv run pytest tests/test_corrections.py -q`

- [ ] **A18 · Export: CSVs, outlines, log, README.txt** [CP: positions, Tracker format, log] (§8.1–8.6, §8.9, §8.11, §8.13, §13.2)
  - Files: `export.py`, `provenance.py` (tool version and commit from `direct_url.json`, machine
    facts), `tests/test_export.py`, `tests/test_provenance.py`.
  - Produces: `export_all(run_folder, overlay=False, log=print) -> ExportReport(files, warnings)`;
    it reads only `session.json` and `results.npz` and never imports torch.
  - Rules: `<model>/<id>.csv` is written by the ported `write_tracker_file` into a temporary file,
    then renamed, and that folder holds nothing but `<id>.csv` files (students' loaders read every
    CSV there). `radial.csv` and `outlines.npz` are always written; with no fine track they hold
    the header or `meta` only. Files of tracks that no longer exist are removed and logged.
    A partial store exports the frames it has.
  - Tests first:
    - every P0 file of §8.1 exists after a coarse-only run and after a run with a fine track;
    - rows: per track every grid frame from first to last, sorted by track then frame; lost frames
      keep their row with empty positions;
    - changing the stick length from 30 to 29 mm and exporting again scales every x and y by
      29/30 (1e-12 relative in memory, 1e-6 mm in the CSV, areas by (29/30)²) with no tracking
      call;
    - `<model>/*.csv` parse with the reference `shrimp.segment.read_tracker_export` and give the
      same x, y as positions.csv on visible rows;
    - descriptors from the full pipeline with `ExactFake` meet the A11 and A12 tolerances; the
      expected radial profile is the exact ray–ellipse intersection from the measured center and
      heading;
    - `CONTACT` appears on the contact frames of `dish_scene()`;
    - `outlines.npz` keys `<id>__frames`, `<id>__xy_mm`, `<id>__xieta_mm`, `meta` with N = 128;
    - run.log has each §8.9 block; the version line reads the commit from a git-install
      `direct_url.json` and prints no `file://` path for an editable install;
    - review focus 1: export into an odd folder name.
  - Check: `uv run pytest tests/test_export.py tests/test_provenance.py -q`

- [ ] **A19 · Overlay video** [CP] (§8.8 P0 content, §13.2)
  - Files: `overlay.py`, `tests/test_overlay.py`.
  - Produces: `write_overlay(run_folder, video_path, out_path=None, width=960)`: re-decodes the
    clip, draws per track the stored outline, centroid dot and id in the track color, and the
    stamp `t = … s   frame …`; H.264 yuv420p, 30 fps, even dimensions, atomic.
  - Tests first: the file decodes with exactly one frame per tracked grid frame; it is 960 px wide
    with even height; a pixel on a known outline has the track's color; after `retrack_from` the
    overlay shows the new outline; it is written through `atomic_write` with a name ending in
    `.mp4`.
  - Check: `uv run pytest tests/test_overlay.py -q`

- [ ] **A20 · Brightness probes** (§4.6, §8.7, §13.2)
  - Files: `probes.py`, `tests/test_probes.py`.
  - Produces: `measure_probes(video_path, boxes, start, end, fps_true) -> DataFrame` with the §8.7
    columns: every frame from start to end inclusive (step 1), mean R, G, B and
    gray = 0.299 R + 0.587 G + 0.114 B over the pixels whose centers lie inside each box; one
    sequential decode, no model.
  - Tests first: on `dish_scene()` the LED onset (first frame where gray exceeds the midpoint) is
    exactly the scene's onset frame; integer corners `u0, v0, u1, v1` select the array slice
    `[v0:v1, u0:u1]`; an empty box raises `ValueError`; two boxes give two rows per frame;
    review focus 2: a clip shorter than `end` stops cleanly.
  - Check: `uv run pytest tests/test_probes.py -q`

### Phase B: commands (Tue night → Wed morning)

- [ ] **B1 · `from-tracker`: the fallback** [CP] (§11, §14.1; X3, X9)
  - Files: `from_tracker.py`, `cli.py`, `tests/test_from_tracker.py`.
  - Command: `outline-tracker from-tracker VIDEO EXPORT [--fine IDS] [--step K] [--seconds S]
    [--fps F] [--out DIR] [--student NAME] [--model edgetam|sam2] [--device D] [--no-overlay]`
    (`--no-video` is accepted as last week's spelling).
  - Produces: `from_tracker(video, export, *, fine_ids=(), step=None, seconds=None, fps=None,
    out=None, student=None, model="edgetam", device="auto", overlay=True, segmenter=None,
    log=print) -> FromTrackerResult` (run folder, files, overlay, flags, plan, seconds per frame). It builds a session from the ported `make_plan`
    (clip = plan start, last frame, step; origin, angle and scale from `fit_calibration` through
    `world_frame_from_calibration`; one positive click per object), runs it on the full frame
    (no circle, so no dish crop), exports, and writes the overlay.
  - Defaults: the run folder is `--out`, else `<video folder>/<video stem>_outline_<student>/`
    (§8.1). `--student` defaults to the name of the export's home folder (last week's per-student
    folder, e.g. `ana`), and the command prints which name it used. It refuses to write into a
    folder that already holds other CSV or TXT files ("this looks like a folder of Tracker files").
    Every shrimp must be marked on the same first frame (the ported error message stays).
    Track names are the export's names, unchanged; file names are sanitized as last week. An
    export that starts beyond the video's last frame gives a plain error (last week: a crash).
    The session records the scale as `calibration.tracker_fit` (there is no stick).
  - Tests first:
    - the two end-to-end tests of `test_segment.py` with `ThresholdFake`: same frames as the
      Tracker track, `t = frame/240`, x and y within 0.1 and 0.25 × MM_PER_PX, overlay written,
      `run.log` present, no flags; files are found in `<run>/stand-in/` (X3);
    - with the reference `DiskFinder` given to `shrimp.segment.track_video` and `ThresholdFake`
      given to `from_tracker` on the same clip and export, the Tracker-format CSVs are identical
      text on this platform;
    - a mirrored calibration is refused with a plain message; an unknown `--fine` id lists the
      available ids; `--fine B` gives B a fine run with a window from its first-frame mask;
    - fps sources in the X9 order, each named in `session.json` and `run.log`; a warning below
      100 fps;
    - the full §8 file set exists; `--no-overlay` skips only the overlay;
    - Ctrl+C (a `KeyboardInterrupt` raised by the stand-in at frame 20) still exports what was
      tracked;
    - the refusal for a folder with foreign CSVs; review focus 1 (odd folder names).
  - Check: `uv run pytest tests/test_from_tracker.py -q`, and
    `uv run outline-tracker from-tracker --help` lists the options above.

- [ ] **B2 · `selftest`** (§11, §13.4)
  - Files: `cli.py` (or `selftest.py`), `tests/test_selftest.py`.
  - The template's selftest, ported: the same made-up 1080p clip, the same OK / PROBLEM line and
    time estimate, run through `from_tracker`. `selftest [--model M] [--device D]`.
  - Tests first: the ported `test_selftest_runs_and_reports_the_time` with `ThresholdFake`
    (`ok`, max error < 1 px, `minutes_ten > minutes_one > 0`).
  - Check: `uv run outline-tracker selftest` ends with `OK: edgetam followed the test shrimp
    within … pixels (should be under 3).` (about a minute; the first run downloads the model).

- [ ] **B3 · `export` and `compare-tracks`** (§11; X10)
  - Files: `cli.py`, `tracker_io.py` (`compare_tracks(new_dir, old_dir) -> DataFrame`),
    `tests/test_cli_export.py`.
  - `export SESSION.json [--overlay]` regenerates every output from `results.npz` and the session;
    without `--overlay` an existing overlay is left alone. `compare-tracks NEW_DIR OLD_DIR
    [--limit PX]` (hidden from `--help`) pairs files by name, joins on frame, and prints per track
    the common frames, RMS and maximum difference in px and where it occurs; exit code 1 above
    the limit (default 2.0).
  - Tests first: after editing `length_mm` in `session.json`, `export` rescales x and y and loads
    neither torch nor Qt; without the video present it still writes every CSV and says the overlay
    was skipped (review focus 5); a partial session exports its frames; `compare-tracks` on two
    folders written with a known 0.3 px offset reports RMS 0.300.
  - Check: `uv run pytest tests/test_cli_export.py -q`

- [ ] **B4 · `probe`** (§4.6, §11)
  - Files: `cli.py`, `tests/test_cli_probe.py`.
  - `probe VIDEO --rect NAME:u0,v0,u1,v1 [...] [--start F] [--end F] [--fps F] [--out DIR]
    [--student NAME]` and `probe SESSION.json`. Defaults: start 0, end = last frame, both
    inclusive; fps from `--fps`, else the manifest, else an error (never the file's frame rate);
    output in `--out`, else the §8.1 run folder for `--student`; with neither, a message showing
    both forms.
  - Tests first: on `dish_scene()` the onset frame in `probes.csv` is exact; two `--rect` options
    give two probes; a missing fps gives the ported "Unknown fps_true" message; `probe SESSION.json`
    uses the session's boxes and clip.
  - Check: `uv run pytest tests/test_cli_probe.py -q`

- [ ] **B5 · Real-model tests of the full pipeline** (§13.4)
  - Files: `tests/slow/test_fine_mode.py`, `tests/slow/test_coarse_lowres.py`, `docs/VALIDATION.md`.
  - Tests first (slow; `cpu`, and the fine test once on `mps`):
    - fine mode on `closeup_scene()` (9 Hz, 240 fps, step 1, 2 s): the solidity spectrum (mean
      removed) peaks within ±0.5 Hz of 9 Hz; RMS difference between measured and true solidity
      < 0.02 (an ideal 256-cell grid alone gives 0.003); `shape_ok` = 1;
    - coarse mode on `dish_scene()` flags `LOWRES`;
    - `from-tracker` on the selftest clip with the real model writes the full §8 set.
  - Numbers, run times (s/frame on cpu and mps) and peak memory go into docs/VALIDATION.md. A
    failing test is reported there with numbers and images and listed under "Questions for J".
  - Check: `uv run pytest -m slow -q` (tens of minutes), or read docs/VALIDATION.md.

- [ ] **B6 · Fallback instructions** (§14.1, §16)
  - Files: `README.md` (install from the repo, `selftest`, `from-tracker` next to last week's
    command, where the files land, how to load `<run>/edgetam` with last week's `load_tracks`),
    `docs/OUTPUTS.md` (generated from `schema.py`; a test checks it is up to date).
  - Check: `uv run pytest tests/test_schema.py -q`; read README.md "If the app does not open".

---

### ▶ GO/NO-GO 1 (Wed ~noon): J tests `from-tracker` on a real clip

The agent posts the final commands in chat when Phase B is done. Draft (Terminal, zsh):

```zsh
setopt interactivecomments
# 0. edit these five lines
OT=~/Developer/outline-tracker                      # the repo folder
VIDEO="/path/to/groupX_2026-09-29_HHMM_main_tracker.mp4"
OLD="/path/to/group-repo/data/tracks/groupX_2026-09-29_HHMM_main/NAME"   # holds extra/start.csv and edgetam/
FPS=239.6                                           # fps_true of that video
NEW=~/outline-gng1                                  # new, empty, not in iCloud

# 1. the tool runs
cd "$OT" && git pull && uv sync
uv run outline-tracker --version
uv run outline-tracker selftest
uv run outline-tracker selftest --device cpu
uv run outline-tracker check "$VIDEO" --seek

# 2. new tool, 2 s, compared with last week's files
uv run outline-tracker from-tracker "$VIDEO" "$OLD/extra/start.csv" --seconds 2 --fps $FPS --out "$NEW/run"
ls -lh "$NEW/run" "$NEW/run/edgetam"
uv run outline-tracker compare-tracks "$NEW/run/edgetam" "$OLD/edgetam"
open "$NEW/run/overlay.mp4"
tail -n 40 "$NEW/run/run.log"

# 3. control on this Mac: new tool against last week's script, cpu, 1 s
uv run outline-tracker from-tracker "$VIDEO" "$OLD/extra/start.csv" --seconds 1 --fps $FPS --device cpu --no-overlay --out "$NEW/cpu"
uv run python -c "import sys; sys.path.insert(0, 'tests/reference'); from shrimp import segment; raise SystemExit(segment.main(sys.argv[1:]))" "$VIDEO" "$OLD/extra/start.csv" --seconds 1 --fps $FPS --device cpu --no-video --out "$NEW/ref/edgetam"
uv run outline-tracker compare-tracks "$NEW/cpu/edgetam" "$NEW/ref/edgetam" --limit 0.01

# 4. commands that need no model
uv run outline-tracker export "$NEW/run/session.json"
uv run outline-tracker probe "$VIDEO" --rect LED1:100,100,140,140 --end 479 --fps $FPS --out "$NEW/probe"
head -n 3 "$NEW/probe/probes.csv"
```

What J should see:

- Step 1: `outline-tracker 0.1.0 (commit …)`; each selftest ends with `OK: edgetam followed the
  test shrimp within X.X pixels (should be under 3).` and a time estimate; `check --seek` ends
  with `OK` and `seek: 20 of 20 frames exact` (if not, the app still works, with slower
  scrubbing; report the line).
- Step 2: the same console lines as last week (number of shrimp, frames, scale, progress, any
  `CHECK:` lines). The run folder holds `session.json`, `positions.csv`, `shapes.csv`,
  `radial.csv` (header only), `outlines.npz` (tiny), `overlay.mp4`, `results.npz`, `run.log`,
  `README.txt`, and `edgetam/` with one CSV per shrimp. `compare-tracks` prints one line per shrimp
  and `OK: worst RMS … px (limit 2 px)`. Expect a fraction of a pixel; it is not bit-identical
  because last week's files came from another machine or device. One shrimp far above 2 px with a
  clear frame number means the two runs diverged at a contact: look at that frame in the overlay.
- Step 3: `OK: worst RMS 0.000 px (limit 0.01 px)`. Same Mac, same libraries, same weights, cpu:
  this isolates the port. If step 3 passes and step 2 does not, the difference is numerics, not
  the port. **If step 3 fails, it is no-go for the port: report the output.**
- Step 4: `export` finishes in seconds and loads no model; `probes.csv` has 480 rows.
- The overlay: about 8 s at 30 fps, each shrimp with outline, dot and letter, and the time stamp.

Go = step 3 passes, step 2 is within 2 px RMS for shrimp that stay tracked, the overlay looks right.

---

### Phase C: GUI, P0 (Wed afternoon)

Conventions for every GUI task:

- Tests use pytest-qt offscreen and the stand-in segmenters; the window takes a segmenter factory,
  so no GUI test imports torch. The app is runnable after every task.
- Widgets expose public slots with explicit arguments; file and message dialogs live only in thin
  button wrappers. Simulated mouse clicks are used only where the click itself is under test.
- One worker object on one thread (X7). Its signals connect only to bound methods of objects
  living in the GUI thread, never to lambdas or partials (those run in the worker thread).
  Cancel is a `threading.Event` that `run_job` checks between frames.
- `session.json` is written only from the GUI thread, `results.npz` only by the worker.
- No Qt object is created at import time. Dialogs go through one helper that uses `open()`, not
  `exec()`, and that tests replace with a recorder. Every GUI test closes its window through a
  fixture that stops the worker and releases the video file.

- [ ] **C1 · Video view and navigation** (§10.1, §13.6)
  - Files: `gui/main_window.py`, `gui/video_view.py`, `tests/gui/test_video_view.py`.
  - Content: pyqtgraph `ImageItem` in a `ViewBox` (`row-major`, `invertY(True)`, aspect locked,
    context menu off, no auto-levels, wheel zoom, drag pan, Fit, 1:1); Open video; bottom bar
    (grid slider, first, −10, −1, +1, +10, last, frame box, t); keys ←/→, Shift+←/→, Home/End,
    Esc; the nine panels in a scrollable dock (they must fit a 1366 × 768 laptop).
  - Tests first: a synthetic clip loads; the slider only lands on grid frames; the arrow keys
    move ±1 and ±10 steps; a click in the middle of pixel (c, r) reports (c + 0.5, r + 0.5) with
    no extra ½, also at display scale 1.5 and 2; a click outside the image is ignored; the image
    shown for frame k equals `FrameSource.get(k)`.
  - Check: `uv run outline-tracker gui ~/closeup_tracker.mp4` shows the clip; arrows step, the
    wheel zooms.

- [ ] **C2 · Student, video, clip, time, session saving** (§2 steps 1–4, §4.1, §8.10, panels 1–2)
  - Files: `gui/panels/video_panel.py`, `gui/panels/time_panel.py`, `gui/session_controller.py`,
    `tests/gui/test_session_panels.py`.
  - Content: student name (required before Track and Export); file info and the `check_video`
    warnings; clip start, end, step (default 2); fps_true from the manifest (path remembered in
    QSettings) or typed, with its source shown and a warning below 100; Open session; the session
    is saved 0.75 s after any change, before a run, on export, on Ctrl/Cmd+S and on exit.
  - Tests first: changing a widget updates the `Session`; the debounced save writes
    `session.json`; closing and reopening restores every field; Track and Export are refused with
    a plain message while the name is empty.
  - Check: open a clip, type a name and fps, quit, reopen with
    `uv run outline-tracker gui "<run folder>/session.json"`: everything is back.

- [ ] **C3 · Calibration stick, tape, circle, axes** (§4.2–4.5, panels 3–4)
  - Files: `gui/tools.py`, `gui/panels/calibration_panel.py`, `gui/panels/dish_panel.py`,
    `tests/gui/test_tools.py`.
  - Content: tool modes Pan, Stick, Tape, Circle, Axes (click to place; Redo clears; Esc returns
    to Pan); graphics for stick, tape, circle points and fitted circle, axes arrows with labels;
    scale shown as "32.40 µm/px ± 0.08%" with a warning under 300 px; tape error in green or red
    (1% rule); circle result (center, R in px and mm, RMS, 2R next to `dish_mm`); Origin to
    Center; axes angle; "Crop to dish for tracking" (default on).
  - Tests first, through `add_tool_point(u, v)` and the panel slots: stick endpoints 926 px apart
    with 30 mm show 32.40 µm/px ± 0.08%; a 20 mm tape check within 1% is green, outside is red;
    six circle points give the center and radius of the fit and Origin to Center copies the center
    to the axes; Redo clears; every value lands in the session.
  - Check: on a real or synthetic clip, place the stick, the tape, the circle; the numbers appear
    and survive a restart.

- [ ] **C4 · Objects, prompts, preview** (§5, panel 6, §10.2)
  - Files: `gui/panels/objects_panel.py`, `gui/prompts.py`, `gui/worker.py` (model loading and
    preview), `tests/gui/test_prompts.py`.
  - Content: the object table (id, color, mode, fine window, start frame, status), Add, Remove;
    tools Positive, Negative, Head; a click is negative when it is a right click (on a Mac a
    two-finger click and a Control-click both arrive as one), or a left click with Alt/Option, or
    with the physical Control key (`MetaModifier` on macOS, `ControlModifier` elsewhere);
    Cmd-click on a Mac stays positive; undo of the last prompt (the platform's Undo shortcut);
    prompts sit on grid frames and store the frame hash; the model loads in the worker after the
    window appears and tracking buttons stay disabled until it is ready; after each prompt change
    a preview runs in the worker (latest request wins) with a busy indicator, and the masks of
    all objects prompted on that frame are drawn.
  - Tests first (stand-in segmenter injected): a left click adds a positive point at the clicked
    (u, v) on the current grid frame with its frame hash; `is_negative_click(button, modifiers,
    platform)` for every case above on both platforms; a head click fills `head_px` and is not
    passed to the segmenter; undo removes the last prompt; the preview mask appears without
    blocking the event loop; the slot that receives it runs in the GUI thread; Track is disabled
    until the model reports ready.
  - Check: click a shrimp: an outline appears within a second or two; right-click a neighbor: the
    outline lets go of it; Cmd+Z brings it back.

- [ ] **C5 · Track in the background** (§6.4, panel 7)
  - Files: `gui/worker.py`, `gui/panels/track_panel.py`, `gui/overlays.py`,
    `tests/gui/test_tracking_panel.py`, `tests/slow/test_gui_worker_real_model.py`.
  - Content: the worker runs previews and jobs one at a time with
    `torch.set_num_threads(max(1, n − 1))` set once; the estimated time before the run (seconds
    per frame from the last selftest, run or preview × frames × (1 + 0.5 per extra object));
    Track, progress bar, s/frame, ETA, Cancel; autosave every 200 frames; outlines, centroids,
    ids and head marks drawn from the results store on the current frame.
  - Tests first: a run with `ExactFake` completes and emits progress; a timer on the UI thread
    keeps firing during the run; Cancel keeps the frames done and leaves `"complete": false`; the
    estimate is shown before the run starts; the outline drawn on frame k is the stored outline
    of frame k; closing the window during a run cancels it and exits cleanly.
    Slow: the real model runs through the GUI worker on the 20-frame selftest clip, on cpu and
    on mps, without crashing.
  - Check: Track 2 s at step 2: estimate, progress and ETA show; the window stays usable; Cancel
    stops and keeps what was tracked.

- [ ] **C6 · Export and the smoke test** (§10.1 panel 9, §13.5 P0)
  - Files: `gui/panels/export_panel.py`, `tests/gui/test_smoke.py`.
  - Content: run folder (default from §8.1, changeable), Export all in the worker, Open folder.
  - Tests first: the §13.5 flow in one test: the window opens, a synthetic clip loads,
    calibration, circle and origin are set through public slots, one object is added by a
    simulated click, a run with `ExactFake` completes, Export writes every P0 file of §8, the
    session reopens in a new window with everything restored, and no error was recorded.
  - Check: `uv run pytest tests/gui -q`; in the app, Export all, then Open folder shows the
    files and `overlay.mp4` plays.

- [ ] **C7 · Review and fix** (§6.6, §9, panel 8)
  - Files: `gui/panels/review_panel.py`, `tests/gui/test_review.py`.
  - Content: flags table (track, frame, t, code) whose rows jump to the frame; previous and next
    flag for the selected track; Re-track from here; End track here; Continue as new track.
  - Tests first (on `dish_scene()`): the table lists the `CONTACT` rows; activating a row moves
    the slider there; next and previous cycle through the selected track's flags; Re-track from
    here changes only frames ≥ k of that track; End track here then Continue as new track gives
    `A` and `A2` in the table and in the export.
  - Check: jump to a flag, re-click the shrimp, Re-track from here; end another track and continue
    it as A2.

- [ ] **C8 · Finish the window; quickstart** (§10.1, §10.2, §13.7, §16)
  - Files: `gui/dialogs.py`, `gui/panels/*.py`, `tests/gui/test_finish.py`, `README.md`
    (10-step quickstart matching the panels), `docs/DEVELOPER.md`.
  - Content, most important first: error dialogs that say what to do next in plain words, with
    the stack trace only in `run.log` (also for uncaught errors in the worker); status bar
    (cursor in px and mm, gray value, device, model state, last message); menus (File: Open
    video, Open session, Save session, Save session as, Export, Quit; Help: Quickstart, About
    with all versions); play/pause (Space, one grid step per tick at 30 Hz); the Stopwatch…
    dialog; model and device selectors; mask-fill toggle; keys 1–9 to select an object.
  - Tests first: a failure inside the worker shows a dialog without the word "Traceback" while
    `run.log` holds the trace; the cursor readout in mm follows the axes; the stopwatch dialog
    gives (f_b − f_a)/(t_b − t_a) and an error for equal readings; About lists the tool, Python,
    torch, transformers and PySide6 versions; Space plays and pauses on the grid.
  - Check: `uv run pytest tests/gui -q`; `gh run list --limit 1` shows both CI jobs green; read
    the README quickstart next to the app.

---

### ▶ GO/NO-GO 2 (Wed evening): J runs the §14.3 checklist

The agent posts the final commands and a filled-in checklist in chat when Phase C is done. Draft.

**On the Mac (Terminal, zsh), about 1 h:**

```zsh
setopt interactivecomments
cd ~/Developer/outline-tracker && git pull && uv sync
uv run outline-tracker selftest                 # default device: the Apple GPU
uv run outline-tracker selftest --device cpu
uv run outline-tracker                          # opens the app: do items 2-10 below
```

The agent also names a commit to install with `uv tool install --force --python 3.12
"git+https://github.com/jd-anabi/outline-tracker@<commit>"`, so the Mac is tested the way
students install. On the trackpad, try all three negative clicks in item 3: two-finger click,
Control-click, Option-click.

Then, in the app, the §14.3 items in order:

2. Open a real day-1 `_tracker.mp4`: fps_true is read from the manifest; a 30 mm stick passes a
   20 mm tape check within 1%; a 6-point circle fit sets the origin.
3. Click 3 shrimp on one frame; separate two touching shrimp with one negative click; the preview
   is right.
4. Track 2 s at step 2, coarse: estimate, progress, ETA; the window stays responsive; Cancel keeps
   what was done; flags appear.
5. At a flagged frame re-click one shrimp and Re-track from here; end another track and continue
   it as A2.
6. Export. Then, in Terminal (`RUN` = the run folder, `OLD` = last week's folder for this video):
   ```zsh
   ls "$RUN" "$RUN/edgetam"
   uv run outline-tracker compare-tracks "$RUN/edgetam" "$OLD/edgetam"
   ```
   Every P0 file exists; the RMS is about 2 px or less for the same shrimp and click; and in a
   group repo `load_tracks("$RUN/edgetam")` loads the files.
7. Fine mode on a close-up clip, or the synthetic one
   (`uv run outline-tracker synth closeup ~/closeup_tracker.mp4`): at 1:1 zoom the outline follows the antennae; `shape_ok` = 1 in
   `shapes.csv`; `radial.csv` shows the lobes.
8. Change the stick length, Export again: x and y rescale exactly, with no tracking.
9. `overlay.mp4` plays in PowerPoint or Google Slides.
10. Close and reopen the session: everything is restored.

**On the Windows machine (PowerShell), about 20 min.** The repo is still private, and uv turns
git's password prompt off, so sign in to GitHub first (step 1):

```powershell
# 0. tools from last week
uv --version
git --version

# 1. sign in to GitHub once: a browser window opens; afterwards this prints one line with a hash
git ls-remote https://github.com/jd-anabi/outline-tracker.git HEAD

# 2. install the way students will; SHA = the commit the agent names in chat
$SHA = "<commit>"
uv tool install --force --python 3.12 "git+https://github.com/jd-anabi/outline-tracker@$SHA"
outline-tracker --version

# 3. self-test (the first run downloads PyTorch and the model: a few minutes)
outline-tracker selftest

# 4. a test clip, then the app: it must open, show "model ready", and track
outline-tracker synth dish "$env:USERPROFILE\Videos\ot_dish_tracker.mp4" --seconds 3
outline-tracker "$env:USERPROFILE\Videos\ot_dish_tracker.mp4"
```

In the app on Windows: type a name and fps 240, click two objects, Track 2 s in coarse mode,
Export. A real `_tracker.mp4` works as well. If step 1 does not open a browser:
`winget install --id GitHub.cli -e`, then `gh auth login --web --git-protocol https` and
`gh auth setup-git`. If `outline-tracker` is "not recognized": `uv tool update-shell`, then open
a new PowerShell.

What J should see on Windows: the version line with the commit; `OK: edgetam followed the test
shrimp …`; the window opens with no `WinError 1114` in PowerShell (the import-order check of
§10.2); the status bar says the model is ready; the run finishes and Export writes the files.

Go = items 1–10 pass. A failure on Windows only, or in items 2–5, means Thursday uses
`from-tracker` (tripwire 5).

---

### Phase D: P1 (Wed night, in the §14.2 order; each is cut if time runs out)

- [ ] **D1 · GUI probe tool** (§4.6, panel 5) — add a box with two corner clicks, name it,
  Measure in the worker. Test: the box lands in the session and `probes.csv` has the scene's onset.
  Check: draw a box over the LED, Measure, open `probes.csv`.
- [ ] **D2 · Live view** (§6.4) — the latest tracked frame with outlines, at most twice a second.
  Test: during a run the view updates and never more often than every 0.5 s.
- [ ] **D3 · Draggable handles** (§10.1) — stick, tape, circle points, origin, probe corners.
  Test: moving a handle through its slot updates the session and the readouts.
- [ ] **D4 · Flag timeline strip** (§9) — colored marks under the slider. Test: marks sit at the
  flagged frames of the selected track.
- [ ] **D5 · Overlay extras** (§8.8 P1) — trail of the last 0.5 s, head tick, dish circle, axes,
  1 mm scale bar, full-resolution option, `overlay_<id>_zoom.mp4` per fine track (512 × 512).
  Tests: the zoom overlay has one frame per tracked frame at 512 × 512; the scale bar is 1/k px long.
- [ ] **D6 · Calibration export and import** (§4.7) — `calibration.json` with fps_true and its
  source, stick, check, circle, axes, video size and hash. Tests: round trip; importing onto a
  different video (size or hash) warns.
- [ ] **D7 · Resume after cancel** (§6.4) — "Re-track from here" at the first untracked frame with
  an automatic positive click at each object's last centroid. Test: resume completes the track.
- [ ] **D8 · `run SESSION.json [--from-frame K]`** (§11) — every pending run, headless. Test: a
  session saved from the GUI with pending objects is tracked and exported by the command.
- [ ] **D9 · GUI "Convert for tracking"** (§10.1 panel 1) — runs the ported converter in the
  worker for `.MOV` files. Test: the button produces `<stem>_tracker.mp4` and opens it.
- [ ] **D10 · Full GUI flow test and screenshots** (§13.5 P1) — jump to a `CONTACT` flag and
  re-track from there; `scripts/screenshots.py` saves a PNG per step into `docs/screenshots/`.
  Check: `uv run python scripts/screenshots.py && open docs/screenshots`.
- [ ] **D11 · `selftest --fine`** (§11) — adds the synthetic close-up shrimp. Test (slow): prints
  OK when the solidity peak is within 0.5 Hz of 9 Hz.
- [ ] **D12 · docs/VALIDATION.md** (§13.4, §7.8) — selftest and synthetic results with numbers,
  the `shape_ok` threshold and why, run times on cpu and mps, peak memory for 10 coarse objects.

### Phase E: release (Thu morning, with J)

- [ ] **E1 · Fixes from go/no-go 2.**
- [ ] **E2 · README, CHANGELOG, LICENSE** (§16, §15) — install; 10-step quickstart; outputs; the
  three things that go wrong (`CONTACT`, `LOWRES`, calibration); troubleshooting (first run needs
  internet, Intel Macs unsupported, `uv tool update-shell`, files open in Excel); how to cite
  SAM 2 and EdgeTAM. Check: read README.md top to bottom as a student.
- [ ] **E3 · Exact pins and tag** (§15) — pin with `==` the versions in `uv.lock` that passed the
  tests and the Windows check (torch, torchvision, transformers, timm, numpy,
  opencv-python-headless, scikit-image, PySide6-Essentials, pyqtgraph, imageio-ffmpeg); CI green;
  build the wheel and install it with `uv tool install` into a clean environment on the Mac;
  tag `v0.1.0` and push the tag when J says go.
  Check: `uv tool install --force --python 3.12 dist/outline_tracker-0.1.0-py3-none-any.whl &&
  outline-tracker selftest`.
- [ ] **E4 · Public install check** — after J makes the repo public:
  `uv tool install --force --python 3.12 git+https://github.com/jd-anabi/outline-tracker@v0.1.0`,
  then `outline-tracker selftest` and `outline-tracker`, on the Mac and on Windows.

---

## 8. Which task owns which test of SPEC §13

| §13 item | task |
|---|---|
| 13.1 Transform; Stick, tape, fps; Circle fit; Grid snapping | A06 |
| 13.1 Centroid | A03 |
| 13.1 Moments (pixel part), Core | A11 |
| 13.1 Moments (world angle), Head continuity, Outline geometry, Radial profile, Solidity and wall distance, Resolution | A12 |
| 13.1 Flags | A13 |
| 13.1 Schema | A05 |
| 13.1 Session | A07 |
| 13.2 synthetic clips and ground truth | A08 |
| 13.2 stand-ins (`ExactFake` 0.01 px, `ThresholdFake` 0.25 px) | A10, A15, A16 |
| 13.2 coarse and fine, dish crop on and off | A15, A16 |
| 13.2 descriptors, `CONTACT`, export rescaling 29/30, Tracker-format files | A18 |
| 13.2 probes onset | A20, B4 |
| 13.2 overlay frame count | A19 |
| 13.2 re-track, end track, new piece | A17 |
| 13.2 frame-hash guard | A15 |
| 13.3 regression against last week's script | A04 (real model), B1 (stand-in, text-identical files) |
| 13.4 selftest, negative prompt, mps runs, step fallback | A04 |
| 13.4 fine mode, coarse `LOWRES` | B5 |
| 13.5 GUI smoke test (P0) | C6 |
| 13.5 full flow and screenshots (P1) | D10 |
| 13.6 frame exactness | A09 (both CI runners), C1 |
| 13.7 CI on ubuntu and windows | A01, A01b |

Where the 27 template tests live (X2, X3):

| template test | new home | change |
|---|---|---|
| `test_video.py` (8) | `tests/test_video.py` | import line |
| `test_convert.py` (2) | `tests/test_convert.py` | import line |
| `test_segment.py`: reading (3), calibration (6), written file (1), plan (3) | `tests/test_tracker_io.py` | import line |
| `test_segment.py`: `mask_center` (1) | `tests/test_measure.py` | import line |
| `test_segment.py`: whole run, many shrimp from a start file (2) | `tests/test_from_tracker.py` (B1) | called through `from_tracker` with `ThresholdFake`; same frames, times, positions and tolerances; files looked up in the §8.1 run folder |
| `test_segment.py`: selftest (1) | `tests/test_selftest.py` (B2) | called through the ported selftest with `ThresholdFake`; same assertions |
| all 27, unmodified | `tests/reference/template_tests/` | none; run against the reference copy by `test_template_tests_pass_on_reference` (A01) |
