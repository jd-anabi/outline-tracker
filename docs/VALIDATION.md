# Validation

Results of the slow tests: the real model on short synthetic clips (SPEC 13.3 and 13.4). Each
section names the command that produced its numbers. A test that fails is reported here with
numbers and images; nothing is tuned until it passes.

## 1. The Hugging Face backend against last week's script (task A04a, 2026-10-06)

**What was compared.** `outline_tracker.segmenter.hf.HFSegmenter` and last week's
`shrimp.segment.TransformersSegmenter` (the unmodified copy in `tests/reference`). One loaded
EdgeTAM was given to both, on `cpu`, in one process and with the same number of torch threads
(8, the default here). Every object got one positive click, on the first frame.

**Setup.** An Apple-silicon laptop with macOS 27; Python 3.12.15, torch 2.14.1, torchvision
0.29.1, transformers 5.18.0, timm 1.0.30, opencv-python-headless 5.0.0.93. EdgeTAM is Meta's
`edgetam.pt` (facebook/EdgeTAM), converted on first use. The converted `model.safetensors` has
55,912,824 bytes and the SHA-256
`8858f8e4757b0b96dab8763f296ecffd845efbbbf698f64163cfa20a63d5fff4`; `HFSegmenter.weights_sha256`
reports this value.

**Command.**

```
uv run pytest -m slow tests/slow/test_regression_reference.py -q -rP
```

**Result.** 9 passed, twice: 96 s for the first run, which downloaded and converted the model,
and 60 s for the second.

### 1.1 Regression (SPEC 13.3): positions within 0.01 px of last week's

| clip | objects | frames | max difference | mask pixels that differ | lost frames (last week / new) |
|---|---|---|---|---|---|
| selftest clip of `shrimp.segment.selftest` | 1 | 0, 2, …, 38 (20 frames) | 0.0005 px | not compared | 0 / 0 |
| three-ellipse clip | 3 | 0, 2, …, 38 (20 frames) | 0.0000 px | 0 of 3 × 20 masks | 0 / 0 |

- Selftest clip: last week's positions are read from the CSV that its `selftest` wrote (three
  decimals), so 0.0005 px is the rounding of that file.
- Three-ellipse clip: 1080p, made with the selftest's recipe, three dark ellipses at least 471 px
  apart on a background whose red, green and blue differ. Last week's segmenter was run directly,
  so the masks themselves were compared: they are identical on every frame.

### 1.2 Selftest criterion (SPEC 13.4): max error under 3 px against the true positions

| device | code | max error | s per frame, mean of 20 | s per frame, median |
|---|---|---|---|---|
| cpu | new backend | 0.460 px | 0.37 | 0.37 |
| cpu | last week's, same loaded model | 0.461 px (from its CSV) | 0.37 to 0.39 | not measured |
| mps | new backend | 0.460 px | 0.44 (first run), 0.14 (second run) | 0.15, 0.12 |

- The mps run stayed on the Apple GPU for all 20 frames: no fall back to the processor.
- The mps positions differ from the cpu positions by at most 0.132 px. That is expected: the two
  devices already differ in the resized input image, so mps can meet a tolerance but never
  reproduce cpu exactly.
- Three objects on cpu took 0.73 s per frame (median 0.75); their max error against the true
  positions was 1.042 px.
- Timing: s per frame includes decoding the frame. Another test job was running on the same
  laptop, so the values may be a little high. The mean of the first mps run contains the warm-up
  of the first frames.

### 1.3 Checks that need no weights

The same test file runs both classes with the real processor and session and a stand-in for the
network (seconds, no download):
- the stored prompt tensors have shape (1, 1, 1, 2), labels ≥ 0, and the session's list of objects
  with new inputs names all three objects;
- for three one-click objects, the session after the new code's one call per object equals,
  attribute by attribute, the session after last week's single call;
- on random 256 × 256 logits and on compact blobs (one object absent), the new masks
  (`post_process_masks(binarize=False) > 0`, one object at a time, cropped to the bounding box
  ± 8 px) equal last week's `binarize=True` masks on every pixel.

### 1.4 Can the regression test fail?

With the mask threshold deliberately set to `logits > 0.5` in the new code, the same tests failed:
0.1176 px on the selftest clip and 0.2607 px on the three-ellipse clip (216 mask pixels differed).
The threshold was then set back. A change of a few boundary pixels is therefore visible at 0.01 px.

### 1.5 Not covered by this section

- SAM 2.1 (`sam2`, `sam2-small`): the weights were not downloaded and the model was not run.
- Pruning with the real model: both clips have 20 tracked frames, fewer than the 21 results the
  pruning keeps. The pruning code is last week's, unchanged, and is tested on plain dictionaries.
- The fall back from mps to cpu was not triggered by the model. The fall back in `start()` is
  tested with a stand-in; the one in `step()` belongs to task A04b.

## 2. Backend extensions: negative prompts, preview, the fall back in `step()` (task A04b, 2026-10-06)

**What was tested.** `HFSegmenter` with several points per object (labels 1 and 0), its
`preview()`, and its fall back from the Apple GPU to the processor in `step()`. Same laptop,
versions and weights as section 1. Scenes are 1080p frames made in memory in the selftest's style
(bright background with a vignette, noise of sigma 3, dark ellipses whose edge pixels are darkened
by the covered fraction); the ground truth is the ellipse itself.

**Command.**

```
uv run pytest -m slow tests/slow/test_real_model.py -q -rP
```

**Result.** 8 passed in 30 s (5 tests without weights, 3 with the real EdgeTAM). The whole slow
folder, `uv run pytest -m slow tests/slow -q -rP`: 17 passed in 82 s; the numbers of section 1 are
unchanged (0.0005 px and 0.0000 px against last week's script).

### 2.1 Negative prompt (SPEC 13.4): overlap with the neighbor under 5% of its area

Two equal ellipses, semi-axes 40 and 15 px, both turned by 30°, touching flank to flank: the
second center lies at 45° on the ellipse with twice the semi-axes around the first, which is the
exact condition for two equal parallel ellipses to touch. Rasterized at pixel centers they have
1885 px each (π·40·15), share no pixel, and come within 2 px of each other. `preview()` on `cpu`,
one object:

| prompt | mask | part of the neighbor covered | part of the clicked ellipse covered |
|---|---|---|---|
| **positive at the center of one ellipse, negative at the center of the other** | 1885 px | **0.0%** (limit 5%) | 98.6% |
| positive click alone | 1906 px | 0.0% | 99.1% |
| both clicks positive | 3801 px | 98.9% | 98.8% |

- The test passes. In this scene the model already keeps to the clicked ellipse without the
  negative click, so the first row alone does not show that the negative click does anything. The
  third row does: with the same two points, the label decides whether the neighbor is in the mask.
- One preview took 0.29 to 0.30 s.

Not asserted, recorded by the same test: the positive click moved from the center of its ellipse
toward the point where the two touch (30 px from the center), the negative click unchanged at
the neighbor's center.

| positive click, part of the way to the contact point | with the negative click: neighbor / clicked ellipse / mask | positive click alone: neighbor / clicked ellipse / mask |
|---|---|---|
| 0.50 | 0.1% / 98.6% / 1896 px | 0.0% / 99.2% / 1901 px |
| 0.80 | 4.1% / 1.3% / 127 px | 0.0% / 98.7% / 1891 px |
| 0.95 | 22.8% / 0.1% / 448 px | 0.0% / 98.9% / 1937 px |
| 1.00 (on the contact point) | 29.9% / 0.0% / 571 px | 99.7% / 100.0% / 4019 px |

- With the positive click 15 px from the contact point (0.50) the result is as at the center.
  At 6 px (0.80) and closer, together with the negative click on the neighbor, the mask shrinks to
  a small patch: neither ellipse. With the positive click alone at the same places the mask is the
  clicked ellipse. A positive click exactly on the contact point selects both ellipses, and the
  negative click does not repair that.
- For students this means: click well inside the animal, and judge a negative click by the
  preview. This is the model's behavior with two points (it then gives one mask instead of
  choosing among three), not a tolerance of the test; nothing was tuned.

### 2.2 Preview equals the first frame of a run

Three ellipses (semi-axes 8 and 3 px, at least 300 px apart) with 1, 3 and 2 points, among them
two negative points on the background 40 px beside an ellipse. On `cpu`, one loaded model:

- `preview(image, prompts)` equals `start(image, prompts)` in every field of every object:
  offset, mask, logits and score, bit for bit;
- a run of three frames with two previews in between (another frame and other prompts; then the
  first frame again) equals the plain run on every frame, bit for bit; the second of those
  previews again equals the first frame;
- the previewed centers are 0.21, 0.37 and 0.47 px from the true centers;
- one preview of three objects took 0.42 to 0.43 s (SPEC 5 asks for under 1.5 s).

### 2.3 Fall back in `step()` with the real model on the Apple GPU

A plain mps run does not fail by itself (section 1.2), so the test makes the model call raise
`NotImplementedError` once, on the third frame, while the device is mps. Three ellipses moving
0.6 px per frame, one positive click each, 12 frames.

| | result |
|---|---|
| device after each frame | mps, mps, then cpu for frames 2 to 11 |
| frame numbers given to the model | 0, 1, 2 (raised) on mps; 0, 1, …, 9 in the new session on cpu |
| prompts of the new session | one positive point per object, at its center on frame 1 (the last frame on which it was found) |
| max error against the true centers | 0.885 px over all frames (limit 3 px): 0.441 px on mps, 0.675 px on the fallback frame, 0.885 px after it |
| lost | 0 of 36 |
| log | one line: `The Apple GPU failed (NotImplementedError: …); using the processor instead. Tracking starts again on this frame, from the last position of A, B, C.` |
| time | 0.77 to 0.84 s per frame on cpu after the fall back (3 objects, three runs), 0.74 to 0.77 s for the fallback frame |

Timing: another test job was running on the same laptop, so the values may be a little high.

### 2.4 Checks that need no weights

With the real processor and session and a stand-in for the network (seconds):
- objects with 1, 3 and 2 points are stored as tensors of shape (1, 1, P, 2) with labels ≥ 0 and
  the given labels: no padding (−10), since every object has a call of its own;
- points outside the image, among them (−10, −10), never reach the session; an object whose only
  positive point is outside is refused before the model is called;
- a preview goes through a session of its own at frame 0; the run's session, its objects and its
  frame count are untouched, and the next frame of the run is the next number;
- with the stand-in failing on its third call on mps: the processor and the first session really
  run on mps, the new session is on cpu, its frames count from 0, and it holds one positive point
  per object at the center where that object was last found (for an object not found on the
  second frame: its center on the first);
- `reserve_ui_thread()` in a process of its own: torch has one thread fewer after the first call,
  and a second call changes nothing.

The same logic without torch (fast tests, `tests/test_hf_helpers.py`): an object that was never
found gets no click and stays lost; a `RuntimeError` on cpu, or an error of another kind on mps,
is raised and not hidden; every model call and every move of the model happens with the one model
lock held.

### 2.5 Can these tests fail?

- Against the backend as it was before this task, the three real-model tests failed: two for the
  missing `preview`, the third with the injected `NotImplementedError`, which `step()` did not
  catch. Four of the five tests without weights failed too.
- The 5% limit is far from both outcomes seen: 0.0% with the negative label and 98.9% with a
  positive label on the same point, so a label sent wrongly would fail the test.

### 2.6 Not covered by this section

- A real failure of the Apple GPU. None was seen; the failure is injected at the model call.
- A fall back in the middle of a longer run with objects close to each other: the model loses its
  memory of the earlier frames there, and each object is found again from one click.
- SAM 2.1; fine mode and the dish crop (they belong to the tracking tasks).

## 3. Exact frame access (task A09, 2026-10-07)

Measured on the build Mac with OpenCV 5.0.0, on two small test clips (320 x 240 px, 120 frames,
H.264 with B-frames): one evenly timed, one with three gaps in its timestamps (before frames 30,
72 and 101), as a phone video with dropped frames has.

| how a frame is fetched | evenly timed clip | clip with timestamp gaps |
|---|---|---|
| plain OpenCV seeking (`CAP_PROP_POS_FRAMES`) | 0 of 27 tested frames wrong | 11 of 27 wrong (a neighbor, one frame early or late) |
| `FrameSource` (seek, identify the frame by its timestamp, step forward) | 0 wrong | 0 wrong |
| `FrameSource` with every seek shifted on purpose by -2, -1, +1, +2 or +6 frames | 0 wrong | 0 wrong |

- The first row is OpenCV's behavior and is printed by the test, not asserted; it was measured
  only on the Mac.
- Tracking never seeks: it decodes from the first frame, as last week. `FrameSource` is for the
  display and for jumps. Before a run starts, the hash of each prompt frame is compared with the
  frame the tracking decoder delivers.
- Not covered: real phone files. `outline-tracker check VIDEO --seek` compares 20 random frames
  on a real file and prints the number of timestamp gaps.

## 4. The real model through the whole pipeline (task B5, 2026-10-07)

Sections 1 and 2 tested the backend alone. Here the real EdgeTAM runs through everything a user
runs: a session, `tracking.run_job`, `export.export_all`. Same laptop, library versions and weights
as section 1; `cpu` with 8 torch threads unless a row says `mps`. Only short synthetic clips were
used. Another job may have been running the model on the same laptop at the same time, so every
time given in this section is an upper bound.

| what | device | s per frame |
|---|---|---|
| coarse, 1 object, whole 1080p frame (4.1) | cpu | 0.38 |
| coarse, 3 objects, whole 1080p frame (4.1) | cpu | 0.76 |
| coarse, 3 objects, dish square (4.3) | cpu | 0.76 to 0.81 |
| coarse, 10 objects, whole 1080p frame (4.4) | cpu | 2.2 to 2.3 |
| fine, 1 object, 189 px window (4.2) | cpu | 0.38 to 0.40 |
| fine, 1 object, 189 px window (4.2) | mps | 0.11 to 0.12 |

Each value includes decoding, measuring the masks and saving.

### 4.1 Regression through the pipeline (SPEC 13.3): the files of `from-tracker` against last week's

**What was compared.** `from_tracker.from_tracker` (coarse, the whole frame, no overlay) and last
week's `shrimp.segment.track_video` with last week's `TransformersSegmenter` (the unmodified copies
in `tests/reference`). One loaded EdgeTAM served both, on `cpu`, in one process and with the same
thread count; both got the same video, Tracker export, `fps_true` = 240 and options. The files
compared are the Tracker-format files, `<run folder>/edgetam/<id>.csv`, against the files last
week's script wrote. Both sides write `pixelx` and `pixely` with three decimals.

**Command.**

```
uv run pytest -m slow tests/slow/test_regression_pipeline.py -q -rP
```

**Result.** 2 passed, twice: in 74 s and in 59 s. The numbers of the two runs are the same except the
times.

| clip | export | objects | frames | max difference in (`pixelx`, `pixely`) | lost rows (last week / new) | s per frame (last week / pipeline) |
|---|---|---|---|---|---|---|
| selftest clip (`synthetic.selftest_clip`) | one track, frames 0, 2, …, 38 | 1 | 20 | 0.0000 px over 20 rows (limit 0.01) | 0 / 0 | 0.38 to 0.40 / 0.38 |
| three-ellipse clip of section 1.1 | `#multi` start file: A, B, C marked on frame 0; `seconds` = 40/240, `step` = 2 | 3 | 20 each | 0.0000 px over 60 rows (limit 0.01) | 0 / 0 | 0.77 / 0.76 |

- File names and frame numbers are the same on both sides: `selftest.csv`; `A.csv`, `B.csv`,
  `C.csv`; frames 0, 2, …, 38.
- Reported by the test, not asserted: all four files equal last week's byte for byte, so the mm
  columns `x`, `y` and the time column `t` are the same too.
- The selftest-clip run folder holds the full set of SPEC 8.1 without the overlay (not asked for)
  and without `probes.csv` (no probes): `session.json`, `positions.csv`, `edgetam/selftest.csv`,
  `shapes.csv`, `radial.csv`, `outlines.npz`, `results.npz`, `run.log`, `README.txt`. Nothing else
  is in it, and no file is empty.
- The pipeline's s per frame includes measuring each mask and saving; it is not slower than last
  week's loop here.

**Can this test fail?** With the new backend's mask threshold moved from `logits > 0` to
`logits > 0.5` (patched in memory for one run, no file changed), both tests failed: 0.1178 px on
the selftest clip and 0.2611 px on the three-ellipse clip. These are the values section 1.4 found
at the level of the segmenter.

**Not covered.** The test asserts the pixel columns only; the bytes of whole files are asserted by
the fast tests with a stand-in model (`tests/test_from_tracker_port.py`). A real video was not run:
that is the owner's go/no-go check.

### 4.2 Fine mode on the close-up shrimp (SPEC 13.4): two of three criteria FAIL

**What was run.** `synthetic.closeup_scene()` as it stands: 1080p, 480 frames at 240 fps (2 s),
0.010 mm per px; object A is a 47 × 20 px body with two antennae 30 px long and 3 px wide that beat
at 9 Hz. A was tracked in fine mode on every frame (step 1) from **one positive click on the
center of its body**, through `tracking.run_job` (which chose the window from the preview mask) and
`export.export_all`. The true solidity is the ground-truth table's: area of the analytic outline
over the area of its convex hull (the definition of SPEC 7.7), per frame.

**Command.** One run per device, three tests on each (`-k cpu` or `-k mps` runs one device):

```
uv run pytest -m slow tests/slow/test_fine_mode.py -q -rP
```

**Result.** On each device 1 passed and 2 failed; the two are marked `xfail(strict=True)` with these
numbers. cpu: 200 s; mps: 73 s. Nothing was tuned.

| | cpu | mps | SPEC 13.4 |
|---|---|---|---|
| `shape_ok` = 1 | 480 of 480 frames | 480 of 480 frames | **passes** |
| peak of the solidity spectrum (mean removed) | 0.56 Hz | 0.56 Hz | **fails** (9 ± 0.5 Hz) |
| RMS difference from the true solidity | 0.4293 | 0.4293 | **fails** (< 0.02) |
| measured solidity | 0.990 to 0.999, mean 0.996 | the same | true: 0.507 to 0.711, mean 0.572, peak at 9.00 Hz |
| mask area | 740 to 926 px, median 814 | 743 to 922 px | body alone 738 px; true mask 863 to 888 px |
| window chosen | 189 px | 189 px | |
| `px_along_major` / `cells_along_major` | 46.2 to 49.8 / 62.6 to 67.4 | 46.2 to 49.8 / 62.6 to 67.5 | both ≥ 20 |
| lost frames; flags | 0; `HEADGUESS` on all (no head click) | the same | |
| s per frame | 0.38 to 0.40 | 0.11 to 0.12 | |
| device at the end | cpu | mps (no fall back) | |

**What the model does.** From one click on the body it outlines the body and leaves the antennae
out, on every frame. Pictures of frames 0, 20 and 60 (saved outside the repository): the model's
outline is the body's ellipse, a convex shape that lies on the true outline around the body and
cuts straight across the base of both antennae; on frame 0 it has a small bump there. The antennae
are clearly in the picture (3 px wide, as dark as the body), and the window gives them 4 grid
cells of width, so this is the model's choice of object, not a lack of resolution. The pipeline
measured that mask correctly: its solidity is that of an ellipse, and it does not beat. The RMS
difference from the solidity of the true pixel mask as scikit-image defines it is 0.4384, so the
choice of the truth does not matter here.

**One more measurement, not a test and not asserted.** The same run with two more positive clicks
on frame 0, one on the middle of each antenna (clicks at (1393.7, 465.9), (1375.3, 446.4) and
(1395.2, 439.2) px), once, 480 frames on mps:

| | three positive clicks |
|---|---|
| peak of the solidity spectrum | 9.00 Hz (would pass) |
| RMS difference from the true solidity | 0.0526 (would fail the limit of 0.02) |
| of that, a constant offset | +0.0522 (measured mean 0.625, true mean 0.572) |
| RMS after removing the offset; correlation with the truth | 0.0068; 0.995 |
| measured solidity | 0.542 to 0.776 |
| mean mask area | 1004 px (true mask 863 to 888 px) |
| `shape_ok`; lost frames; window | 1 on all 480; 0; 190 px |

In the pictures of this run the outline follows the body and both antennae on frames 0, 20 and 60,
a little outside the true outline along the antennae: the mask draws them about 1 px wider on each
side (it has 1004 px on average), which is the offset. The first 40 frames on cpu gave similar
numbers (RMS 0.044, window 191 px).

**For J.**
- With the real model, fine mode measures antennae only if the clicks say that the antennae belong
  to the object. Students who want the stroke need a positive click on each antenna and must judge
  the preview; one click on the body gives a clean body outline (good for position and heading,
  useless for solidity). This belongs in the how-to.
- The absolute solidity is then about 0.05 too high, while its variation is right (9.00 Hz,
  correlation 0.995). The limit of 0.02 on the absolute value is not met by this one try either.
  Whether the criterion should be the variation, or whether the clicks should be different, is J's
  decision; the tests still ask what SPEC 13.4 asks.
- Not tried: other click positions, a negative click, SAM 2.1, a real close-up clip.

### 4.3 Coarse mode at dish scale flags `LOWRES` (SPEC 13.4): in 0.1.0 B failed on one frame; with the size check of the largest piece all three pass

The record of version 0.1.0 comes first, as it was written. What changed after it, and the run
with the changed rule, are under it.

**What was run.** `synthetic.dish_scene()` as it stands (1080p, 0.0324 mm per px, bodies 14.5 px
long), its three objects tracked coarse from one click each on frames 0, 2, …, 58 with the scene's
dish circle, so the model saw the dish square, 1002 × 1002 px (a grid cell of 3.9 px, a body of
3.7 cells). `cpu`. Asserted per object: found at its click, and every frame with a mask has
`shape_ok` = 0 and `LOWRES`.

**Command.**

```
uv run pytest -m slow tests/slow/test_coarse_lowres.py -q -rP
```

**Result.** 2 passed, 1 failed (marked `xfail(strict=True)` with these numbers) in 36 s, the same
numbers on both runs.

| object | frames with a mask | `px_along_major` | `cells_along_major` | `LOWRES` | max error against the true centers |
|---|---|---|---|---|---|
| A (shrimp, at the wall) | 30 of 30 | 16.5 to 24.4 px | 4.2 to 6.2 | 30 of 30: **passes** | 2.25 px |
| B (plain body) | 15 of 30 | 15.0 to 16.0 px on 14 frames; 134.8 px on frame 36 | 3.8 to 4.1; 34.4 on frame 36 | 14 of 15: **fails** | 0.71 px on the 14 frames; 5.0 px on frame 36 |
| C (plain body) | 30 of 30 | 14.9 to 16.7 px | 3.8 to 4.3 | 30 of 30: **passes** | 0.48 px |

- The model lost B on frames 6 to 34 (flag `LOST`), although nothing is near it: B and C are more
  than 200 px apart on these frames.
- On frame 36 B came back with a mask of 99 px in two pieces: the body, and 2 px far from it
  (about 240 px, from the second moments). `px_along_major` comes from the second moments of the
  whole mask (SPEC 7.8), so two stray pixels make it 134.8 px, `shape_ok` becomes 1 and the frame
  is not `LOWRES`. It is flagged `ORIENT`; `MULTI` needs a second piece of at least 10%. From
  frame 38 on the mask is the body again and `LOWRES`.
- So the flag works wherever the mask is the object, and the rule was applied as the spec states
  it. **For J:** should `px_along_major` and `cells_along_major` be taken from the largest piece of
  the mask, so that a few stray pixels cannot switch `shape_ok` on? That is a change of SPEC 7.8,
  not made here.
  **Answer (decision 26 of `docs/ROADMAP.md`, decided 2026-10-07): yes.** It is made after 0.1.0;
  see below.
- 0.76 to 0.81 s per frame for three objects on the dish square.

**What changed after 0.1.0 (decision 26 of `docs/ROADMAP.md`, decided 2026-10-07).** The size
check is of the largest piece of the mask. `px_along_major`, `cells_along_major` and `shape_ok`,
and with them the flag `LOWRES`, are taken from the largest connected piece (pixels that touch at
a side or a corner belong together): L1 = 4 sqrt(lambda1) of the covariance of that piece's pixel
centers, the formula of SPEC 7.8 on fewer pixels. It is the piece the outline goes around.

- The three numbers are derived at export from the mask that `results.npz` stores, and only on
  frames whose mask has more than one piece. A frame with one piece has the numbers it had. No
  saved format changed, so a run folder of 0.1.0 shows the new numbers at its next export.
- Not changed: the position and the area are of the whole mask, and so are `major_mm`,
  `minor_mm` and `eccentricity`. The core and the heading were stored at tracking time. `SIZE`,
  `MULTI` and the limit of 20 are as they were.
- The test asks what it asked. Its `xfail` mark on B and the reason text are gone; no assertion
  changed.

**Run with the changed rule (2026-10-07).** The same command, once, on `cpu`, on an
Apple-silicon laptop with macOS 27; the library versions and the SHA-256 of the weights were
checked and are those of section 1. Nothing was tuned.

**Result.** 3 passed in 35 s.

| object | frames with a mask | `px_along_major` | `cells_along_major` | `LOWRES` |
|---|---|---|---|---|
| A (shrimp, at the wall) | 30 of 30 | 16.5 to 24.4 px | 4.2 to 6.2 | 30 of 30: **passes** |
| B (plain body) | 15 of 30 | 15.0 to 16.0 px | 3.8 to 4.1 | 15 of 15: **passes** |
| C (plain body) | 30 of 30 | 14.9 to 16.7 px | 3.8 to 4.3 | 30 of 30: **passes** |

- The table holds the numbers the run printed (its `VALIDATION` line). The next three points
  were read from the `shapes.csv` and `positions.csv` that the same run wrote; the true centers
  are those of the scene's ground-truth table.
- The model did what it did before: B is lost on frames 6 to 34, and on frame 36 its mask has
  99 px in two pieces, 97 px and 2 px. Frame 36 now reads `px_along_major` 15.0 px and
  `cells_along_major` 3.8, `shape_ok` is 0 and its flags are `LOWRES;ORIENT;HEADGUESS`. Its
  `major_mm` is still of the whole mask: 4.37 mm, which is 134.8 px.
- Left as it is, on purpose: on frame 36 the position is the centroid of all 99 px, so it is
  still 5.0 px from the true center, and the frame keeps `ORIENT`, because the core's opening
  radius was taken from the whole mask at tracking time. Decision 26 is about the size check
  only. The other errors against the true centers are those of the table above too: A 2.25 px,
  B 0.71 px on its other 14 frames, C 0.48 px.
- A's mask has two pieces on one frame (28). There the largest piece is 0.3 px shorter than the
  whole mask; A's range did not move. C's masks have one piece on every frame.
- 0.78 s per frame for three objects on the dish square.

### 4.4 Memory (SPEC 6.5): 10 coarse objects at 1080p

**What was run.** A 1080p clip of 100 frames with ten dark ellipses (semi-axes 8 and 3 px, more
than 250 px apart), all ten tracked in one coarse run on the whole frame, every frame, on `cpu`,
through `tracking.run_job`. The run is in a child process of its own, which reads its resident
memory with `ps` after every tracked frame and its peak with `resource.getrusage` at the end.
GB = 10⁹ bytes, MB = 10⁶ bytes. macOS and Linux only; skipped on Windows.

**What is asserted.** The peak is under 3 GB, and memory grows by less than 50 MB from frame 40 to
frame 100. Since the review of this task, the memory at a frame is the median of the ten readings
that end with the one after that frame (frames 31 to 40, and frames 91 to 100). Before, it was the
single reading after the frame; why that was changed is below. A second test in the file checks
this estimator on series whose answer is known by arithmetic; it needs no model.

**Command.**

```
uv run pytest -m slow tests/slow/test_memory.py -q -rP
```

**Result.** Four runs, all passed. Runs 1 and 2 asserted the two single readings (1 passed, 236 s
and 232 s); another job may have been running the model at the same time. Runs 3 and 4 assert the
two medians (2 passed, 245 s and 238 s); no other job ran the model during them (other checkouts
ran their fast test suites). The reading of every frame was kept for runs 3 and 4 only.
Differences are taken before rounding.

| | run 1 | run 2 | run 3 | run 4 | limit |
|---|---|---|---|---|---|
| peak resident memory | 1.43 GB | 1.44 GB | 1.43 GB | 1.42 GB | 3 GB |
| median of the ten readings after frames 31 to 40 | not kept | not kept | 1400 MB | 1393 MB | |
| median of the ten readings after frames 91 to 100 | not kept | not kept | 1418 MB | 1394 MB | |
| growth between the two medians | | | +18 MB | +1 MB | 50 MB (asserted in runs 3 and 4) |
| single reading after frame 40 | 1388 MB | 1403 MB | 1274 MB | 1393 MB | |
| single reading after frame 100 | 1398 MB | 1392 MB | 1418 MB | 1401 MB | |
| growth between the two single readings | +10 MB | −10 MB | +144 MB | +9 MB | 50 MB (asserted in runs 1 and 2 only) |
| growth of the peak from frame 40 to frame 100 | +32 MB | 0 MB | +8 MB | +22 MB | not asserted |
| single readings after frames 1, 20, 60, 80 | 1181, 1378, 1402, 1290 MB | 1181, 1318, 1405, 1400 MB | 1177, 1407, 1404, 1426 MB | 1180, 1234, 1406, 1404 MB | |
| all readings from frame 40 on | up to 1430 MB | 1260 to 1407 MB | 1253 to 1428 MB | 1179 to 1418 MB | |
| low readings (more than 50 MB under those around them), frames 2 to 100 | | | 20 | 18 | |
| most low readings among ten in a row, from frame 21 on | | | 3 | 3 | |
| s per frame (10 objects) | 2.26 | 2.21 | 2.33 | 2.27 | |
| object-frames without a mask | 112 of 1000 | 112 of 1000 | 112 of 1000 | 112 of 1000 | not asserted |

- Last week 1.7 GB was measured for 10 objects; every run here stays below that.
- **Why medians.** A reading of resident memory after one frame is not a steady number. In runs 3
  and 4 about every fifth reading (20 and 18 of 99) is 57 to 214 MB below the readings around it,
  for one frame; the next reading is back where it was. These low readings come about every 10 s
  (every 4 or 5 frames at 2.3 s per frame). They came with no other job running the model, so
  they belong to this process on this laptop; their cause was not looked for. In run 3 one of
  them fell on frame 40: the two single readings differ by +144 MB, so the test as first written
  would have failed there with no leak, and a low reading on frame 100 would have hidden a leak
  of that size. Of the ten readings that end at frame 40 and of the ten that end at frame 100,
  two or three were low; a median of ten is not moved by up to four. A steady leak shows in the
  medians in full: the middles of the two groups are 60 frames apart, so a leak of 50 / 60 =
  0.83 MB per frame or more fails the test.
- **Can this test fail?** With the backend's pruning switched off (patched in memory for one run
  of the same child, no file changed): peak 3.06 GB, and resident memory 1181, 1638, 1980, 2343,
  2701, 3063 MB after frames 1, 20, 40, 60, 80, 100: +1083 MB from frame 40 to frame 100, 18 MB
  per frame. Both limits are exceeded. That run was made before the change; on its saved readings
  the two medians differ by +1076 MB. The test of the estimator was written first and failed with
  the single readings: +108 MB for a flat series with one low reading on frame 40, where 0 is
  right. It also holds a leak of 1.7 MB per frame that a low reading on frame 100 hides from the
  single readings (−6 MB); the medians give +100 MB.
- **Uncertain.** The level between the low readings moves too, in steps and without a trend: from
  frame 21 on between 1389 and 1428 MB in run 3 and between 1375 and 1418 MB in run 4. The
  medians of ten readings in a row span 37 MB in run 3 and 20 MB in run 4, with no leak, and the
  limit is 50 MB. So the test can still fail without a leak, if such a step exceeds 50 MB, or if
  five of ten readings in a row are low (a median then moves by about half the depth of a low
  reading; if they come every 10 s on other computers too, that needs about 5 s per frame).
  Neither was seen in these two runs. With 100 frames the limit of 3 GB catches the run without
  pruning only just (3.06 GB); the growth is what shows it clearly.
- The model lost about a tenth of the object-frames on this clip (a plain background, objects of
  16 × 6 px). That is tracking quality, not memory, and is not asserted here.

## 5. The `selftest` command with the real model (task B2, 2026-10-07)

**What was tested.** `outline_tracker.selftest.selftest`, the port of last week's selftest, called
as the command `outline-tracker selftest` calls it: it makes last week's clip (one dark ellipse,
semi-axes 8 and 3 px, on a 1080p frame; 20 tracked frames at step 2), loads EdgeTAM itself, tracks
the clip through `from_tracker` (coarse, whole frame, no overlay, fps_true 240) and compares the
positions in the run's Tracker-format file with the true centers. Same laptop, versions and weights
as section 1; the model was already on the laptop, so nothing was downloaded.

**Command.**

```
uv run pytest -m slow tests/slow/test_selftest_real.py -q -rP
```

**Result.** 2 passed in 30 s.

| device | verdict | max error (limit 3 px) | mean error | s per frame | estimate for 1,200 frames: one shrimp / 10 shrimp | device at the end |
|---|---|---|---|---|---|---|
| cpu | OK | 0.461 px | 0.265 px | 0.38 | 7.7 min / 42.2 min | cpu |
| mps | OK | 0.461 px | 0.272 px | 0.14 | 2.8 min / 15.7 min | mps (no fall back to the processor) |

- The max error is worked out twice, by `selftest` and by the test from the Tracker-format file and
  the true centers (700.5 + 0.6 f, 500.5 + 0.2 f px in frame f): the two agree. On cpu it is the
  0.461 px that last week's own selftest got with the same model (section 1.2).
- The test shrimp was found on all 20 frames on both devices.
- s per frame is what `from_tracker` reports for the whole job: decoding, the model, measuring the
  mask and saving the results, without loading the model. Section 1.2 measured 0.37 s on cpu around
  the segmenter alone. Another job may have been running the real model on the same laptop at the
  same time, so these values are upper bounds.
- The estimate is last week's arithmetic: s per frame × 1,200 frames, and for 10 shrimp together
  that × (1 + 9 × 0.5) for EdgeTAM.

**The command itself**, run once on each device (exit code 0 both times):

```
uv run outline-tracker selftest --device cpu
uv run outline-tracker selftest
```

| command | ran on | wall time | last two lines |
|---|---|---|---|
| `selftest --device cpu` | cpu | 18 s | `OK: edgetam followed the test shrimp within 0.5 pixels (should be under 3). 0.38 s per frame here.` / `Estimate for 10 s at step 2 (1,200 frames): one shrimp about 8 min; 10 shrimp together about 42 min.` |
| `selftest` (device `auto`) | mps | 14 s | `OK: edgetam followed the test shrimp within 0.5 pixels (should be under 3). 0.14 s per frame here.` / `Estimate for 10 s at step 2 (1,200 frames): one shrimp about 3 min; 10 shrimp together about 15 min.` |

### 5.1 Can the selftest say PROBLEM?

Checked with stand-ins in the fast tests (`tests/test_selftest.py`, no model): a stand-in whose
masks lie 5 px beside the ellipse gives `PROBLEM`, `ok` false and exit code 1; one that loses the
ellipse after 10 frames gives `PROBLEM` and `ok` false too. A run stopped with Ctrl+C, and a model
that cannot be loaded, give one `ERROR:` line and exit code 1, without a verdict.

### 5.2 Not covered by this section

- SAM 2.1 (`--model sam2`): the weights were not downloaded and the model was not run; the option
  is tested with a stand-in.
- `--device cuda` (no NVIDIA GPU here), and Windows.
- A first run that downloads the model, and a run without internet, with the real loader.
- A shrimp that the real model loses on every frame: the verdict is then `PROBLEM`, with `nan` in
  place of the pixels and a numpy warning next to it, as last week (seen once with a stand-in, by
  hand; no test holds it).

## 6. The real model through the window (tasks C4 and C5, 2026-10-07)

Sections 1 to 5 ran the model from tests and from the command line. Here it runs where a student
runs it: inside the window, on the window's worker thread, with the frames coming from the
window's frame cache. Same laptop, versions and weights as section 1. Only short synthetic clips.
The window ran offscreen (no screen was looked at; pictures of the window were grabbed and looked
at afterwards).

### 6.1 A tracking run through the window's worker (the slow test of C5)

**Command.** One per device:

```
uv run pytest -m slow tests/slow/test_gui_worker_real_model.py -q -rP -k cpu
uv run pytest -m slow tests/slow/test_gui_worker_real_model.py -q -rP -k mps
```

**Result.** 2 passed (26 s and 14 s). The clip is last week's selftest clip, 20 tracked frames, one
object, one click, tracked by the Track button's own job.

| device | status | max error against the true centers (limit 3 px) | mean error | s per frame | device at the end |
|---|---|---|---|---|---|
| cpu | complete | 0.460 px | 0.265 px | 0.42 | cpu |
| mps | complete | 0.460 px | 0.272 px | 0.12 | mps (no fall back) |

The errors are those of sections 1.2 and 5 (0.461 px): the window adds nothing to them. No error
was recorded by the worker in either run.

### 6.2 The whole path by hand, once (not a test)

A script drove the window the way a user does, with the real EdgeTAM on the Apple GPU, on the
synthetic close-up clip (`outline-tracker synth closeup`, 1920 × 1080 px, 240 fps): open the clip,
type a name and fps_true into the fields of panels 1 and 2, Add in panel 6, one positive click on
the body of the shrimp on frame 0, Track in panel 7 with the clip ending at frame 119 (60 tracked
frames at step 2).

| what | measured |
|---|---|
| model ready after the window opened | 7 to 8 s (the model was on the laptop already) |
| outline after the click (the preview) | 0.69 s |
| the run: 60 frames, 1 object, coarse, whole frame | about 9 s; the panel said "Tracking is complete: 1 object, 60 frames." |
| files in the run folder afterwards | `session.json`, `results.npz`, `run.log` |
| stored position of the object | (1391.0, 461.2) px on frame 0, (1315.3, 363.8) px on frame 60, (1201.2, 324.4) px on frame 118; found on every frame |

In the picture of frame 60 the stored outline lies on the animal and includes both antennae. On
frame 0 the preview from the one click on the body included one antenna and left the other out;
section 4.2 found the body alone from one click in fine mode. So what one click includes varies:
the user has to look at the outline, and click on an antenna that is missing.

**Found on the way, and fixed.** The first run gave a warning from torch: the frame cache hands
out arrays that cannot be written to. The worker now gives the model a copy it may write to
(task C5), and a test holds that. Click points of another frame stayed on the picture when a part
other than the bottom bar changed the frame; they now follow the frame on the screen.

### 6.3 The whole flow with three small objects, and what fine mode does for a lost one (not a test)

The same kind of script, on the synthetic dish clip (`outline-tracker synth dish`, 1920 × 1080 px,
240 fps, 32.4 µm per px; three bodies about 14.5 px long), with the real EdgeTAM on the Apple GPU:
name and fps_true typed into panels 1 and 2, a stick and a dish circle put into the session, three
objects added with one positive click each on frame 0 at their true centers, Track with the clip
ending at frame 119 (60 tracked frames at step 2, the model seeing the dish square), then the flags
table of panel 8 and Export all in panel 9. All panels were on the merged main (commit `ae870e2`).

| | all three coarse | B fine, A and C coarse |
|---|---|---|
| the run | complete, 14.5 s | complete, 18.3 s |
| A (shrimp at the wall): found; max error against the true centers | 60 of 60; 2.54 px | 60 of 60; 2.54 px |
| B (plain body): found; max error | **45 of 60** (`LOST` on 15 frames); 5.02 px | **60 of 60**; 0.38 px |
| C (plain body): found; max error | 60 of 60; 0.56 px | 60 of 60; 0.56 px |
| rows in the flags table of panel 8 (position flags) | 15, all `LOST` of B | 0 |
| Export all | 10 files written, the panel listed them | the same |

- B is the object the model also lost in section 4.3. In fine mode the model sees a small square
  that follows the object, so a 14 px body covers many more of the model's cells: B was then found
  on every frame, within 0.4 px. **Advice that follows: an object that coarse tracking loses
  (`LOST` in panel 8) is worth switching to fine in panel 6 and tracking again.** Fine objects are
  tracked one at a time, so the run takes longer (here 18 s against 15 s).
- Every object keeps `LOWRES` and `HEADGUESS` on every frame in both columns: a 14.5 px body is
  under the 20 px that shapes need, in any mode, and no head click was given.
- Pictures of panels 8 and 9 after the run were looked at: the table, the three correction
  buttons, the list of written files and Open folder are there. The label of the Fill switch in
  the bar above the picture did not show (dark text on the dark bar); that is being fixed.

### 6.4 Not covered by this section

- A real screen on Windows: nobody has yet used the window by hand there. On a Mac the owner did, on
  2026-10-07: installed commit `ae870e2` the way a user does (`uv tool install` from the repository),
  ran the selftest on the Apple GPU and on the processor, made the two synthetic clips, and went
  through every panel in the window by hand; the report was that all of it works. That is one
  person's check on one laptop, and it was made before the last follow-up (the lock during an
  export, the model box after results, a thread of its own that makes the model) was merged.
- A real video of shrimp, and a long one (opening time, memory over minutes).
- The three corrections of panel 8 with the real model (they are tested with stand-in models),
  and the real model's antennae outline in fine mode in the window.
- Windows with the real model: the Windows test machine runs only the fast tests, with stand-in
  models.
