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
