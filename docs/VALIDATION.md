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
