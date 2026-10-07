# Roadmap inventory: several tracking methods (backends)

> Inventory for `docs/ROADMAP.md`: several tracking methods. Read-only findings at commit `f73d29f` (2026-10-07).
> File and line pointers are true for that commit. "The ledger", "the build ledger" and "the
> scratch folder" are private working notes of the first build; they are not in the repository.
> The design note of the window is `docs/design/gui_design.md`.

Read on 2026-10-07 at main `f73d29f`. Read only: nothing in the repository was changed, no model was loaded.
Paths are relative to the repository root. "Not confirmed" marks what no source stated; "not verified" what I did not check.
Web sources are given by address. Sizes in bytes come from the HTTP headers of the download addresses (nothing was downloaded).

## A. The code as it is

### A1. The contract
- A backend is any object with `start(image, prompts)`, `step(image)`, `preview(image, prompts)`, `close()` (`outline_tracker/segmenter/base.py:43-50`, the `Segmenter` protocol).
- Every call returns one `MaskResult` per object, in the order of the prompts: `obj_id`, `offset`, `mask` (bool), `logits` (or None), `score` (or None) (`segmenter/base.py:34-40`).
- A prompt is points with labels 1 or 0. `ObjectPrompt.box_px` exists but nothing uses it (`segmenter/base.py:31`; `segmenter/hf.py:135-145` sends points and labels only). A box tool in the window: not verified.
- Tracking also reads optional attributes by `getattr`: `log` (`tracking.py:191-193`), `set_view` (`tracking.py:287`), `device`, `model_id`, `weights_sha256` (`tracking.py:377-380`). These are informal capabilities already.
- The real backend is one class, `HFSegmenter`: transformers `Sam2VideoModel` or `EdgeTamVideoModel` in a streaming session, one frame per call (`segmenter/hf.py:262-288`).
- It prunes the session's state by hand (`KEEP_FRAMES = 20`, `segmenter/hf.py:58, 400-419`) and sets `session.obj_with_new_inputs` (`hf.py:164`). Both reach into the internals of transformers 5.18.0, which is pinned (`pyproject.toml:26`). A newer transformers can break this silently.
- Device: `auto` means cuda, else mps, else cpu (`hf.py:238-246`). The Apple GPU falls back to cpu in `start`, `step` and `preview` (`hf.py:333-344, 356-369, 381-394`). The fall back in `step` starts again from the centroid of the last mask (`hf.py:313-315, 361-362`).

### A2. The list of models lives in six places
- `MODELS` with three keys: `sam2` (tiny), `sam2-small`, `edgetam` (`segmenter/hf.py:53-57`); the loader branches on `startswith("sam2")` (`hf.py:99-106`).
- The command line: `choices=["edgetam", "sam2"]`, twice (`cli_from_tracker.py:67`, `cli_selftest.py:42`). `sam2-small` cannot be chosen there.
- The window: `MODEL_NAMES` with two entries (`gui/panels/track_panel.py:68`); devices from `devices_for` (`track_panel.py:97-103`).
- `selftest.EXTRA_PER_SHRIMP` with three keys (`selftest.py:28`); the default `Processing.model = "edgetam"` (`session_parts.py:188`).
- The factory `from_tracker.load_segmenter(model, device)` (`from_tracker.py:92-101`), given to the window in `gui/app.py:39-45`. The worker compares the factory by identity to decide on the thread limit (`gui/worker.py`, `before_load_for`).
- docs/DEVELOPER.md, "Adding a model", names three of these places by hand. There is no registry.
- The model's key also names a results folder (`export.py:165`) and a line of run.log (`export_log.py:72-76`). Each run stores `device`, `model_id`, `weights_sha256` (`session_parts.py:238-254`).

### A3. How a job uses a backend
- `tracking.run_job` makes the segmenter once per job (`tracking.py:190`) and runs the plans one after another (`tracking.py:200-210`).
- Coarse: all pending coarse tracks with the same start frame and the same last frame share one run, so one session of the model (`tracking_plan.py:167-177`). Fine: one run per object.
- A run decodes its frames in order, cuts each to what the model sees, calls `start` on the first and `step` on the others, and needs one result per track at once (`tracking.py:305-317`, `zip(..., strict=True)`). Then it measures and forgets the frame.
- So the loop is forward only, one frame in, one answer out. A method that answers late (a window of frames) or needs the whole clip does not fit. CLAUDE.md also forbids loading a whole video.
- Fine mode asks `preview` for the mask on the start frame before any run (`tracking.py:197`, `tracking_fine.py:262`). The window is W = clip(ceil(3 F), 96, 512), F the largest diameter of that mask's outline (`tracking_fine.py:46-78`). The crop then follows the centroid of the last mask (`tracking_fine.py:164-183`).
- The time estimate assumes 1 + 0.5 per extra object of a run, measured for EdgeTAM (`gui/estimate.py:18, 27-32`).

### A4. What the rest of the tool assumes: a mask per object per frame
- A position is the centroid of a mask (`measure.py:46-55`, used at `measure.py:177`). An empty mask is a lost frame: every measured number is NaN (`measure.py:163-165, 213-222`). `visible` means "the mask is non-empty" (`schema.py:268`).
- A record has 23 fields and all but `frame`, `mode`, `score` and `cell_px` come from the mask (`measure.py:59-108`). The store refuses a record of another shape, and a mode other than coarse or fine (`results.py:229-244`).
- results.npz has no field for the method that made a record. Its format version is 1, and a file of another version is refused (`schema.py:28`, `results.py:204-205`).
- `cell_px` is max(width, height) / 256: the 256 x 256 mask grid of the SAM family is a constant of the measuring code (`measure.py:43, 160`; `hf.py:250`).

### A5. What breaks or goes empty when a method returns only a point
A point returned as an empty mask is a lost frame, so even the position is lost. A point returned as a one-pixel mask loses the sub-pixel position (the centroid is a pixel center). Neither works: a point needs its own path. Per consumer:

| consumer | what it needs | with a point only |
|---|---|---|
| positions.csv | centroid of the mask | NaN and `LOST` on every row, unless the record learns to hold a point (`qc.py:87`) |
| shapes.csv | area, second moments, core, the 256-point outline (`derive.py:144-173`) | every shape column NaN; `shape_ok` 0 |
| radial.csv, outlines.npz | the outline; rows only for tracks with a fine record (`export.py:117-118`) | header only, or NaN rows (`export_tables.py:156`) |
| heading | covariance of the core mask (`derive.py:154`) | undefined; `HEADGUESS` |
| flags | `LOST`, `SIZE`, `CONTACT`, `EDGE`, `MULTI`, `LOWRES`, `ORIENT` all read the mask (`qc.py:86-96`) | only `JUMP` works from positions (`qc.py:145-153`) |
| overlay.mp4 | outline and centroid; a non-finite outline is skipped, the dot is still drawn (`overlay.py:98-104`) | works once the position is finite |
| the window's overlay | draws a track only if `visible`; the label sits on the outline (`gui/overlays.py:200`, `_draw`) | nothing is drawn |
| the outline after a click | a result without a mask counts as "not found" (`gui/worker.py`, `found_in`) | every click says the model found nothing |
| fine mode | window from the mask's size, first center from the mask, crop follows the mask (`tracking_fine.py:68-78, 263-268, 177-182`) | W is always 96 px and the crop never moves |
| `from-tracker` CHECK lines | the mask's area (`from_tracker.py:62-67`) | the size check is meaningless |
| `LOWRES`, `CONTACT` limit | `cell_px` (`derive.py:149-150`, `qc_contact.py:37, 90`) | no meaning without a mask grid |

### A6. Facts about the present state
- SAM 2.1 is offered but was never run here: "the weights were not downloaded and the model was not run" (docs/VALIDATION.md:81 and :528). The task text says it "already runs"; the documents do not support that. The sizes base plus and large are not listed.
- Every measured number in docs/VALIDATION.md is EdgeTAM: 0.38 s per frame on cpu and 0.11 to 0.12 on mps for one object; 2.2 to 2.3 s per frame for ten objects; peak memory 1.43 GB (VALIDATION sections 4 and 4.4).
- Small objects are the measured weak point. Coarse tracking lost a 14.5 px body on 15 of 30 frames, and 112 of 1,000 object-frames in a ten-body clip (VALIDATION 4.3, 4.4). Fine mode found the same body on 60 of 60 frames (VALIDATION 6.3).
- On main the model box is off only during a run, a load, or without a video (`track_panel.py:368`), so one session can mix models. A lock "once the session has results" is being built on another branch (not on main; not verified).
- torch, torchvision, transformers, timm, huggingface_hub and safetensors are base dependencies; there are no extras (`pyproject.toml:24-29`). In this checkout torch is 557 MB of a 1.4 GB environment (`du` of `.venv`, macOS arm64).
- The weights cache is `~/.cache/shrimp-models` and `$SHRIMP_MODEL_CACHE` (`segmenter/edgetam_convert.py:82-85, 161`): a name of the class.
- EdgeTAM's checkpoint is read with `torch.load(..., weights_only=False)` (`edgetam_convert.py:168`), which runs whatever the pickle holds. The `.pt` and `.pth` files of TAPIR and CoTracker3 are pickles too. A public tool should load safetensors or use `weights_only=True`.
- The repository has no LICENSE file yet; SPEC.md:842 suggests Apache-2.0.

## B. The methods

### B1. SAM 2.1 (tiny, small, base plus, large)
- Tracks: masks of prompted objects; "streaming memory for real-time video processing" (https://arxiv.org/abs/2408.00714). Prompts: points with labels, boxes, masks (installed transformers 5.18.0, `processing_sam2_video.py:565-572`).
- License: "The SAM 2 model checkpoints, SAM 2 demo code (front-end and back-end), and SAM 2 training code are licensed under Apache 2.0" (https://github.com/facebookresearch/sam2). Model card tag: Apache-2.0 (https://huggingface.co/facebook/sam2.1-hiera-tiny).
- Install: already here, through transformers `Sam2VideoModel` (`hf.py:97-101`). Meta's own package is a git clone and `pip install -e .` (python >= 3.10, torch >= 2.5.1). The PyPI name `sam2` belongs to someone else's fork (https://pypi.org/pypi/sam2/json): do not depend on it.
- Size and speed (Meta's README): 38.9 / 46 / 80.8 / 224.4 M parameters; 91.2 / 84.8 / 64.1 / 39.5 FPS, "measured on an A100 with torch 2.5.1, cuda 12.4". `model.safetensors`: 155,908,064 / 184,305,280 / 323,476,296 / 897,897,416 bytes (https://huggingface.co/facebook/sam2.1-hiera-{tiny,small,base-plus,large}).
- Devices: cuda per the README. mps and cpu: not confirmed by a source, and not measured here (A6).
- Memory: online; the model card documents frame-by-frame streaming without loading the video. The session grows unless pruned (18 MB per frame measured for EdgeTAM without pruning, VALIDATION 4.4). For SAM 2.1: not measured.
- Limits: the same 256 grid, so the same question about objects of a few px; for SAM 2.1 not measured. Adding base plus and large is one line each in `MODELS`, plus the five other lists of A2.
- Seen in passing: transformers 5.18.0 also holds SAM 3 classes (`transformers/models/sam3_tracker_video`). The weights are gated, license "Other", 0.9 B parameters (https://huggingface.co/facebook/sam3). Not a candidate for a default.

### B2. EdgeTAM (in the tool)
- Tracks masks, same prompts. License: "The EdgeTAM model checkpoints and code are licensed under Apache 2.0" (https://github.com/facebookresearch/EdgeTAM); model card tag Apache-2.0 (https://huggingface.co/facebook/EdgeTAM).
- Install here: Meta's `edgetam.pt` (56,116,523 bytes) is converted on first use (`edgetam_convert.py:156-181`). The transformers documentation loads a converted copy, `yonigozlan/edgetam-video-1` (https://huggingface.co/docs/transformers/model_doc/edgetam_video). Its license tag: not confirmed. Using such a copy would remove the conversion code and the pickle of A6.
- Speed: "16 FPS on iPhone 15 Pro Max" (Meta's README). Here: A6. Runs on cpu and mps here; cuda not measured here.

### B3. TAPIR and its successors (BootsTAPIR, TAPNext, TAPNext++)
- Tracks: points. "tracks any queried point on any physical surface throughout a video sequence" (https://arxiv.org/abs/2306.08637). A query is (t, y, x). It also says whether the point is visible (the exact outputs: not confirmed).
- License: code Apache 2.0; "All pre-trained model checkpoints ... are also licensed under Apache 2.0" (https://github.com/google-deepmind/tapnet).
- Install: git clone and `pip install .`; no `tapnet` on PyPI (https://pypi.org/pypi/tapnet/json answers "not found"). Weights are plain files in a Google bucket, with no hash service. JAX is the first framework; some models have PyTorch files.
- Models and sizes (addresses under https://storage.googleapis.com/dm-tapnet/): TAPIR `.pt` 124,348,474 bytes; causal TAPIR, JAX only, 127,043,639; BootsTAPIR `.pt` 218,886,140; causal BootsTAPIR `.pt` 218,887,028; TAPNext, JAX only, 776,980,182; TAPNext++ `.pt` 2,532,282,370.
- Online: standard TAPIR "runs on a whole video at once"; "Online TAPIR ... allows for online tracking" (README). TAPNext "is causal, tracks in a purely online fashion" (https://arxiv.org/abs/2504.05579). TAPNext++ "tracks points in sequences that are orders of magnitude longer while preserving the low memory and compute footprint" (https://arxiv.org/abs/2604.10582).
- Speed: "~17 fps on 480x480 images on a quadro RTX 4000" for the causal demo (README). Other numbers: not confirmed.
- Devices: cuda is named. mps and cpu: not confirmed.
- Limits: trained at 256 x 256 (BootsTAPIR also 512 x 512), so a 1080p frame is resized or cropped. Accuracy in full-frame px, and behavior on objects of a few px: not confirmed.

### B4. CoTracker3
- Tracks: points, jointly; "reliably tracks visible and occluded points" (https://arxiv.org/abs/2410.11831). Input: chosen points or a grid (`grid_size`). Output: tracks and visibility per frame.
- License: "The majority of CoTracker is licensed under CC-BY-NC" (https://github.com/facebookresearch/co-tracker); the license file is "Attribution-NonCommercial 4.0 International"; model card tag cc-by-nc-4.0 (https://huggingface.co/facebook/cotracker3). Non-commercial: an Apache or MIT tool should not depend on it or ship it. At most the user installs it themself, with a clear label. The legal reading is the owner's.
- Install: `torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")` or `"cotracker3_offline"`, or a git clone; not on PyPI. `torch.hub` downloads and runs code from the repository. Weights: `scaled_online.pth` 101,695,610 bytes, `scaled_offline.pth` 101,890,938 bytes.
- Online or not: offline takes the whole clip as one tensor. Online works in overlapping chunks of frames: "more memory-efficient and allows for the processing of longer videos" (README). It answers per chunk, not per frame. Window lengths: not confirmed.
- Devices: "A GPU is strongly recommended"; cpu "for small tasks". mps for version 3: not confirmed. Speed and the largest number of points for version 3: not confirmed.

### B5. Phase correlation with an upsampled DFT (scikit-image)
- Tracks: the shift of one patch between two images. "Efficient subpixel image translation registration by cross-correlation"; registered to within 1 / `upsample_factor` of a pixel (https://scikit-image.org/docs/stable/api/skimage.registration.html). Translation only: no rotation, no scale.
- Prompt: a box around the object or feature. License: BSD (installed `scikit_image-0.26.0` METADATA). No weights. Already a base dependency (`pyproject.toml:18`). cpu only. Online: two patches in memory.
- One probe here (not a test, not tuned; uncompressed frames of `synthetic.dish_scene()`, frames 0 to 58 every 2, a 64 x 64 px patch, upsample 100): 0.59 ms per object and frame pair. Chaining the shifts drifted: 0.4, 2.1 and 2.5 px at most over 29 steps.
- Limits: the background inside the patch counts as much as the object; chained shifts drift, so a fixed reference patch and a measure of quality are needed; nothing handles occlusion.

### B6. Pyramidal iterative Lucas-Kanade (OpenCV)
- Tracks: points between two frames. "Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids"; a `status` and an `err` per point (docstring of `cv2.calcOpticalFlowPyrLK`, OpenCV 5.0.0).
- Prompt: points, best on corners or texture. License: "Apache 2.0" (installed `opencv_python_headless-5.0.0.93` METADATA). No weights. Already a base dependency (`pyproject.toml:16`). cpu only. Online: two frames in memory.
- The same probe (window 21 px, 3 pyramid levels, 10 threads, 1080p): 1.8 ms per frame pair for 3 or 100 points, 8.0 ms for 10,000, 48.7 ms for about 100,000. Largest error against the true centroid over 29 steps: 0.10 and 0.11 px for the two plain bodies, 1.52 px for the shrimp with moving antennae.
- Limits: assumes constant brightness and small motion per pyramid level; drifts; never finds a point again after it is hidden; on a rotating or deforming object the point is not the centroid.
- Seen in passing: torchvision, already installed, ships RAFT (dense flow, `torchvision/models/optical_flow/raft.py`). License of its weights: not verified.

### B7. Licenses at a glance
| method | code | weights | base, extra or user-installed |
|---|---|---|---|
| SAM 2.1, EdgeTAM | Apache 2.0 | Apache 2.0 | base or extra |
| TAPIR family | Apache 2.0 | Apache 2.0 | extra; git only, so vendored or installed by hand |
| CoTracker3 | CC-BY-NC 4.0 | CC-BY-NC 4.0 | user-installed at most |
| phase correlation | BSD | none | base |
| Lucas-Kanade | Apache 2.0 | none | base |

## C. A design sketch, for discussion

### C1. A backend declares what it can do
- One record per backend, readable without importing torch: key, label, family (mask, point, patch), what it gives (mask, point, visibility, shift), prompts it takes (point, negative point, box), several objects in one pass or not, online or needs a window of n frames, devices, grid cells or none, weights (address, bytes, hash, license), the extra it needs.
- One registry (a dict in one module) replaces the six lists of A2. The command line takes its `choices` from it, the model box its entries, `selftest` its timing constants. A backend whose extra is missing is shown, greyed, with the install line.
- The protocol must allow an answer that comes late (a chunk method returns frames k-8 to k at frame k) and an answer without a mask. Today's loop (A3) needs both changed.

### C2. Three ways to place a point method (choose with the owner)
- a. Points as tracks of their own: position and visibility only, no shape files.
- b. Hybrid: a point method steers the crop, a mask model draws the outline inside it. This is fine mode with the center from the point tracker instead of the last mask (`tracking_fine.py:164-183`). It aims at the measured weakness (A6).
- c. Patch methods give a shift of a box: position = start + shift. A start mask moved rigidly is not a measured outline and must not be written as one.

### C3. Optional dependencies
- Base without torch: numpy, scipy, pandas, OpenCV, scikit-image, the window. Lucas-Kanade and phase correlation then work in a small install. Extras in `[project.optional-dependencies]`: `sam` (torch, torchvision, transformers, timm, huggingface_hub, safetensors), `tapir`, `all`.
- Consequences: `gui/app.py` imports torch before PySide6 (CLAUDE.md), which must become conditional; `selftest` and the default model need a rule for an install without torch.
- "Public index servers SHOULD NOT allow the use of direct references in uploaded distributions" (https://packaging.python.org/en/latest/specifications/version-specifiers/). So on PyPI an extra cannot point at a git repository. TAPIR's PyTorch code would be vendored (Apache 2.0 allows it, with a NOTICE) or installed by hand. The present install, `uv tool install git+...` (README.md:12), has no such limit.

### C4. What the outputs say when a method gives no shape
- results.npz version 2: the method per track, and `visible` no longer tied to a mask; a reader for version 1.
- positions.csv keeps its rows and gains where the position comes from (mask centroid, point, patch shift). shapes.csv has no rows for such tracks, or empty cells; decide once and say it in the generated README.txt. radial.csv and outlines.npz stay header only, as decision X12 already does.
- Flags that need a mask are not computed for such tracks (not "clear"). New flags for points: low confidence, hidden, drift.
- Overlays draw a dot and the id when there is no outline; the window's overlay needs that change (A5).

### C5. One benchmark table across methods
- Extend docs/VALIDATION.md's method: synthetic clips with ground truth, limits per method, numbers only from measurements. Rows: backend x device. Columns: clip, object size in px, objects or points, frames, found (%), median and largest error in px, outline error (mask methods only), s per frame, peak memory (the child process of `tests/slow/test_memory.py`), install size, license.
- Clips: the present `dish_scene` (14.5 px bodies), `closeup_scene` (47 x 20 px), the selftest clip; new sweeps of size (3 to 200 px), count (1 to 1,000), length (100 to 10,000 frames), a crossing with occlusion, textured against flat objects, raw against H.264.
- Ground truth for a point is the body-fixed point, not the centroid; `synthetic.py` knows each pose, so it can give it.
- One command writes one row per run to a CSV; the table in the docs is generated from the CSVs, as docs/OUTPUTS.md is generated. Slow, not in CI, never on long real videos (CLAUDE.md).

### C6. What "scale orders of magnitude" can mean here
| axis | today | next orders | method class that serves it |
|---|---|---|---|
| things tracked | 1 to 10 objects with outlines (2.2 s per frame for 10 on cpu) | 100 to 100,000 points | Lucas-Kanade (measured, B6); TAP models for hundreds to thousands; not mask models |
| clip length | hundreds of frames (README.md:118: 601) | 10,000 to 1,000,000 | online methods with bounded state: B5, B6, causal TAPIR, TAPNext++, CoTracker3 online, SAM family with pruning |
| object size | `LOWRES` below 20 px or 20 grid cells (`session_parts.py:193`) | a few px, and hundreds of px | few px: point or patch methods, or the hybrid of C2; large: mask models |
| speed | 0.1 to 0.4 s per frame and object | ms per frame | B5, B6 on cpu; learned methods on a GPU |

### C7. Open questions for the owner
1. Which axis of C6 is meant, and is there a target (for example N points over M frames in so many minutes on a laptop)?
2. Is the tool still about outlines, or about positions and, where possible, outlines? Are point-only tracks wanted (C2a), the hybrid (C2b), or both?
3. The tool's license. May a non-commercial method (CoTracker3) be offered at all, and how? Or is the Apache-licensed TAPIR family enough?
4. Should torch become optional, and which method is then the default and the selftest?
5. Stay with `uv tool install git+...`, or publish on PyPI (then no git dependencies, C3)?
6. Must every offered method run on a laptop's cpu or mps? TAPNext++ is 2.5 GB of weights; mps is unconfirmed for every point model.
7. One method per session, or per track? What happens to version 1 results files?
8. SAM 2.1: add all four sizes? None was ever run here, so each needs its slow test first (DEVELOPER.md, "Adding a model").
9. May a backend hold a window of frames (CoTracker3 online, offline TAPIR)? CLAUDE.md says "Never load a whole video into memory"; what bound replaces it?
10. Is tracking backward from a click wanted (transformers' forward has `reverse`; not verified in use)?
11. Is JAX acceptable as an extra (the first TAPNext is JAX only), or PyTorch only?
12. Weights: pin a hash per file and refuse others? Rename the cache folder of A6?
