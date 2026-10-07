# Outline Tracker: specification

Version 1.2, Tue Oct 6, 2026 (built on J's Mac). Owner: J (the TA). Builder: Claude Code.
Working name: rename freely (package `outline_tracker`, command `outline-tracker`, repo `jd-anabi/outline-tracker`).

**Deadline: 16 students install and use this on their own laptops on Thursday Oct 8, 1 pm Pacific.** §14 sets the order of work, the go/no-go points and the cut lines. Read it before planning.

---

## 0. Read this first

**What it is.** A desktop app that works like Tracker (physlets.org/tracker), with EdgeTAM doing the tracking. A student:
1. opens a video;
2. sets the time scale (fps_true) and the distance scale (calibration stick);
3. puts the origin at the dish center;
4. clicks one or more objects on a frame;
5. presses Track.

EdgeTAM then follows each object's outline (its mask) through the clip. For every object and tracked frame the app records:
- the position: the area centroid of the outline;
- the outline's shape.

It also writes an overlay video, a log, and a session file that reopens the whole state.

**Why now.** The instructor wants EdgeTAM as the standard tracker: for brine shrimp this week, and for other objects in the final projects (weeks 5–10). Last week's EdgeTAM script (`shrimp.segment` in the student template) has two limits:
- it needs Tracker for calibration and start points;
- it outputs positions only.

**What it reuses.** These tested files from `https://github.com/jd-anabi/shrimp-tracker-template` are provided course code, already public:
- `src/shrimp/segment.py`: streaming EdgeTAM/SAM 2 through Hugging Face transformers with memory pruning, `mask_center`, the Tracker-format reader and writer, the calibration fit, overlay drawing, CHECK flags, selftest;
- `src/shrimp/_edgetam.py`: conversion of Meta's checkpoint;
- `src/shrimp/video.py`, `src/shrimp/convert.py`, `src/shrimp/check_video.py`;
- `tests/conftest.py`, `tests/test_segment.py`, `tests/test_video.py`, `tests/test_convert.py`.

Port them; do not rewrite them. Their behavior is the baseline (§13.3 is a regression test against them). Nothing else from that repo may be copied: its other modules are student work.

**What it does not do.** No velocities, accelerations, smoothing, distribution fits, statistics or PCA. The analysis template (a separate repo, built in a separate session) does those. This app measures; the template analyzes. §8 is the contract between them.

**Two modes of tracking.**
- **Coarse** (default): many objects at once on the whole dish. Good positions, but only a blob-level outline.
- **Fine**: one object at a time, on a crop that follows it, so the outline can resolve appendages such as antennae. §6.3 explains why both modes are needed.

---

## 1. Decisions and assumptions

| # | Decision | Decided by |
|---|---|---|
| D1 | EdgeTAM through Hugging Face `transformers==5.18.0`, streaming one frame at a time, as in `segment.py`; SAM 2.1 tiny stays available as an option | instructor / J (last week) |
| D2 | Runs locally on student laptops (Windows 10/11, Apple Silicon macOS), on CPU or Apple GPU; no Colab | J |
| D3 | Desktop GUI with PySide6 (Essentials) + pyqtgraph | Claude (proposed; J may override) |
| D4 | Positions and outline features both ship by Thursday | J |
| D5 | Position = area centroid of the full mask, with Tracker's +0.5 px pixel convention (`segment.mask_center`); the body frame for shape uses a separate "core" centroid (§7.3) | Claude (proposed) |
| D6 | Reaction-time stimulus: support both the dish wall (circle fit) and an LED (brightness probe) | J |
| D7 | The antenna option uses close-up clips and fine mode | J + Claude |
| D8 | Units, axes and time follow the student template: mm; origin at the dish center; x right, y up; $t_s = \text{frame}/f_\text{true}$ | template |
| D9 | The app computes no derivatives or statistics | Claude (proposed) |

Assumptions (flagged; J corrects if wrong):
- A1. Students open the `_tracker.mp4` copies made by `shrimp.convert` (H.264, every frame kept, GOP 24, rotation applied). The originals are HEVC `.MOV` at 240 fps.
- A2. Dish videos: a 35 mm dish fills a 1920×1080 landscape frame, about 32.4 µm/px. Close-up clips (antenna option) are recorded this week at higher magnification; their scale and frame rate are not known yet.
- A3. `fps_true` comes from the stopwatch clip and is recorded in `data/manifest.csv` (columns include `video_file`, `fps_true`, `dish_mm`).
- A4. Claude Code runs on J's Mac, assumed Apple Silicon. On an Intel Mac there is no current PyTorch, so the real-model steps would move to J's Windows machine. J also has a Windows machine. The Mac cannot show Windows-only behavior (the torch/Qt import bug of §10.2, Windows paths, HiDPI scaling), so the CI Windows job (§13.7) and one run on J's Windows machine at go/no-go 2 cover it. Conversely, the Mac is where the Apple-GPU (mps) path gets tested.
- A5. Students already ran last week's selftest, so the converted EdgeTAM weights sit in `~/.cache/shrimp-models/edgetam`.
- A6. Laptop speed from last week's estimate: one coarse object ≈ 0.25–0.4 s per frame on CPU; each extra object adds ≈ 0.5× that; peak RAM 1.7 GB measured for 10 objects.
- A7. The repo is created private during development, and J makes it public on Thursday morning so students can install with one command. Nothing private (answer keys, student data, rosters, personal paths) ever goes into it.

---

## 2. Users, workflow, scope

Students know Tracker's vocabulary: Clip Settings, Calibration Stick, Tape Measure, Circle Fitter → Move Origin to Center, Axes, point masses, autotrack, Export Data. Reuse those words in the GUI and the README.

The workflow, in order:
1. Enter your name (required; it names the run folder, §8.1).
2. Open the `_tracker.mp4`.
3. Clip: start frame, end frame, step (default 2).
4. Time: fps_true (from the manifest, from the stopwatch helper, or typed in).
5. Calibration stick (e.g., 30 mm), then a tape-measure check (e.g., 20 mm within 1%).
6. Circle fitter on the dish wall, then Origin to Center; axes angle if wanted.
7. Optional: brightness probe boxes (LED).
8. Add objects (A, B, C …): click on each, adding positive and negative points and a head point, and choose coarse or fine.
9. Track: background job with progress, ETA and cancel.
10. Review: step through the flags; re-track from a frame; end a track and continue it as A2.
11. Export: CSVs, overlay video, log. The session is saved continuously.

In scope by Thursday: §14.2 P0 and, if time allows, P1. Out of scope: everything in §14.2 P2.

---

## 3. Conventions

### 3.1 Image coordinates
Pixel (column $c$, row $r$) is the unit square $[c, c+1) \times [r, r+1)$. Its center is $(u, v) = (c + \tfrac12,\ r + \tfrac12)$, with $u$ to the right, $v$ down, and the origin at the top-left corner of the frame. This is Tracker's convention (its `pixelx, pixely` columns) and `segment.mask_center`'s.
- Every stored pixel coordinate (clicks, centroids, outlines, circle, stick, probes) uses this convention.
- Measurements on arrays indexed `[row, col]` add ½ to convert to $(u, v)$.
- Prompt coordinates go to the model unchanged, as in `segment.py`.

### 3.2 World coordinates
The calibration gives three things:
- the scale $k$ (mm per px), from the calibration stick;
- the origin $(u_0, v_0)$;
- the axis angle $\alpha$: the direction of $+x$, measured counterclockwise on screen from the image's rightward direction (Tracker's convention).

Then

$$\begin{pmatrix}x\\y\end{pmatrix} = k\,J\begin{pmatrix}u-u_0\\ v-v_0\end{pmatrix},\qquad J = \begin{pmatrix}\cos\alpha & -\sin\alpha\\ -\sin\alpha & -\cos\alpha\end{pmatrix},\ \det J = -1,$$

that is, $x = k[(u-u_0)\cos\alpha - (v-v_0)\sin\alpha]$ and $y = -k[(u-u_0)\sin\alpha + (v-v_0)\cos\alpha]$. The inverse is $u = u_0 + (x\cos\alpha - y\sin\alpha)/k$, $v = v_0 - (x\sin\alpha + y\cos\alpha)/k$.

With $\alpha = 0$: $x = k(u - u_0)$ and $y = -k(v - v_0)$, so y points up.

This is the "flip" form of `segment.Calibration` ($x = a u + c v + t_x$, $y = c u - a v + t_y$) with $a = k\cos\alpha$, $c = -k\sin\alpha$, $t_x = -(a u_0 + c v_0)$, $t_y = -(c u_0 - a v_0)$. It is also `tracker_map` in the template's `tests/test_segment.py`. A test checks both equivalences. Conversely, from a fitted `Calibration`: $k = \operatorname{hypot}(a, c)$, $\alpha = \operatorname{atan2}(-c, a)$, and $(u_0, v_0)$ = `to_px(0, 0)`.

Because $\det J = -1$, orientation reverses between $(u, v)$ and $(x, y)$: the sign of a signed area, and the sense of an angle computed with `atan2` on raw $(u, v)$, flip. Seen on screen, a counterclockwise rotation in world coordinates is still counterclockwise, since y is up. Compute every angle, signed area and polygon orientation in world coordinates.

### 3.3 Time
- $t_s = \text{frame}/f_\text{true}$. Frame 0 is 0 s.
- The clip's start frame and step select frames; they never shift $t$.
- The video file's frame rate is displayed, never used for physics.
- Warn (do not block) when $f_\text{true} < 100$: for the shrimp videos that usually means a re-timed copy.

### 3.4 Names, units, the frame grid
- Units appear in column names: `_mm`, `_mm2`, `_px`, `_s`, `_rad`.
- Track ids are capital letters, optionally followed by a piece number: `A`, `B`, `A2`, `A3` (regex `^[A-Z]+[0-9]*$`). Pieces of the same animal share the letter prefix; this is the template's convention.
- **The frame grid** is $\{\text{clip.start} + n\Delta\}$, with $\Delta$ the clip step. The slider and step buttons move on the grid. Start, re-track and new-piece frames are snapped to it (with a note if the user typed an off-grid frame). All tracks therefore share frames, which `CONTACT` (§9) needs.

### 3.5 Frame numbering and frame identity
Frame $n$ is the $n$-th frame of a sequential decode from the start of the file, with OpenCV, as in `segment.iter_rgb_frames`. This equals Tracker's numbering.

The frame displayed for index $n$ must be exactly the frame the tracker processes for index $n$ (§13.6 tests it). The ported `video.iter_frames(start > 0)` and `video.read_frame` seek with `CAP_PROP_POS_FRAMES` (video.py lines 100–101 and 120): do not use them for display or probes unless §13.6 proves them exact on our files. Acceptable strategies:
- decode forward from a cached earlier position;
- a PyAV packet/PTS index (then pin `av`, §15);
- verified OpenCV seeking.

**Runtime guard:** when the user prompts on a frame, store a hash of that decoded frame in the session. At run start, compare it with the hash of the frame the tracking decoder delivers. If they differ, abort with a clear message.

---

## 4. Time, calibration, dish, probes

### 4.1 fps_true
Three sources, shown in the GUI and logged:
1. **manifest**: port `segment._fps_from_manifest`. Search for `data/manifest.csv` upward from the video's folder; the user can also point to a manifest file. Remember that path between sessions (QSettings).
2. **stopwatch helper**: the user enters two frame numbers and the two stopwatch readings; $f_\text{true} = (f_b - f_a)/(t_b - t_a)$. If the readings are equal, raise an error.
3. **typed in**.

### 4.2 Calibration stick
The user places two endpoints by clicking (zoom in for precision; "Redo" clears them) and types the length $L_\text{mm}$. Then $k = L_\text{mm}/\lVert\mathbf p_2 - \mathbf p_1\rVert$.

Report the scale uncertainty from click precision $\sigma_c$ (default 0.5 px, stated in the log as an assumption):

$$\frac{\sigma_k}{k} = \frac{\sqrt2\,\sigma_c}{\lVert\mathbf p_2 - \mathbf p_1\rVert}.$$

Display the scale as "32.40 µm/px ± 0.08%" (a 30 mm stick at 32.4 µm/px spans 926 px). Warn when the stick is shorter than 300 px.

This is click precision only. A ruler at a different height from the shrimp adds a systematic error that the tape check cannot catch, because it uses the same ruler. The dish diameter is an independent check (§4.4).

### 4.3 Tape-measure check
The user clicks two other ruler marks and types their true distance. Show the measured distance $k\lVert\mathbf q_2 - \mathbf q_1\rVert$ and the relative error. It passes at |error| ≤ 1% (the template's rule). Log it.

### 4.4 Circle fitter (dish wall)
- The user clicks ≥ 3 points on the inner wall (6 or more, spread around, recommended); "Redo" clears them.
- Fit: algebraic (Kåsa) start, then geometric least squares $\min_{\mathbf c,R}\sum_i(\lVert\mathbf p_i-\mathbf c\rVert - R)^2$ with `scipy.optimize.least_squares`.
- Report the center (px), $R$ (px and mm) and the RMS residual (px).
- If the dish's inner diameter is known (manifest `dish_mm`, or typed in), show $2R$ (mm) next to it as an independent scale check. Only an approximate one: the wall's apparent edge depends on viewing angle and meniscus.
- A button sets the axes' origin to the circle center.
- The circle has three uses: the origin, the dish crop (§6.2), and the wall for the reaction-time option (§7.9).

### 4.5 Axes
The origin is placed by clicking; the angle is typed in degrees. Defaults: origin at the circle center if a circle exists, else at the frame center; $\alpha = 0$. Draw x and y arrows with labels.

### 4.6 Brightness probes (LED stimulus)
The equivalent of Tracker's RGB Region.
- Named rectangles (default `LED1`), defined by two corner clicks in the GUI or `--rect` on the command line (§11).
- For every frame in the clip (step 1, whatever the tracking step), compute the mean R, G, B and gray ($0.299R + 0.587G + 0.114B$) inside each rectangle.
- This is its own fast pass, decode-bound, with no model.
- Output: `probes.csv` (§8.7).

### 4.7 Sharing a calibration (P1)
Export and import `calibration.json`: fps_true and its source, stick, check, circle, axes, and the video's size and hash. A group can then share one calibration per video, as it shared one `.trk` last week.

---

## 5. Objects and prompts

- **Object table:** id, color, mode (coarse or fine), fine window $W$ (fine only; auto by default), start frame, status (no prompts / ready / tracked / ended).
- **Prompts** live on an object's start frame (and on correction frames, §6.6), always on the frame grid (§3.4):
  - left click: positive point;
  - negative point: right click (including a two-finger trackpad click), Alt/Option-click, or a click with the physical Control key. On macOS, Qt reports the Command key as `ControlModifier` and the Control key as `MetaModifier`, so check the platform's modifier;
  - Head tool: one click on the head. It is not sent to the model; it fixes the head side of the body axis (§7.3).
  - Ctrl/Cmd+Z: undo the last prompt.
  - Box prompts are P2.
- **Prompt encoding is new code.** `segment.py` sends exactly one positive point per object (lines 359–362). Multiple points and negative labels extend it, using the SAM-style nesting `input_points[image][object][point] = [x, y]` and `input_labels[image][object][point] ∈ {0, 1}`. Verify the nesting against transformers 5.18.0 and test it with the real model (§13.4).
- **Preview:** after every prompt change, run the model on that frame only and draw the masks of all objects prompted on it. Target under 1.5 s on a laptop CPU; show a busy indicator; never block the UI.
- **Different start frames are allowed.** A run is a set of tracks plus a start frame (§6.1), so objects added later, or on other grid frames, get their own runs.

---

## 6. Tracking

### 6.1 Runs
"Track" launches one run per group of pending objects with the same start frame and mode:
- **Coarse objects** that start on the same frame share one streaming session (one image encoding per frame).
- **Fine objects** each get their own session (§6.3). A fine object is not also tracked in coarse mode.

A run processes the grid frames from its start frame to the clip end, forward only. Backward tracking is P2.

### 6.2 Coarse mode, the backend, the dish crop
**Backend:** port `segment.TransformersSegmenter` and keep its behavior:
- the streaming session with explicit `frame_idx`;
- pruning of frames and outputs older than `KEEP_FRAMES = 20`;
- device auto (cuda > mps > cpu).

Extensions:
- return mask logits via `post_process_masks(..., binarize=False)` (the mask is logits > 0), for §7.4;
- multi-point and negative prompts (§5);
- **an Apple-GPU fallback in `step()` too.** `segment.py` falls back from mps to cpu only in `start()` (lines 372–382), but memory attention first runs on later frames. On an mps error in `step()`, restart the session on cpu at the current frame, re-seeded with a positive click at each object's last centroid, and log it.

**Dish crop:** when a circle exists and "Crop to dish" is on (default on), the model sees only the square $[u_c - R - m,\ u_c + R + m] \times [v_c - R - m,\ v_c + R + m]$ with $m = 0.03R$, clipped to the frame. Prompts are shifted into the crop, and outputs shifted back.
- Effect: the model resizes its input to 1024×1024 and predicts masks on a 256×256 grid. One grid cell therefore covers $W_\text{in}/256 \times H_\text{in}/256$ input pixels:
  - full 1920×1080 frame: 7.5 × 4.2 px, about 0.24 × 0.14 mm at 32.4 µm/px;
  - a 1080×1080 dish crop: 4.2 × 4.2 px, isotropic.
- The crop also removes the ruler and the background around the dish.

**Regression mode:** with "Crop to dish" off and one positive click per object, coarse mode must reproduce `segment.track_video` exactly (§13.3).

### 6.3 Fine mode (a crop that follows one object)
**Why.** A 0.47 mm nauplius in the dish view is about 2–3 grid cells long on the full frame and about 3.4 with the dish crop. Such an outline carries area and orientation at best; antennae tens of µm wide are invisible to it. Fine mode feeds the model a $W \times W$ crop centered on one object, so a grid cell is $W/256$ px. With $W = 3L$ ($L$ = object length in px) the object spans about 85 cells at any magnification. The limit then becomes the camera's own pixels and motion blur. That is why the antenna option also needs close-up footage, and why §7.8 checks camera pixels as well as grid cells.

**Algorithm** (one streaming session per fine object):
1. **Window:** $W$ is user-set, or by default $W = \operatorname{clip}(\lceil 3F\rceil, 96, 512)$ px, where $F$ is the maximum Feret diameter (px) of the object's preview mask on its start frame. $W$ stays fixed for the run, so the object's scale in the model's input never changes.
2. **Crop center:** on the start frame, the centroid of the preview mask. On each later frame, the object's centroid on the previous tracked frame; if it is lost, keep the last center (and flag `LOST`). Round the crop corner to integer pixels. Where the window extends past the frame, pad by edge replication (`cv2.BORDER_REPLICATE`), so the crop size never changes.
3. **Coordinates:** shift prompts into crop coordinates, and shift masks and logits back by the crop corner.
4. **Cost:** each fine object costs about one single-object coarse run (the model input is always 1024×1024). Recommend 1–3 fine objects per video and step 1 for antenna work. At 240 fps a 9 Hz stroke then has about 27 samples per cycle; at step 2, about 13.

### 6.4 Execution
- **Core API:** `run_job(job, callbacks)`. Callbacks: `progress(done, total, s_per_frame, eta_s)`, `frame_result(track_id, frame, record)`, `log(msg)`, `finished(status)`.
- **Threading:** the GUI runs it in a worker thread (QThread). Set `torch.set_num_threads(max(1, torch.get_num_threads() - 1))` so the UI stays responsive.
- **ETA up front:** show the estimated run time before the run starts (selftest timing × frames × objects).
- **Cancel** stops after the current frame and keeps everything tracked so far.
- **Autosave** every 200 tracked frames, as `segment.py` does: results and session to the run folder. Partial outputs are valid and marked `"complete": false` in `session.json`.
- **P1:**
  - live view: the latest tracked frame with outlines, at most twice a second;
  - resume after cancel: "Re-track from here" at the first untracked frame, with an automatic positive click at each object's last centroid.

### 6.5 Memory and speed budgets
- Memory must stay flat in the number of frames: keep the pruning.
- Peak RAM ≤ 3 GB for 10 coarse objects at 1080p (1.7 GB was measured last week).
- Never hold the decoded video. Decode, process and discard each frame; the display keeps an LRU cache of at most 64 frames.
- Per object and frame, keep only bounding-box crops of the mask and logits (bbox ± 8 px). No full-frame float arrays survive past the current frame.
- Report s/frame and ETA. Expected values are in A6.

### 6.6 Corrections
Last week's synthetic 10-shrimp test lost 3 of 10 tracks: two after contact, and one that jumped to an untracked shrimp with nothing touching it. Real videos showed lost and swapped shrimp too. Corrections are therefore P0:
- **Re-track from here:** the user goes to grid frame $k$, selects the track or tracks, and adds new prompts (a positive point on the right animal, negative points on the wrong one). A new run starts at $k$ for those tracks only. Their results at frames ≥ $k$ are replaced; other tracks are untouched. The correction is recorded in `session.json`.
- **End track here:** the track's results after frame $k$ are deleted. Its rows stop at $k$ (§8.2).
- **Continue as new track:** at grid frame $m$, create the next free piece (e.g., `A2`) with its own prompts at $m$, tracked from $m$.

---

## 7. Measurements

The tracking stage stores **pixel-space** records (§8.12, `results.npz`). Every world-unit quantity is derived at export from those records and the current calibration. Changing the calibration or fps_true therefore re-exports in seconds without re-tracking (a test, §13.2).

For each track and tracked frame, given the binary mask $M$ (logits > 0) in full-frame pixel coordinates and the logits $\Lambda$ where available:

### 7.1 Visibility and area
The track is visible iff $M$ is non-empty. $A_\text{px}$ is the pixel count of $M$, and $A = k^2 A_\text{px}$ (mm²).

### 7.2 Position (the official position)
$(\bar u, \bar v)$ is the mean of the pixel centers of $M$, exactly `segment.mask_center`. Then $(\bar x, \bar y)$ comes from §3.2.

This is "the center of the contour" in the sense of the area centroid, $\bar{\mathbf r} = \frac1A\iint_M \mathbf r\, dA$. It is not the perimeter-weighted centroid $\frac1P\oint\mathbf r\,ds$, which appendages pull much harder (much perimeter, little area). It keeps this week's numbers comparable with last week's.

### 7.3 Core mask, orientation, head
**Why a core.** With resolved antennae spread sideways, the full mask's second moments can be dominated by the antennae. For a 0.47 × 0.2 mm body with two 0.3 mm antennae, the lateral second moment slightly exceeds the axial one, so the full-mask major axis would swing about 90° on every stroke. The body frame therefore uses a **core** mask with thin appendages removed:
- **Core:** the largest component of the morphological opening of $M$ with a disk of radius $r_o = \max(1, \operatorname{round}(0.1\,L_{1,\text{px}}))$ px. $L_{1,\text{px}}$ is the full mask's equivalent-ellipse major axis (below), and the factor 0.1 is configurable (`core_open_frac`).
- **Fallback:** if the opening removes more than half of the area, use $M$ itself for that frame and flag `ORIENT`.
- **Centroid:** the core centroid $\mathbf r_c$, exported as `core_x_mm, core_y_mm`. It is the "body center", which the antennae barely move.

**Moments.** For any mask $X$, map its pixel centers to world coordinates $\mathbf r_i$ and form $\Sigma_X = \frac{1}{|X|}\sum_i (\mathbf r_i - \bar{\mathbf r}_X)(\mathbf r_i - \bar{\mathbf r}_X)^\top$ (mm²), with eigenvalues $\lambda_1 \ge \lambda_2$ and unit eigenvector $\mathbf e_1$ for $\lambda_1$.
- Store the pixel-space covariances. At export, $\Sigma = k^2 J\,\Sigma_\text{px} J^\top$ with $J$ from §3.2.
- **From the full mask** (shape signals): axes $L_1 = 4\sqrt{\lambda_1}$ and $L_2 = 4\sqrt{\lambda_2}$ (mm), and eccentricity $e = \sqrt{1-\lambda_2/\lambda_1}$.
- **From the core** (body frame): the axis $\mathbf e_1^\text{core}$, defined only mod $\pi$.

**Head direction** $\hat{\mathbf h} = s\,\mathbf e_1^\text{core}$, $s = \pm1$:
- on the track's start frame, $s$ makes $\hat{\mathbf h}\cdot(\mathbf r_\text{head} - \mathbf r_c) > 0$, where $\mathbf r_\text{head}$ is the user's head click;
- without a head click, $s$ makes $\hat{\mathbf h}$ agree with the core centroid's displacement over the first 10 tracked frames. Every frame of that track gets the flag `HEADGUESS`, since all inherit the guess;
- on later frames, $s$ maximizes $\hat{\mathbf h}_i\cdot\hat{\mathbf h}_\text{ref}$, where "ref" is the last earlier frame not flagged `ORIENT`.

**Export:** the heading $\theta = \operatorname{atan2}(h_y, h_x)$, unwrapped so it is continuous and may leave $(-\pi, \pi]$.

**Flag `ORIENT`** when:
- $\lambda_2/\lambda_1 > 0.8$ for the core (nearly round, axis undefined); or
- $|\hat{\mathbf h}_i\cdot\hat{\mathbf h}_\text{ref}| < \cos 60^\circ$ (the axis jumped); or
- the core fallback was used.

### 7.4 Outline
- **Extraction:** the 0-level set of $\Lambda$ (marching squares, `skimage.measure.find_contours`) for the largest connected component of $M$. Without logits, use the boundary of $M$ (`cv2.findContours`, `CHAIN_APPROX_NONE`).
- **Contour choice:** outer contour only; holes are ignored. Record the number of components and the largest component's area fraction.
- **Storage:** resample the dense contour to 256 points in pixel space (`results.npz`).
- **At export:**
  - map the contour to world coordinates and orient it counterclockwise there (signed area > 0);
  - perimeter $P$ is the polygon length;
  - resample to $N = 128$ points equally spaced in arc length ($\Delta s = P/N$), starting at the **head point**: where the ray from $\mathbf r_c$ along $\hat{\mathbf h}$ leaves the polygon (the farthest crossing).

### 7.5 Body frame
$R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$ rotates counterclockwise in world coordinates. Body coordinates are $(\xi, \eta)^\top = R(-\theta)\,(\mathbf r - \mathbf r_c)$: $\xi$ points toward the head, and $\eta$ is 90° counterclockwise from $\xi$ (world coordinates, y up). Do not call $\eta$ "left": Artemia often swims ventral side up.

### 7.6 Radial profile
$r(\phi_j)$ for $\phi_j = j\cdot 5^\circ$, $j = 0,\ldots,71$, with $\phi$ measured counterclockwise from $\hat{\mathbf h}$. It is the distance from $\mathbf r_c$ to the farthest crossing of the ray at angle $\phi_j$ with the outline polygon (vectorized ray–segment intersection).
- For a star-shaped outline this is the boundary itself.
- Otherwise it is the outer envelope; README.txt says so.
- $r(0)$ is the head radius.

### 7.7 Shape scalars
All from the stored 256-point outline polygon:
- solidity $S = A_\text{poly}/A_\text{hull}$ (shoelace area; convex hull via `scipy.spatial.ConvexHull`);
- circularity $4\pi A_\text{poly}/P^2$;
- maximum Feret diameter $F_\text{max}$: the largest distance between hull vertices;
- the number of components and the largest component's area fraction;
- `core_frac` $= |{\rm core}|/|M|$.

### 7.8 Resolution indicators
- `px_along_major` $= L_1/k$: the camera's pixels along the full mask's major axis.
- `cells_along_major` $= (L_1/k)/c$, where $c = \max(W_\text{in}, H_\text{in})/256$ px for the image the model saw (full frame, dish crop, or fine crop).
- `shape_ok` $= 1$ iff $\min(\texttt{px\_along\_major}, \texttt{cells\_along\_major}) \ge 20$ (configurable, `shape_ok_min`). Otherwise flag `LOWRES`.

Both limits matter. Fine mode on a dish-scale shrimp has many grid cells, but only about 14 camera pixels along the body. The threshold of 20 is a heuristic: tune it on real clips and report the choice in docs/VALIDATION.md.

### 7.9 Distance to the dish wall (when a circle exists)
With the circle center $\mathbf c$ in world coordinates:
- $d_c = R - \lVert\bar{\mathbf r} - \mathbf c\rVert$ for the position;
- $d_\text{min} = R - \max_s \lVert\mathbf r(s) - \mathbf c\rVert$ for the outline's nearest point ("contact").

Both are in mm; a negative value means outside the fitted circle.

---

## 8. Outputs: the contract with the analysis template

### 8.1 Run folder
One run folder per video per student. The default is `<video folder>/<video stem>_outline_<student>/`, so students sharing a video folder never overwrite each other; the user can choose another location. The folder is self-contained and relocatable: paths inside `session.json` are relative to it. The video is stored by relative path, absolute path, size, and the SHA-256 of its first 64 MiB, so it can be found again.

All output files come from one source of truth: `outline_tracker/schema.py` (column names, order, dtypes, units, descriptions). README.txt and docs/OUTPUTS.md are generated from it.

```
<run folder>/
  session.json        everything needed to reopen, re-run and re-export (8.10)
  positions.csv       one row per track per tracked frame (8.2)
  edgetam/A.csv ...   Tracker-format copies, folder named after the model (8.3)
  shapes.csv          shape scalars per track per frame (8.4)
  radial.csv          body-frame radial profile, fine tracks (8.5)
  outlines.npz        resampled outlines, fine tracks (8.6)
  probes.csv          brightness probes, every frame (8.7; only if probes exist)
  overlay.mp4         the clip with outlines drawn (8.8)
  results.npz         internal pixel-space store (8.12)
  run.log             human-readable log (8.9)
  README.txt          what each file and column means (8.11)
```

Writes are atomic (temporary file, then rename), because cloud-sync clients lock files. On Windows, retry for a few seconds if the target is locked (e.g., a CSV open in Excel). If it stays locked, write `<name>.new.csv` and tell the user.

### 8.2 positions.csv
Rows: for each track, every grid frame from its first to its last, sorted by `track_id`, then `frame`. Frames where the track is lost keep their row with NaN positions.

| column | type | unit | meaning |
|---|---|---|---|
| track_id | str | | A, B, …; pieces A2, A3 |
| frame | int | | video frame number (§3.5) |
| t_s | float | s | frame / fps_true |
| x_mm | float | mm | area centroid of the full mask, in the user's axes; NaN if not visible |
| y_mm | float | mm | same, y up |
| u_px | float | px | the same point in image coordinates (Tracker's pixelx) |
| v_px | float | px | same (Tracker's pixely) |
| area_mm2 | float | mm² | mask area |
| visible | int | | 1 if the mask is non-empty |
| mode | str | | `coarse` or `fine` |
| flags | str | | QC codes (§9), separated by `;`; empty if none |

Number formats: t with 7 decimals, mm with 6, px with 3 (as in `segment.write_tracker_file`).

### 8.3 \<model\>/\<id\>.csv (Tracker format)
Port `segment.write_tracker_file` itself, so the bytes are identical to the files students' `load_tracks` already read last week: a line `,A,,,,,`, then the header `t,frame,x,y,pixelx,pixely`, then one row per tracked frame, with x and y empty where the track is lost.

The folder is named after the model (`edgetam/` or `sam2/`), as last week. The template's `compare.py` identifies the tool by that folder name. A folder named `tracks/` would be mislabeled as Tracker data and would break its figure paths.

Tests parse these files with the vendored `segment.read_tracker_export` (§13.3). Do not vendor the template's `load.py`: it is a student stub.

### 8.4 shapes.csv
`track_id, frame, t_s, area_mm2, perimeter_mm, major_mm, minor_mm, eccentricity, theta_rad, core_x_mm, core_y_mm, core_frac, solidity, circularity, feret_max_mm, n_components, largest_fraction, px_along_major, cells_along_major, shape_ok, wall_dist_centroid_mm, wall_dist_min_mm, flags`

- `major_mm`, `minor_mm` and `eccentricity` come from the full mask.
- `theta_rad` (heading, unwrapped), `core_x_mm` and `core_y_mm` come from the core (§7.3).
- The wall distances are NaN without a circle.

Same rows as positions.csv, for every track.

### 8.5 radial.csv
`track_id, frame, t_s, r_000, r_005, …, r_355` (mm). Column `r_ddd` is the radius at $\phi$ = ddd degrees counterclockwise from the head direction (§7.6). Rows are NaN when not visible.

Written for fine tracks by default. The setting `shape_files_for_coarse` (default off) adds coarse tracks; their outlines are blobs (`LOWRES`).

### 8.6 outlines.npz
`np.savez_compressed`, with keys per track:
- `<id>__frames` (int32 $[n]$);
- `<id>__xy_mm` (float32 $[n, N, 2]$, world coordinates);
- `<id>__xieta_mm` (float32 $[n, N, 2]$, body frame of §7.5).

Outlines run counterclockwise in world coordinates and start at the head point; rows for invisible frames are NaN. A key `meta` holds a JSON string with N, the conventions, and the tool version.

Fine tracks by default (same setting as §8.5). About 2 MB per fine track for 1,200 frames.

### 8.7 probes.csv
`frame, t_s, probe, r, g, b, gray`: means on the 0–255 scale, for every frame of the clip (step 1).

### 8.8 overlay.mp4
- **Encoding:** H.264 yuv420p via `imageio-ffmpeg`, as in `segment.py`. Playback at 30 fps (a 240 fps clip at step 2 plays 4× slower than real time). 960 px wide by default.
- **P0 content:** per track, the outline (track color), centroid dot and id label; the stamp `t = … s   frame …`.
- **P1 content:** a trail of the last 0.5 s, a head tick, the dish circle, the axes, a 1 mm scale bar, a full-resolution option, and `overlay_<id>_zoom.mp4` per fine track (the follow-crop upscaled to 512×512 with its outline, for slides on antenna motion).
- **Production:** the overlay is made from `results.npz` by re-decoding the clip, not during tracking, so corrections show up. It must play in PowerPoint and Google Slides.

### 8.9 run.log
Plain text, appended per run and per export. Contents:
- **Software and machine:** tool version and commit (from `direct_url.json` in the installed package's metadata, since an installed tool has no `.git`); date; OS, Python, torch, transformers; device.
- **Model:** model id and the SHA-256 of the weights file.
- **Video:** path, size, SHA-256 of the first 64 MiB, container fps, frame count, resolution.
- **Time:** fps_true and its source.
- **Calibration:**
  - $k$ ± its uncertainty, the stick endpoints and length;
  - the tape check (true, measured, error %);
  - the circle (center, R, RMS, and $2R$ vs `dish_mm` if known);
  - the axes (origin, $\alpha$).
- **Runs:** for each run, the tracks, mode, start frame, step, number of frames, crop (dish, or fine $W$), s/frame and total time.
- **Corrections.**
- **QC summary:** counts per flag per track, with first occurrences and times, in the style of `segment.py`'s CHECK lines.
- **Outputs:** the files written, with sizes.

### 8.10 session.json (schema version 1)
```json
{
  "format": "outline-tracker-session", "schema_version": 1, "tool_version": "0.1.0",
  "student": "", "notes": "", "complete": true,
  "video": {"relpath": "../groupB_2026-10-06_1325_main_tracker.mp4", "abspath": "...", "size": 0,
            "sha256_first_64MiB": "...", "width": 1920, "height": 1080, "n_frames": 0, "fps_container": 240.0},
  "clip": {"start": 0, "end": 2399, "step": 2},
  "time": {"fps_true": 239.6, "source": "manifest|stopwatch|typed", "manifest_path": null,
           "stopwatch": {"frame_a": 0, "time_a_s": 0.0, "frame_b": 0, "time_b_s": 0.0}},
  "calibration": {"stick": {"p1_px": [0, 0], "p2_px": [0, 0], "length_mm": 30.0},
                  "click_sigma_px": 0.5,
                  "check": {"p1_px": [0, 0], "p2_px": [0, 0], "true_mm": 20.0}},
  "axes": {"origin_px": [960.5, 540.5], "angle_deg": 0.0},
  "circle": {"points_px": [[0, 0]], "center_px": [0, 0], "radius_px": 0.0, "rms_px": 0.0, "dish_mm": null},
  "processing": {"model": "edgetam", "device": "auto", "dish_crop": true,
                 "outline_points": 128, "radial_step_deg": 5, "shape_ok_min": 20,
                 "core_open_frac": 0.1, "jump_mm_s": 100.0, "fine_window_factor": 3.0,
                 "shape_files_for_coarse": false},
  "probes": [{"name": "LED1", "rect_px": [0, 0, 0, 0]}],
  "tracks": [{"id": "A", "color": "#00FFFF", "mode": "coarse", "fine_window_px": null,
              "start_frame": 0, "head_px": null, "ended_at": null,
              "prompts": [{"frame": 0, "frame_hash": "...", "points_px": [[0, 0]], "labels": [1]}]}],
  "runs": [{"tracks": ["A"], "start_frame": 0, "mode": "coarse", "started": "ISO-8601",
            "finished": "ISO-8601", "frames_done": 0, "seconds_per_frame": 0.0, "device": "cpu"}],
  "corrections": [{"time": "ISO-8601", "tracks": ["A"], "action": "retrack|end|new_piece",
                   "frame": 0, "prompts": []}]
}
```
The session is saved on every change (debounced) and on exit. Loading an older or newer `schema_version` must give a clear message, not a crash.

### 8.11 README.txt
Generated from `schema.py`. It describes:
- every file and column (units, meaning);
- the conventions of §3;
- the flags of §9;
- what goes into git and what does not (§8.13).

### 8.12 results.npz (internal)
Pixel-space records for every track (coarse and fine):
- `frames`, `visible`, `area_px`;
- `u`, `v`;
- `cov_full` and `cov_core` ($[n, 3]$: the covariance of pixel centers, $\mu_{uu}, \mu_{uv}, \mu_{vv}$, px²);
- `core_u`, `core_v`, `core_frac`;
- `outline_px` ($[n, 256, 2]$ float32);
- `n_components`, `largest_fraction`;
- `cell_px`, `edge` (mask touches the border of the model's input), `mode`, `score` (object score if the backend exposes it, else NaN).

"Re-track from here" replaces the entries for frames ≥ $k$ of the selected tracks. "Export" recomputes every output from this file plus `session.json`. The format carries a version key.

### 8.13 What goes into git
Small files go into git: the CSVs, the Tracker-format folder, `session.json`, `run.log`, `README.txt` and `calibration.json`.

`outlines.npz` must also be committed, since it is the only full-shape output. The student template's `.gitignore` excludes `*.npz`, so the new analysis template must whitelist it (`!**/outlines.npz`). `results.npz` and `*.mp4` stay out; share them through Drive.

---

## 9. Quality flags

| code | condition | meaning for the student |
|---|---|---|
| `LOST` | mask empty | no position; re-click if needed |
| `JUMP` | centroid speed between consecutive visible frames > `jump_mm_s` (default 100 mm/s, as last week) | probably jumped to another object |
| `SIZE` | area outside [0.5, 2] × the track's median area (as last week) | merged with a neighbor, or partly lost |
| `CONTACT` | this track's outline comes within max(3 px, 2 coarse grid cells) of another track's outline on the same frame | identities may swap here: check |
| `EDGE` | the mask touches the border of the model's input (frame, dish crop, or fine crop) | outline may be cut off |
| `MULTI` | more than one component, the second ≥ 10% of the largest | ragged or split mask |
| `LOWRES` | `shape_ok` = 0 (§7.8) | shape columns unreliable; use fine mode or closer footage |
| `ORIENT` | §7.3 (core nearly round, axis jump, or core fallback) | heading unreliable |
| `HEADGUESS` | head side taken from motion (no head click); on every frame of the track | check the head in the overlay |

GUI support (P0):
- a flags table (track, frame, t, code) whose rows jump to the frame;
- previous/next-flag buttons for the selected track.

A colored timeline strip under the slider is P1. The log carries the summary (§8.9).

---

## 10. GUI (PySide6 + pyqtgraph)

### 10.1 Main window (P0 unless marked)
- **Video view (center):** a pyqtgraph `ImageItem` in a `ViewBox`:
  - `imageAxisOrder='row-major'` and `invertY(True)`;
  - the ViewBox context menu disabled (right click is a negative prompt);
  - wheel zoom, drag pan, Fit and 1:1 buttons.

  In item coordinates, pixel $(c, r)$ spans $[c, c+1)\times[r, r+1)$, so clicks already arrive in §3.1 $(u, v)$: do not add ½.
  - Overlays: outlines (or translucent mask fill, toggle), centroids, ids, head marks, and the tool graphics (stick, tape, circle points and circle, axes, probe boxes, fine-mode window).
  - P0 places tool points by clicking, with Redo. Draggable handles are P1.
- **Status bar:** the cursor position in px and mm, the gray value under the cursor, the device, the model state (loading / ready), and the last message.
- **Bottom:** the frame slider on the grid, with buttons for first, −10 steps, −1 step, play/pause, +1, +10 and last. A frame-number box and the time $t$.
- **Right dock, in workflow order** (numbered like a Tracker tutorial):
  1. Student name; video: open; show the info and `check_video` warnings; clip start, end and step. ("Convert for tracking" for `.MOV` files is P1; meanwhile the `convert` command, §11.)
  2. Time: fps_true with its source; a Stopwatch… dialog; manual entry.
  3. Calibration: the Stick tool and length; the scale with its uncertainty; the Tape check tool, true length, error % (green/red).
  4. Dish and axes: the Circle tool and fit result; Origin to Center; axes angle; a "Crop to dish for tracking" checkbox.
  5. Probes (P1 in the GUI; P0 on the command line): add a box; name; Measure.
  6. Objects: the table of §5; Add, Remove; prompt tools (positive, negative, head).
  7. Track: model (EdgeTAM default, SAM 2.1 tiny), device (auto/cpu/mps/cuda), estimated time, Track, progress bar, s/frame, ETA, Cancel.
  8. Review and fix: flags table; previous/next flag; Re-track from here; End track here; Continue as new track.
  9. Export: run folder; Export all (CSVs, overlay, log, README.txt); Open folder.
- **Menus:** File (Open video, Open session, Save session, Save session as, Export, Quit), Help (Quickstart, About with all versions).

### 10.2 Interaction and startup rules
- **Tool modes:** Pan (default), Stick, Tape, Circle, Axes, Probe, Positive, Negative, Head. Esc returns to Pan.
- **Keys:**
  - ←/→: ±1 step;
  - Shift+←/→: ±10 steps;
  - Home/End;
  - Space: play/pause;
  - Ctrl/Cmd+S: save;
  - Ctrl/Cmd+Z: undo last prompt;
  - 1–9: select object.
- **Windows import order (critical):** with PyTorch 2.9.x on Windows, importing torch after PySide6/PyQt fails with WinError 1114 (c10.dll). The GUI entry point therefore imports torch before anything from PySide6, at startup in the main thread, on every platform. A Windows CI job checks it (§13.7).
- **Non-blocking work:** model weights load in a background thread after the window appears, and tracking buttons stay disabled until the model is ready. No long operation runs on the UI thread.
- **Error dialogs** say what to do next in plain language: no stack traces for students, but stack traces go to `run.log`.

---

## 11. Command line

`outline-tracker` with no arguments opens the GUI. Subcommands:

| command | priority | does |
|---|---|---|
| `gui [VIDEO \| SESSION.json]` | P0 | open the GUI with a video or session |
| `from-tracker VIDEO EXPORT [--fine IDS] [--step K] [--seconds S] [--fps F] [--out DIR] [--model edgetam\|sam2] [--device D]` | P0 | last week's workflow with the new outputs (below). **The fallback if the GUI slips.** |
| `export SESSION.json [--overlay]` | P0 | regenerate all outputs from `results.npz` and the session (e.g., after fixing the calibration) |
| `probe VIDEO --rect NAME:u0,v0,u1,v1 [...] [--start F] [--end F] [--fps F] [--out DIR]`, or `probe SESSION.json` | P0 | brightness probes only; no model |
| `selftest [--model M] [--device D]` | P0 | the ported selftest |
| `convert VIDEO`, `check VIDEO` | P0 | the ported `shrimp.convert` and `shrimp.check_video` |
| `run SESSION.json [--from-frame K]` | P1 | run every pending run in the session, headless (overnight batches) |
| `selftest --fine` | P1 | adds the synthetic close-up shrimp |

**`from-tracker` details:**
- Calibration and start points come from a Tracker export (one track, or a `#multi` start file), via the ported `read_tracker_export`, `fit_calibration` and `make_plan` (module `tracker_io.py`).
- Convert the fitted map to $(k, \alpha, u_0, v_0)$ with §3.2. If the fit is the mirrored form (`flip=False`), refuse with a clear message.
- It runs on the full frame (there is no circle, so no dish crop), exactly like last week, and writes the full §8 output set.
- The objects are coarse, except those listed in `--fine` (fine windows from the first-frame masks).

---

## 12. Architecture

```
outline_tracker/
  schema.py         column tables, flag codes, file names: single source of truth
  geometry.py       image<->world transform, stick, tape, circle fit, fps helpers, manifest lookup
  tracker_io.py     ported read_tracker_export, write_tracker_file, fit_calibration, make_plan
  video.py          ported (probe, check_video) + FrameSource: sequential decode + exact random access
  convert.py        ported
  segmenter/
    base.py         Segmenter protocol, ObjectPrompt, MaskResult
    hf.py           ported TransformersSegmenter (+ logits, multi-point/negative prompts, mps fallback in step)
    edgetam_convert.py  ported _edgetam.py (same cache path and env var)
    fake.py         stand-ins for tests: ThresholdFake (dark components near the last position, like the
                    template's DiskFinder) and ExactFake (ground-truth masks from synthetic.py, told the
                    runner's crop offset through a test hook); both return signed-distance logits
  tracking.py       runs, coarse runner (dish crop), fine runner (follow crop), cancel/progress/autosave,
                    re-track, pieces, frame-hash guard
  measure.py        mask (+ logits) -> pixel-space record (area, centroid, covariances, core, outline,
                    components, edge, cell)
  results.py        results.npz store with partial replacement (tracks x frames >= k)
  derive.py         pixel-space records + calibration -> world quantities (heading continuity, outline
                    resampling, body frame, radial profile, shape scalars, wall distances)
  qc.py             flags
  probes.py         brightness probes
  session.py        dataclasses, JSON (de)serialization, schema version
  export.py         all files of §8
  overlay.py        drawing and MP4 writers
  synthetic.py      test videos with ground truth
  cli.py            entry point and subcommands
  gui/              app.py, main_window.py, video_view.py, panels/*.py, worker.py
```

Segmenter protocol:
```python
class Segmenter(Protocol):
    def start(self, image: np.ndarray, prompts: list[ObjectPrompt]) -> list[MaskResult]: ...
    def step(self, image: np.ndarray) -> list[MaskResult]: ...
    def close(self) -> None: ...

@dataclass
class ObjectPrompt:       # coordinates in the pixel frame of `image` (§3.1)
    obj_id: str
    points_px: list[tuple[float, float]]
    labels: list[int]     # 1 positive, 0 negative
    box_px: tuple[float, float, float, float] | None = None

@dataclass
class MaskResult:         # mask and logits cropped to bbox ± 8 px, with the crop's offset in `image`
    obj_id: str
    offset: tuple[int, int]          # (col0, row0)
    mask: np.ndarray                 # bool
    logits: np.ndarray | None        # float32, same shape as mask
    score: float | None
```

Rules:
- Nothing outside `gui/` imports Qt.
- `gui/` contains no measurement or file-format logic.
- torch and transformers are imported only inside `segmenter/` (`hf.py`, `edgetam_convert.py`), lazily. The one exception is the GUI entry point's import-order rule (§10.2). Tests, `export` and `probe` never load them.
- `pathlib` everywhere. Paths with spaces and non-ASCII characters must work.
- Every public function's docstring states its units and coordinate frame.
- Keep files focused: when one grows past ~400 lines, split it.

---

## 13. Tests (test-first)

Expected values come from geometry, from analytic shapes, or from synthetic ground truth, never from running the code and copying its output. Never weaken or delete a test to make it pass. If a test seems wrong, mark it `xfail(strict=True, reason=...)`, record it under "Questions for J" in docs/PLAN.md, and continue; J decides.

### 13.1 Unit tests (fast)
- **Transform:** forward/inverse round trip for $\alpha$ = 0°, 90°, −30°. y is up. Equivalence with `segment.Calibration` (flip form) and with `tracker_map` for random $k, \alpha, u_0, v_0$. The `Calibration` → $(k, \alpha, u_0, v_0)$ conversion. The mirrored form is refused.
- **Stick, tape, fps:** stick scale and uncertainty; tape check error; fps from stopwatch, including the equal-readings error.
- **Circle fit:** 6 points with σ = 0.5 px noise on R = 500 px (fixed seed) give center and radius within 1.0 px; 3 points give an exact fit.
- **Centroid:** `mask_center` equals `segment.mask_center` on random masks.
- **Moments:** on rasterized ellipses (a = 40, b = 15 px, several angles), $L_1, L_2$ within 2% of $2a, 2b$ and the axis angle within 1°. The world-frame angle comes out correct when $\alpha \ne 0$, despite $\det J = -1$.
- **Core:** an ellipse body with two long thin rods spread sideways (lateral second moment > axial). The core axis stays within 5° of the body axis, the core centroid within 0.5 px of the body centroid, and `core_frac` matches the body/total area within 5%.
- **Head continuity:** an ellipse rotating through 2π with a head click on frame 0 gives no sign flips, and the unwrapped θ matches truth within 2°. The body-with-rods shape beating through ±40° keeps its heading within 5°. Without a head click, `HEADGUESS` appears on every frame and the head is taken from the motion.
- **Outline geometry** (logits = signed distance to a known shape, so the level set is known):
  - circle R = 50 px: perimeter within 0.5% of $2\pi R$;
  - $N$ points, counterclockwise in world coordinates, equal spacing within 1%;
  - the outline starts at the head point.
- **Radial profile:** constant $R$ for a circle (within 0.5 px). For an ellipse with semi-axes $a$ along the head direction and $b$ across, $r(\phi) = ab/\sqrt{b^2\cos^2\phi + a^2\sin^2\phi}$ within 1%.
- **Solidity and wall distance:**
  - a disk gives ≥ 0.99;
  - an L-shape gives its analytic value within 1%;
  - wall distances for points at known radii.
- **Resolution:** `shape_ok` uses $\min(\texttt{px\_along\_major}, \texttt{cells\_along\_major})$. A 14 px object in a 96 px fine window is `LOWRES`.
- **Flags:** constructed tracks trigger `LOST`, `JUMP`, `SIZE`, `CONTACT`, `EDGE`, `MULTI`, `LOWRES` and `ORIENT` exactly where expected.
- **Grid snapping:** an off-grid start frame snaps to the grid.
- **Schema:** CSV column names and order exactly as in §8; dtypes; README.txt generated and containing every column.
- **Session:** `session.json` round trip is lossless; an unknown `schema_version` gives a clear error.

### 13.2 Synthetic videos and the stand-in segmenters
`synthetic.py` writes H.264 test clips with libx264 and GOP 24 with B-frames, like `_tracker.mp4`. Each clip comes with a ground-truth table. Contents:
- **Shrimp-like objects:** an ellipse body plus two antennae drawn as thick segments attached near the front. Each antenna's angle from the body axis is $\beta(t) = \beta_0 + B\sin 2\pi f t$. Use $\beta_0 = 45^\circ$, $B = 30^\circ$: the sweep never crosses the sideways position, so the hull area oscillates at $f$, not $2f$.
- **Motion:** straight and curved paths with known speed.
- **A contact event:** two objects passing within 1 px.
- **An object near the dish wall.**
- **An LED rectangle** that switches on at a known frame.
- **Two scales:** dish scale (32.4 µm/px, body ≈ 14 px) and close-up scale (10 µm/px, body ≈ 47 px, antennae ≈ 3 px wide).

The ground truth includes, per frame, the true solidity of the rendered shape.

`ExactFake` returns the ground-truth masks in whatever crop the runner used, with signed-distance logits ($\Lambda$ = signed Euclidean distance to the mask boundary, positive inside). `ThresholdFake` segments like the template's `DiskFinder`.

Integration tests through the full pipeline, coarse and fine, with the dish crop on and off:
- with `ExactFake`, positions equal the centroids of the ground-truth masks within 0.01 px. `ThresholdFake` is tested on disks only, within 0.25 px of the true centers (the template's tolerance);
- descriptors fall within the §13.1 tolerances;
- `CONTACT` is flagged at the contact frames;
- the `probes.csv` onset is at the right frame exactly;
- `overlay.mp4` decodes with one frame per tracked frame;
- "Re-track from here" changes only frames ≥ k of the selected tracks;
- "End track" and "Continue as new track" produce `A` and `A2`;
- `export` after changing the stick length from 30 to 29 mm scales every x and y by exactly 29/30, without re-tracking;
- `edgetam/*.csv` parse with the vendored `segment.read_tracker_export` and give the same x, y as `positions.csv`;
- the frame-hash guard aborts when the decoded start frame differs from the prompted one.

### 13.3 Regression against last week's script (slow; real model)
Vendor unmodified copies of the template's `__init__.py`, `segment.py` and `_edgetam.py` as the package `tests/reference/shrimp/`. Tests that need them put `tests/reference` on `sys.path`; the package never imports them.

On the selftest clip, with the same single positive click, frames and device (`cpu`, for determinism), coarse mode with the dish crop off must reproduce `segment.track_video`'s pixel positions within 0.01 px.

### 13.4 Real-model tests (marked `slow`; not in CI)
These run on short synthetic clips only; each takes minutes, not hours. Run them with `device=cpu`, which is deterministic and is what laptops without an Apple GPU use. On J's Mac, also run `selftest` and the fine-mode test once with `device=mps`. Mac students will use that path, and it exercises the `step()` fallback (§6.2); last week's code was never tested on an Apple GPU.
- **`selftest`:** max error < 3 px, as before.
- **Negative prompt:** two touching ellipses, a positive click on one and a negative click on the other. The preview mask excludes the neighbor (overlap with it < 5% of its area).
- **Fine mode** on the synthetic close-up shrimp (9 Hz beat, 240 fps, step 1, 2 s):
  - the solidity spectrum peaks within ±0.5 Hz of 9 Hz;
  - the RMS difference between measured and true solidity is < 0.02;
  - `shape_ok` = 1.
- **Coarse mode** on the dish-scale version flags `LOWRES`.

Report the numbers in docs/VALIDATION.md. If a real-model test fails, report it with numbers and images; do not tune it until it passes.

### 13.5 GUI tests (pytest-qt, `QT_QPA_PLATFORM=offscreen`)
- **P0 smoke test:**
  - the main window opens and a synthetic clip loads;
  - calibration, circle and origin are set through the widgets' public slots;
  - one object is added by a simulated click;
  - a run with `ExactFake` completes, Export writes the §8 P0 files, and the session reopens.
- **P1:** the full flow (jump to a `CONTACT` flag, re-track from there). `scripts/screenshots.py` drives the flow and saves a PNG of each step to `docs/screenshots/`, for J's how-to slides.

### 13.6 Frame exactness
On a synthetic GOP-24 H.264 clip with B-frames: for 20 random k, the GUI's random-access frame k equals, bit for bit, the k-th frame of the sequential decode used for tracking.

### 13.7 CI
GitHub Actions:
- **ubuntu-latest**, Python 3.12: `uv sync`, then `QT_QPA_PLATFORM=offscreen uv run pytest -m "not slow"`. CPU-only torch on Linux keeps CI small, e.g. a uv index for `https://download.pytorch.org/whl/cpu` with a `sys_platform == 'linux'` marker. Fast tests must not need torch.
- **windows-latest**, Python 3.12, with the pinned torch and PySide6-Essentials:
  - `uv run python -c "import torch; import PySide6.QtWidgets"` succeeds;
  - `uv run outline-tracker --version` succeeds;
  - the fast tests pass.

---

## 14. Thursday plan, priorities, acceptance

### 14.1 Timeline (Pacific)

| when | who | what |
|---|---|---|
| Tue evening | J | on the Mac, put SPEC.md in an empty folder outside synced folders (not in iCloud's Desktop/Documents or Proton Drive). Check that `uv`, `git` and an authenticated `gh` work. Start Claude Code there with the kickoff prompt, answer its questions, approve the plan (~30–45 min). Before leaving it running: allow file edits and `uv`/`git` commands without prompts, and keep the Mac awake (plugged in, lid open, Claude Code started as `caffeinate -i claude`). |
| Tue night → Wed late morning | Claude Code | creates the private repo; P0 core: ports, geometry, measure/derive, results store, export, `from-tracker`, `probe`, tests (slow tests on synthetic clips allowed) |
| **Wed ~noon: go/no-go 1** | J | `from-tracker` with `--seconds 2` on a real day-1 clip with last week's `extra/start.csv`; compare with last week's `edgetam/` CSVs; look at the overlay (~45 min) |
| Wed afternoon → evening | Claude Code | P0 GUI, then P1 |
| **Wed evening: go/no-go 2** | J | the §14.3 checklist on the Mac (~1 h), then the install, selftest and a short GUI track on J's Windows machine (~20 min; the only real test of §10.2) |
| Thu morning | J + Claude Code | fixes; README quickstart; pin versions and tag `v0.1.0`; J makes the repo public |
| Thu 1 pm | students | install, selftest, track |

If go/no-go 2 fails, students use `outline-tracker from-tracker` on Thursday: the same outputs, with Tracker for calibration and clicks as last week.

### 14.2 Priorities

**P0 (must ship Thursday):**
- **Core:** the ported core and its tests; geometry (fps_true, stick, tape check, circle, axes, grid snapping); coarse mode with dish crop; fine mode; multi-point and negative prompts; mps fallback in `step()`; the frame-hash guard.
- **Outputs:** `results.npz`; export of positions.csv, the Tracker-format folder, shapes.csv, radial.csv, outlines.npz, overlay.mp4 (P0 content), run.log, session.json and README.txt.
- **Probes** in the core and CLI.
- **Commands:** `from-tracker`, `export`, `probe`, `selftest`, `convert`, `check`.
- **GUI:**
  - video view with zoom, the grid slider and a cursor readout;
  - click-to-place tools with Redo;
  - objects with positive, negative and head prompts, and preview;
  - background runs with up-front estimate, progress, cancel and autosave;
  - flags table with previous/next; re-track from here; end track; new piece;
  - export; continuous session save and open;
  - the Windows import order.
- **Docs and CI:** install instructions; the GUI smoke test; CI on ubuntu and windows.

**P1 (if on time):**
- the GUI probe tool;
- live view;
- draggable handles;
- the flag timeline strip;
- overlay extras (trail, head tick, circle, axes, scale bar, zoom overlays);
- calibration import/export;
- resume after cancel;
- the `run` command;
- the GUI Convert button;
- the full GUI flow test and the screenshots script;
- `selftest --fine`;
- docs/VALIDATION.md.

**P2 (after Thursday):**
- backward tracking; box prompts; undo beyond prompts;
- image-sequence input (microscope TIFF stacks for final projects);
- `.trk` import;
- a probe plot in the GUI;
- batch multi-video UI;
- macOS CI;
- a standalone installer.

**Cut order if Wednesday runs late:**
1. Cut P1 items in the order listed.
2. Then, within P0: `radial.csv` (the template can compute it from `outlines.npz`), circularity and Feret, and the cursor readout.

Never cut: calibration, axes and circle; prompts with preview; coarse and fine tracking; flags with re-track, end and new piece; positions, Tracker-format, shapes and outlines exports; the overlay; `run.log`; the session; the Windows import order.

### 14.3 Acceptance checklist (J, go/no-go 2)
1. `outline-tracker selftest` prints OK on J's Mac (default device, i.e. the Apple GPU, and again with `--device cpu`) and on J's Windows machine. On Windows, the GUI also opens, loads the model and tracks 2 s in coarse mode (this checks the import order of §10.2). Items 2–10 run on the Mac.
2. Open a real day-1 `_tracker.mp4`. fps_true is read from the manifest. A 30 mm stick passes a 20 mm tape check within 1%. A 6-point circle fit sets the origin to its center.
3. Click 3 shrimp on one frame. Separating two touching shrimp with one negative click works, and the preview is correct.
4. Track 2 s at step 2 in coarse mode. The estimate, progress and ETA show, the UI stays responsive, and Cancel keeps what was done. Flags appear in the table.
5. At a flagged frame, re-click one shrimp and re-track from there. End another track and continue it as A2.
6. Export:
   - every P0 file of §8 exists;
   - the `edgetam/*.csv` files load with `load_tracks` as implemented in a group repo;
   - positions agree with last week's `edgetam/` output for the same shrimp and click, with an RMS difference of about 2 px or less. The dish crop changes the model's input, so small differences are expected; with the crop off they must match (§13.3).
7. Fine mode on a close-up clip (or the synthetic one): at 1:1 zoom in the GUI, the outline follows the antennae; `shape_ok` = 1; `radial.csv` shows the lobes.
8. Change the stick length and export without re-tracking: x and y rescale exactly.
9. `overlay.mp4` plays in PowerPoint or Google Slides.
10. Close and reopen the session: everything is restored.

---

## 15. Install and distribution

- **pyproject:**
  - Python ≥ 3.11.
  - Dependencies: `numpy`, `scipy`, `pandas`, `opencv-python-headless` (not `opencv-python`: its bundled Qt conflicts with PySide6), `imageio-ffmpeg>=0.6`, `scikit-image`, `PySide6-Essentials`, `pyqtgraph`, `torch`, `transformers==5.18.0`, `timm`, `huggingface_hub`, `safetensors`; `av` only if §3.5 uses PyAV.
  - Dev: `pytest`, `pytest-qt`.
  - **Before tagging v0.1.0, pin exact versions (`==`)** of torch, transformers, timm, numpy, opencv-python-headless, scikit-image, PySide6-Essentials, pyqtgraph and imageio-ffmpeg: the versions in uv.lock that passed the tests and the Windows import check. `uv tool install` resolves from pyproject and ignores both the lockfile and `.python-version`.
  - Entry point: `outline-tracker = outline_tracker.cli:main`. Version `0.1.0`.
- **Student install** (uv and git are already on their laptops from last week):
  ```
  uv tool install --python 3.12 git+https://github.com/jd-anabi/outline-tracker@v0.1.0
  outline-tracker selftest
  outline-tracker
  ```
  Updates install a newer tag with `uv tool install --force --python 3.12 git+…@vX.Y.Z`.
- **Model cache:** use `~/.cache/shrimp-models/edgetam` and `$SHRIMP_MODEL_CACHE` exactly as `_edgetam.py` does, so students who ran last week's selftest neither download nor convert again.
- **Pitfalls to test:**
  - paths with spaces and non-ASCII characters, on both platforms;
  - cloud-synced folders (atomic writes, lock retries);
  - HiDPI scaling on Windows;
  - negative clicks on a Mac trackpad: two-finger click, Control-click (`MetaModifier` in Qt on macOS) and Option-click (§5);
  - Apple GPU failure falling back to CPU, in `start()` and `step()`;
  - the torch-before-Qt import order.
- **Platforms:** Intel Macs are not supported (no current PyTorch); say so in the README.
- **License:** add a LICENSE (Apache-2.0 suggested: it matches EdgeTAM and transformers, whose conversion code `_edgetam.py` adapts, with attribution). J's choice.

---

## 16. Documentation

- **README.md (for students):**
  - install;
  - a 10-step quickstart matching the GUI's numbered panels;
  - outputs (link to docs/OUTPUTS.md);
  - the three things that go wrong: identity swaps (`CONTACT`, re-track), low resolution (`LOWRES`, fine mode, closer footage), calibration (tape check);
  - troubleshooting;
  - how to cite SAM 2 and EdgeTAM.

  Plain language: most students are new to programming.
- **docs/OUTPUTS.md:** the same content as README.txt; the contract for the analysis template.
- **docs/DEVELOPER.md:** architecture, tests, adding a backend.
- **docs/PLAN.md:** the plan, with checkboxes and a "Questions for J" section.
- **docs/VALIDATION.md:** selftest and synthetic results with numbers (P1).
- **docs/screenshots/:** from the screenshots script (P1).
- **CHANGELOG.md.**

---

## 17. Risks

| risk | mitigation |
|---|---|
| Two days is short | core, `from-tracker` and `probe` first; GUI niceties in P1; cut lines (§14.2); CLI fallback |
| Overnight work stalls on prompts or disputed tests | J allows edits and `uv`/`git` before leaving; disputed tests become strict xfail with a question in PLAN.md; slow tests on synthetic clips allowed |
| torch import fails after Qt on Windows (invisible on the Mac where it is built) | torch imported first (§10.2); Windows CI job; a GUI run on J's Windows machine at go/no-go 2 |
| Apple GPU (mps) fails mid-run | `step()` fallback to cpu (§6.2); exercised on J's Mac (§13.4) |
| Qt or torch install fails on some laptops | `from-tracker` fallback; selftest before class; pinned versions; Intel Macs excluded up front |
| Fine mode untested on real close-ups | synthetic close-up tests; a short real close-up clip as early as possible, filmed in slow motion (check that the close-up setting still records 240 fps) |
| Fine mode multiplies run time | 1–3 fine objects per video; estimate shown before the run |
| Identity swaps (seen last week) | `CONTACT`, `JUMP` and `SIZE` flags; re-track; end and new piece |
| Display and tracked frame differ | §13.6 and the frame-hash guard |
| transformers API drift | pinned 5.18.0, as tested last week |
| Antennae corrupt the body axis | core mask (§7.3); `ORIENT` flags; head click |
| Low resolution mistaken for shape | `shape_ok` from both camera pixels and grid cells; `LOWRES` |
| Large files in git | only `outlines.npz` (fine tracks) is whitelisted; `results.npz` and mp4 stay out |

---

## Appendix A. CLAUDE.md for the new repo (create this file first)

```markdown
# outline-tracker: rules for the coding agent

- The spec is SPEC.md. The plan, with checkboxes and a "Questions for J" section, is docs/PLAN.md: update it as tasks finish.
- Deadline: students install Thursday Oct 8, 1 pm Pacific. Work in the order of SPEC §14: core, `from-tracker` and `probe` before the GUI; P0 before P1.
- You will often work unattended. Don't stop for routine steps (uv, pytest, git add/commit/push, editing files in this repo). Push after each task.
- Test first. Expected values come from geometry, analytic shapes or synthetic ground truth, never from running the code and copying its output. Never weaken, skip or delete a test to make it pass. If a test seems wrong, mark it xfail(strict=True, reason=...), add it under "Questions for J" in docs/PLAN.md, and continue with other work.
- Run `uv run pytest -m "not slow"` after every change and report the result. Slow tests (real model, short synthetic clips only) are fine to run without asking once the HF segmenter is ported. Never run the model on long real videos yourself.
- Conventions (SPEC §3): Tracker's pixel convention (pixel centers at +0.5), mm, the session's origin and axes, y up, t_s = frame / fps_true, the frame grid. Units go in column names.
- Ported code keeps its behavior. You may copy from jd-anabi/shrimp-tracker-template only these provided files: src/shrimp/__init__.py, segment.py, _edgetam.py, video.py, convert.py, check_video.py and tests/conftest.py, test_segment.py, test_video.py, test_convert.py. tests/reference/ holds unmodified copies for regression tests and is never imported by the package.
- Nothing outside outline_tracker/gui imports Qt. torch and transformers are imported only inside outline_tracker/segmenter, lazily, except that the GUI entry point imports torch before PySide6 (Windows DLL issue).
- Never load a whole video into memory. Never keep full-frame float arrays per object beyond the current frame.
- Ask (in PLAN.md) before adding a dependency not listed in SPEC §15.
- Never commit videos, results.npz or model weights. Never force-push or rewrite history.
- This repo will be public. Never put answer keys, student data, rosters or personal paths in it.
- Finish every task by saying what changed, how it was verified, and what is uncertain.
```

---

## Appendix B. Notes on shape analysis (for the analysis template and the slides)

The app outputs geometry; the analysis decides what "shape" means. Three levels, from simple to ambitious:

1. **One-number stroke signals.** Solidity $S(t) = A/A_\text{hull}$: spread antennae inflate the convex hull, so $S$ drops when they are out.
   - **Caution:** when the hull area is nearly symmetric about the antennae's sideways position, $S$ is largest twice per cycle. It then carries a strong $2f$ harmonic, or oscillates at $2f$ outright. Whether that happens depends on where the antennae attach and how far they sweep. In a rendered test, antennae attached mid-body sweeping 30°–150° gave a pure $2f$ signal; attached near the front, the $2f$ amplitude was 0.8 of the $f$ amplitude. Confirm the frequency with a phase-resolving signal (level 2), or with the antenna angle read from the radial profile's lobes.
   - $S$'s cross-correlation with the swimming speed gives the timing of stroke against thrust.
   - Eccentricity and $L_1$ of the full mask respond too, less selectively.
2. **Shape modes ("eigenshapes").**
   - Stack the radial profiles $\mathbf r_t = (r(\phi_0,t),\ldots,r(\phi_{71},t))$ and subtract the mean $\bar{\mathbf r}$.
   - Form $C = \frac1T\sum_t(\mathbf r_t-\bar{\mathbf r})(\mathbf r_t-\bar{\mathbf r})^\top$. Its eigenvectors $\mathbf u_k$ are the shape modes, and $a_k(t) = \mathbf u_k^\top(\mathbf r_t - \bar{\mathbf r})$ their amplitudes.
   - A periodic stroke traces a closed loop in the $(a_1, a_2)$ plane. The phase is $\varphi(t) = \operatorname{atan2}(a_2, a_1)$ and the stroke frequency $f = \langle\dot\varphi\rangle/2\pi$; the phase distinguishes power and recovery strokes, which the solidity cannot.

   This is the approach of Stephens et al. (2008) for *C. elegans* postures.
3. **Elliptic Fourier descriptors** (Kuhl & Giardina 1982), computed from `outlines.npz`, for a representation that does not assume a star-shaped outline.

Caveats to teach:
- The official position (full-mask centroid) wobbles at the stroke frequency (≈ 2 px in last week's synthetic test). The core centroid `core_x_mm, core_y_mm` is the steadier body center, and the body frame uses it.
- Exclude frames flagged `LOWRES`, `ORIENT`, `CONTACT` or `MULTI` from shape statistics.
- Consecutive frames are not independent samples.

---

## Appendix C. References

- Ravi, N. et al. (2024). SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714.
- Zhou, C. et al. (2025). EdgeTAM: On-Device Track Anything Model. CVPR 2025; arXiv:2501.07256. Code and weights under Apache 2.0: github.com/facebookresearch/EdgeTAM.
- Kuhl, F. P. & Giardina, C. R. (1982). Elliptic Fourier features of a closed contour. *Computer Graphics and Image Processing* 18, 236–258.
- Stephens, G. J., Johnson-Kerner, B., Bialek, W. & Ryu, W. S. (2008). Dimensionality and dynamics in the behavior of *C. elegans*. *PLoS Computational Biology* 4(4), e1000028.
- Tracker, Open Source Physics: physlets.org/tracker.
- Student template (source of the ported code): github.com/jd-anabi/shrimp-tracker-template.
