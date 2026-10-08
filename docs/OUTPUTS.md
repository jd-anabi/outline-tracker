<!-- Generated from outline_tracker/schema.py by "uv run python -m outline_tracker.schema_docs docs/OUTPUTS.md". Do not edit by hand. -->

# Outline Tracker: the files of a run

This page describes every file that Outline Tracker writes for one run on one video: its columns, their units, and what the numbers mean. It is the contract between the tracker and the analysis template, which may rely on the names, the order and the formats written here.

It is generated from the schema of the tool, and a test fails when it is out of date, so do not edit it by hand. Every run folder holds the same information as plain text in `README.txt`.

## Words used on this page

- **Tracker**: the video-analysis program you used last week (physlets.org/tracker); this tool follows its pixel and file conventions.
- **frame**: one picture of the video, counted from 0 as Tracker counts.
- **track**: one animal followed through the video. Its id is a capital letter such as `A`; later pieces of the same animal are `A2`, `A3`.
- **mask**: the pixels of one frame that the model says belong to the animal.
- **centroid**: the average position of all pixels of the mask, like its balance point. It is the official position of the animal.
- **outline**: the line around the mask, kept as 128 points per frame for fine tracks.
- **coarse and fine**: in coarse mode the model looks at the whole frame (or the dish); in fine mode it looks at a small crop that follows the animal, which gives a sharper outline.
- **core**: the mask with thin parts such as antennae cut away; its centroid is the body center.
- **heading**: the direction from the body center to the head, as an angle in your axes.
- **empty cell**: how a CSV file says that a number does not exist (for example, the animal is lost on that frame). pandas reads it as `NaN`, short for not a number.

## Conventions

- Units: `_s` seconds, `_mm` millimetres, `_mm2` square millimetres, `_rad` radians, `_px` image pixels. Not every number has its unit in its column name: ratios (`eccentricity`, `core_frac`, `solidity`, `circularity`, `largest_fraction`) have no unit, and these columns have a unit that their names do not show: in `<model>/<id>.csv`, `t` is in seconds, `x` and `y` are in millimetres and `pixelx` and `pixely` are in image pixels; in `shapes.csv`, `px_along_major` is in image pixels and `cells_along_major` is in model grid cells; in `radial.csv`, `r_000`, `r_005`, ..., `r_355` (72 columns) are in millimetres; in `probes.csv`, `r`, `g`, `b` and `gray` are in the 0-255 range of image brightness. The description of every column below gives its unit.
- Image coordinates (`u_px`, `v_px`; Tracker's `pixelx`, `pixely`): the origin is the top-left corner of the frame, u grows to the right, v downward, and the pixel in column c and row r has its center at `(c + 0.5, r + 0.5)`.
- Your axes (`x_mm`, `y_mm`): the origin and the direction of +x were set in the calibration, and y points up. With k the scale in mm per pixel, alpha the angle of +x counterclockwise on screen from the image's rightward direction, and (u0, v0) the origin: `x = k ((u - u0) cos alpha - (v - v0) sin alpha)` and `y = -k ((u - u0) sin alpha + (v - v0) cos alpha)`. Angles in your axes are counterclockwise.
- Time: `t_s = frame / fps_true`, the frame rate you gave or measured; the frame rate stored in the video file is never used. Start frame and step choose frames but never shift `t_s`.
- Frames: counted from 0 in a sequential decode of the file, as Tracker counts. The tracked frames are the grid `start + n * step`, so all tracks share frames.
- Rows: for each track, every grid frame from its first to its last, sorted by `track_id` (text order) and `frame`. A lost frame keeps its row, with its `track_id`, `frame` and `t_s`: the measured numbers are empty cells, except `visible = 0`, `n_components = 0` and `shape_ok = 0` (integers stay integers), and `flags` holds `LOST`.
- Numbers: written with 7 decimals for seconds (`_s`), 6 decimals for `_mm`, `_mm2`, `_rad` and ratios, and 3 decimals for `_px`, pixel and cell counts and probe means. A number that does not exist is an empty cell, never the text `nan`.
- Text files: UTF-8 with LF line ends on every platform. The files in the model folder keep your computer's line ends, as last week.

## The files

The run folder holds these files, in this order. `<model>` is the name of the model, for example `edgetam`, and `<id>` is the track id. The type `str` is text, `int` a whole number and `float` a number with decimals.

### `session.json`

- **What it holds:** everything needed to reopen the run, run it again and export again: settings, calibration, clicks, run history.
- **In git:** yes.

### `positions.csv`

- **What it holds:** one row per track and tracked frame: where the object is.
- **Rows:** one row per track and grid frame, from the track's first frame to its last.
- **In git:** yes.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `track_id` | str | none | track name: A, B, ...; later pieces of the same animal are A2, A3, ... |
| `frame` | int | none | video frame number, counted from 0 as Tracker counts |
| `t_s` | float, 7 decimals | s | time, frame / fps_true (frame 0 is 0 s) |
| `x_mm` | float, 6 decimals | mm | area centroid of the full mask in your axes; empty if the track is not visible |
| `y_mm` | float, 6 decimals | mm | the same point, y pointing up |
| `u_px` | float, 3 decimals | px | the same point in image coordinates (Tracker's pixelx) |
| `v_px` | float, 3 decimals | px | the same point in image coordinates (Tracker's pixely) |
| `area_mm2` | float, 6 decimals | mm^2 | mask area |
| `visible` | int | none | 1 if the mask is non-empty, else 0 |
| `mode` | str | none | coarse (the model saw the whole frame or dish) or fine (a crop that follows the object) |
| `flags` | str | none | quality codes, separated by ';' (see the quality flags); empty if none |

### `<model>/<id>.csv`

- **What it holds:** one file per track in the layout of last week's Tracker files, in a folder named after the model, e.g. edgetam/A.csv; the folder holds nothing else.
- **Rows:** first line ",A,,,,," (a comma, the track name, then commas), then the column names, then one row per tracked frame; written exactly as last week's files were.
- **In git:** yes.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `t` | float, 7 decimals | s | time, frame / fps_true |
| `frame` | int | none | video frame number |
| `x` | float, 6 decimals | mm | x in your axes; empty where the track is lost |
| `y` | float, 6 decimals | mm | y in your axes, y pointing up; empty where the track is lost |
| `pixelx` | float, 3 decimals | px | image coordinate u; empty where the track is lost |
| `pixely` | float, 3 decimals | px | image coordinate v; empty where the track is lost |

### `shapes.csv`

- **What it holds:** one row per track and tracked frame: size, axes, heading, solidity, resolution, wall distance.
- **Rows:** the same rows as `positions.csv`.
- **In git:** yes.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `track_id` | str | none | track name: A, B, ...; later pieces of the same animal are A2, A3, ... |
| `frame` | int | none | video frame number, counted from 0 as Tracker counts |
| `t_s` | float, 7 decimals | s | time, frame / fps_true (frame 0 is 0 s) |
| `area_mm2` | float, 6 decimals | mm^2 | mask area |
| `perimeter_mm` | float, 6 decimals | mm | length of the outline polygon |
| `major_mm` | float, 6 decimals | mm | major axis L1 = 4 sqrt(lambda1) of the full mask |
| `minor_mm` | float, 6 decimals | mm | minor axis L2 = 4 sqrt(lambda2) of the full mask |
| `eccentricity` | float, 6 decimals | none | sqrt(1 - lambda2 / lambda1) of the full mask |
| `theta_rad` | float, 6 decimals | rad | heading: direction from the body center to the head, counterclockwise from +x in your axes, unwrapped so that it is continuous (it may leave -pi..pi) |
| `core_x_mm` | float, 6 decimals | mm | x of the core's centroid: the body without thin appendages, the 'body center' |
| `core_y_mm` | float, 6 decimals | mm | y of the core's centroid |
| `core_frac` | float, 6 decimals | none | area of the core / area of the full mask (the core is the body without thin parts such as antennae, found by a morphological opening); if the core fallback was used (the opening removed more than half the area, so the full mask serves as the core) it is still the fraction the opening kept: below 0.5, not 1 |
| `solidity` | float, 6 decimals | none | outline polygon area / convex hull area |
| `circularity` | float, 6 decimals | none | 4 pi polygon area / perimeter^2 (1 for a circle) |
| `feret_max_mm` | float, 6 decimals | mm | largest distance between two points of the outline |
| `n_components` | int | none | number of connected pieces of the mask (0 if the track is lost) |
| `largest_fraction` | float, 6 decimals | none | area of the largest piece / area of the mask |
| `px_along_major` | float, 3 decimals | px | camera pixels along the major axis of the largest piece of the mask (L1 / k) |
| `cells_along_major` | float, 3 decimals | cells | model grid cells along the major axis, (L1 / k) / cell, with cell = max(width, height) / 256 px of the image the model saw |
| `shape_ok` | int | none | 1 if min(px_along_major, cells_along_major) >= 20 (setting shape_ok_min), else 0 |
| `wall_dist_centroid_mm` | float, 6 decimals | mm | dish radius minus the centroid's distance from the dish center; empty without a dish circle; negative = outside the circle |
| `wall_dist_min_mm` | float, 6 decimals | mm | dish radius minus the outline's largest distance from the dish center (how close the outline comes to the wall); empty without a dish circle |
| `flags` | str | none | quality codes, separated by ';' (see the quality flags); the same text as in positions.csv |

### `radial.csv`

- **What it holds:** the outline's radius at 72 angles around the body center, for fine tracks.
- **Rows:** one row per fine track and frame (the same rows as `positions.csv` for those tracks).
- **In git:** yes.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `track_id` | str | none | track name: A, B, ...; later pieces of the same animal are A2, A3, ... |
| `frame` | int | none | video frame number, counted from 0 as Tracker counts |
| `t_s` | float, 7 decimals | s | time, frame / fps_true (frame 0 is 0 s) |
| `r_000`, `r_005`, ..., `r_355` (72 columns) | float, 6 decimals | mm | radius at the angle ddd degrees of the column name, counterclockwise from the head direction: the distance from the body center to the farthest crossing of that ray with the outline; r_000 is the head radius; empty when the track is not visible |

The 72 radius columns, in order: `r_000`, `r_005`, `r_010`, `r_015`, `r_020`, `r_025`, `r_030`, `r_035`, `r_040`, `r_045`, `r_050`, `r_055`, `r_060`, `r_065`, `r_070`, `r_075`, `r_080`, `r_085`, `r_090`, `r_095`, `r_100`, `r_105`, `r_110`, `r_115`, `r_120`, `r_125`, `r_130`, `r_135`, `r_140`, `r_145`, `r_150`, `r_155`, `r_160`, `r_165`, `r_170`, `r_175`, `r_180`, `r_185`, `r_190`, `r_195`, `r_200`, `r_205`, `r_210`, `r_215`, `r_220`, `r_225`, `r_230`, `r_235`, `r_240`, `r_245`, `r_250`, `r_255`, `r_260`, `r_265`, `r_270`, `r_275`, `r_280`, `r_285`, `r_290`, `r_295`, `r_300`, `r_305`, `r_310`, `r_315`, `r_320`, `r_325`, `r_330`, `r_335`, `r_340`, `r_345`, `r_350`, `r_355`

`radial.csv` and `outlines.npz` hold the fine tracks by default: the outline of a coarse track is a blob of a few grid cells (flag `LOWRES`). The setting `shape_files_for_coarse` (off by default) adds the coarse tracks. A fine track comes from `--fine` on the command line, or from the object's fine mode in the app. With no fine track, `radial.csv` holds only its header line and `outlines.npz` only the key `meta`, so both files can always be read.

The radial profile `r_ddd` is the distance from the body center to the farthest crossing of the ray at ddd degrees (counterclockwise from the head direction) with the outline. For an outline that is star-shaped around the body center this is the boundary itself; otherwise it is the outer envelope. `r_000` is the head radius.

### `outlines.npz`

- **What it holds:** the outlines themselves (128 points per frame), in your axes and in the body frame, for fine tracks.
- **In git:** yes.

`outlines.npz` has these arrays for every track, named `<id>__<name>`, for example `A__xy_mm` (`n` is the number of tracked frames, `N` the number of points):

| key | dtype | shape | unit | meaning |
| --- | --- | --- | --- | --- |
| `<id>__frames` | int32 | [n] | none | video frame numbers |
| `<id>__xy_mm` | float32 | [n, N, 2] | mm | outline points in your axes, counterclockwise, starting at the head point; NaN for frames where the track is not visible |
| `<id>__xieta_mm` | float32 | [n, N, 2] | mm | the same points in the body frame: xi toward the head, eta 90 degrees counterclockwise from xi |

Every outline has `N` = 128 points. Which tracks the file holds is explained under `radial.csv`.

`meta` is a JSON text with these fields:

- `n_points`: N, the number of points of every outline
- `tracks`: the track ids that have outlines in the file (empty when there is no fine track)
- `conventions`: a short statement of the coordinate conventions
- `tool_version`: the Outline Tracker version that wrote the file

### `probes.csv`

- **What it holds:** brightness of the probe rectangles (the LED) on every frame of the clip; written only if probes exist.
- **Rows:** one row per probe and frame, for every frame of the clip (step 1, whatever the tracking step).
- **In git:** yes.

| column | type | unit | meaning |
| --- | --- | --- | --- |
| `frame` | int | none | video frame number, counted from 0 as Tracker counts |
| `t_s` | float, 7 decimals | s | time, frame / fps_true (frame 0 is 0 s) |
| `probe` | str | none | name of the probe rectangle (default LED1) |
| `r` | float, 3 decimals | 0-255 | mean red inside the rectangle, on the 0-255 scale |
| `g` | float, 3 decimals | 0-255 | mean green inside the rectangle, on the 0-255 scale |
| `b` | float, 3 decimals | 0-255 | mean blue inside the rectangle, on the 0-255 scale |
| `gray` | float, 3 decimals | 0-255 | 0.299 r + 0.587 g + 0.114 b of the three means |

### `overlay.mp4`

- **What it holds:** the clip with each outline, centroid and id drawn on it; plays in PowerPoint and Google Slides.
- **In git:** no (large; share it through Drive).

### `results.npz`

- **What it holds:** internal pixel-space store: every other file is rebuilt from it and session.json; not meant for analysis.
- **In git:** no (large; share it through Drive).

The arrays of each track are stored under the keys `<id>__<name>`; `n` is the number of tracked frames of the track, and `U6` is text of up to 6 characters. The key `version` holds the format version, a whole number; it is currently 1.

| key | dtype | shape | unit | meaning |
| --- | --- | --- | --- | --- |
| `<id>__frames` | int32 | [n] | none | video frame numbers of the records, ascending, on the clip's frame grid |
| `<id>__visible` | bool | [n] | none | the mask is non-empty |
| `<id>__area_px` | int32 | [n] | px | number of pixels of the mask; 0 if not visible |
| `<id>__u` | float64 | [n] | px | area centroid of the mask, image u; NaN if not visible |
| `<id>__v` | float64 | [n] | px | area centroid of the mask, image v; NaN if not visible |
| `<id>__cov_full` | float64 | [n, 3] | px^2 | covariance of the pixel centers of the full mask: mu_uu, mu_uv, mu_vv |
| `<id>__cov_core` | float64 | [n, 3] | px^2 | the same for the core mask |
| `<id>__core_u` | float64 | [n] | px | core centroid, image u |
| `<id>__core_v` | float64 | [n] | px | core centroid, image v |
| `<id>__core_frac` | float64 | [n] | none | area of the core / area of the mask; with the core fallback it is the fraction the opening kept, below 0.5, not 1 |
| `<id>__core_fallback` | bool | [n] | none | the opening removed more than half the area, so the full mask was used as the core |
| `<id>__core_r_px` | int32 | [n] | px | radius of the opening disk that made the core; 0 if not visible |
| `<id>__outline_px` | float32 | [n, 256, 2] | px | outline resampled to 256 points in image coordinates (u, v); NaN if not visible |
| `<id>__n_components` | int32 | [n] | none | number of connected pieces of the mask |
| `<id>__largest_fraction` | float64 | [n] | none | area of the largest piece / area of the mask |
| `<id>__second_fraction` | float64 | [n] | none | area of the second-largest piece / area of the largest; 0 with one piece |
| `<id>__cell_px` | float64 | [n] | px | size of one model grid cell in image pixels: max(width, height) / 256 of the image the model saw |
| `<id>__edge` | bool | [n] | none | the mask touches the border of the model's input |
| `<id>__mode` | U6 | [n] | none | coarse or fine |
| `<id>__score` | float32 | [n] | none | the model's object-presence logit; NaN if the backend has none |
| `<id>__mask_offset` | int32 | [n, 2] | px | (column, row) of each mask crop's top-left pixel in the full frame |
| `<id>__mask_shape` | int32 | [n, 2] | px | (rows, columns) of each mask crop |
| `<id>__mask_bits` | uint8 | [total_bytes] | none | all mask crops one after another, each flattened row by row and packed with numpy.packbits; crop i takes ceil(rows * columns / 8) bytes |

### `run.log`

- **What it holds:** what was run: software, video, time and calibration, runs, corrections, quality summary, files written.
- **In git:** yes.

### `README.txt`

- **What it holds:** the same information as this page, as plain text; written into every run folder.
- **In git:** yes.

## Quality flags

The column `flags` of `positions.csv` and `shapes.csv` holds the quality codes of the row, separated by `';'` in the order below (empty if none). It is the same text in both files.

- Flags that concern positions: LOST, JUMP, SIZE, CONTACT, EDGE.
- Flags that concern only shapes: MULTI, LOWRES, ORIENT, HEADGUESS.

| code | concerns | condition | meaning |
| --- | --- | --- | --- |
| `LOST` | position | mask empty | no position; re-click if needed |
| `JUMP` | position | centroid speed between consecutive visible frames above jump_mm_s (default 100 mm/s) | probably jumped to another object |
| `SIZE` | position | area outside 0.5 to 2 times the track's median area | merged with a neighbor, or partly lost |
| `CONTACT` | position | this track's outline comes within max(3 px, 2 grid cells) of another track's outline on the same frame | identities may swap here: check |
| `EDGE` | position | the mask touches the border of the model's input (frame, dish crop or fine crop) | outline may be cut off |
| `MULTI` | shape | more than one component, the second at least 10% of the largest | ragged or split mask |
| `LOWRES` | shape | shape_ok = 0 (fewer than 20 camera pixels or fewer than 20 grid cells along the major axis) | shape columns unreliable; use fine mode or closer footage |
| `ORIENT` | shape | core nearly round (lambda2 / lambda1 > 0.8), heading axis jumped by more than 60 degrees, or the core fallback was used | heading unreliable |
| `HEADGUESS` | shape | head side taken from motion (no head click); on every frame of the track | check the head in the overlay |

Filter on the codes you care about, not on "flags is empty": at dish scale every row of a coarse track carries `LOWRES`, and a track without a head click carries `HEADGUESS` on every frame.

## Reading the files

To read the main files with Python:

```python
import json
import numpy as np
import pandas as pd
df = pd.read_csv("positions.csv")
df["flags"] = df["flags"].fillna("")          # an empty cell is read as NaN
jumps = df[df["flags"].str.contains("JUMP")]
z = np.load("outlines.npz")
xy = z["A__xy_mm"]                            # [n, 128, 2], mm
meta = json.loads(str(z["meta"]))
```

Write `df["flags"]`, not `df.flags`: pandas has its own attribute called flags. Lost frames are empty cells, which pandas reads as NaN; the integer columns (frame, visible, n_components, shape_ok) have no empty cells and stay integers.

## What goes into git

**In git (small files):**

- `session.json`
- `positions.csv`
- `<model>/<id>.csv`
- `shapes.csv`
- `radial.csv`
- `outlines.npz`
- `probes.csv`
- `run.log`
- `README.txt`
- `calibration.json` (if you exported one)

`outlines.npz` is the only full-shape output, so it must be committed. A `.gitignore` that excludes `*.npz` hides it: add the line `!**/outlines.npz` below the line `*.npz`.

**Not in git (large; share them through Drive):**

- `overlay.mp4`
- `results.npz`

## If something went wrong

If a run stopped early, or a file is missing, run this command; it rebuilds every file from what was tracked, without running the model:

```
outline-tracker export <run folder>/session.json
```

If a file is locked (for example a CSV open in Excel), the tool writes the new data next to it as `<name>.new<extension>`, for example `positions.new.csv`, and tells you. Close the program that holds the file and export again.
