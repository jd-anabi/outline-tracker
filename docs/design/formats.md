# File formats after version 0.1.0

This page is the contract of every saved file whose format changes after version 0.1.0 (decision 15 of
[ROADMAP.md](../ROADMAP.md)). Each section is written by hand before the code changes. The owner approves it
first, and the tests then take their expected values from it. The formats of 0.1.0 are in
[SPEC-v0.1.md](SPEC-v0.1.md). [OUTPUTS.md](../OUTPUTS.md), which is generated, describes the files that a run
writes.

## session.json, version 2

W3 step 1 (decisions 12 and 32 to 39). Status: written, and waiting for the owner's approval. The code
still writes version 1.

### What changes

Version 2 changes the format in these ways:

- `schema_version` is 2.
- `student` becomes `label`.
- `circle` becomes `boundary`, and its new first key is `shape` with the value `"circle"`.
- `circle.dish_mm` becomes `boundary.diameter_mm`.
- `processing.dish_crop` becomes `processing.boundary_crop`.
- `time.manifest_path` is dropped.
- A `time.source` of `"manifest"` is read as `"typed"`, and the number in `fps_true` is kept.
- `time.source` has the new value `"file"`: the video file's own frame rate (decision 10).
- `processing.tracker_format` is new. It says whether an export writes the Tracker-format files
  (decision 7).
- `processing.jump_mm_s` may be `null`, which means no `JUMP` flag (decision 13).

Everything else is as in version 1 (SPEC-v0.1.md 8.10): the same keys, the same order, the same units.
The whole format changes in step 1 (decision 32). After step 1, steps 2 to 4 switch the code over one row at
a time: the names inside the code, the window and the texts. The table's last column names the step that
switches each row.

### Reading a file

The build reads a file as follows:

1. **Versions read.**
   - A file with `schema_version` 2 is read as it is.
   - A file with `schema_version` 1 first goes through the version 1 reader. The reader turns the
     parsed version 1 object into a version 2 object, as the table says, and only then are the records
     made. The reader reads only: it writes no file.
   - Any other `schema_version` is refused, and so is a file without one. This includes 3 and
     later. The message says that this build reads versions 1 and 2.
2. **Order.** A renamed key keeps its place in its object. A new key comes last in its object, except
   `boundary.shape`, which comes first. Written files have the order of the examples below; keys that the
   build does not know follow the known keys of their object.
3. **Values.** The reader changes no value that the table does not name. Above all, it keeps
   `tool_version`, which names the version that made the session (rule 1 of the ROADMAP).
4. **Keys missing from a version 1 file** are read as 0.1.0 read them.
   - The reader writes 0.1.0's value for the two keys whose default for new sessions W3 changes:
     - `processing.jump_mm_s`, as `100.0` (decision 13);
     - `probes[].name`, as `"LED1"` (decision 9).
   - It always writes `processing.tracker_format` as `true`. If the file has no `processing` object, the
     reader makes one for these keys.
   - Every other missing key takes the default of version 2. That default is the same as 0.1.0's
     (column "Default" below), and a test holds that the two agree.
5. **Unknown keys** are kept in their object and written back, as in version 1.
6. **Refused, with a message that names the file and the key:**
   - a version 1 object that already holds a key that version 2 gives that object, as an unknown
     key: `label` or `boundary` at the top, `shape` or `diameter_mm` in `circle`, or `boundary_crop` or
     `tracker_format` in `processing`. 0.1.0 never writes one, so the reader cannot tell which value
     counts (decision 39);
   - a `boundary.shape` other than `"circle"`. In a version 2 file a missing `shape` reads as
     `"circle"`.

   A block that is neither an object nor, where the table allows it, `null` is refused as in version 1.
   The message uses the name the file uses: `circle` in a version 1 file.

### Writing over a version 1 file

This follows rule 5 of the ROADMAP and decisions 36 to 39.

- **Opening and closing writes nothing.** A session read as version 1 is written only after a real
  change. Its video having moved is not such a change (decision 37).
- **The first save keeps the old file.** It goes to `session.v1.json` beside the new `session.json`.
  - If `session.v1.json` holds the same bytes already, nothing more is written.
  - If it holds other bytes, the old file goes to the next free name: `session.v1-2.json`,
    `session.v1-3.json`, and so on.
  - A kept file is never overwritten.
  - If the copy fails, `session.json` is not replaced, and the save says why.
  - `Session.save` does this, so it holds for every writer: the window, a tracking run, a correction,
    and `from-tracker` when it replaces an earlier run.
- **The user is told once.**
  - The window shows one message per session, at the save that kept the file. It names the kept
    file and says that version 0.1.0 opens it after it is renamed to `session.json`.
  - A command prints one line (decision 39).
- **What saves first.** Track and Export in the window save first, as in version 1 (decision 38).
  `outline-tracker export` writes no `session.json`.

### The table

Column "Version 1" lists every key that version 1 knows, and column "Version 2" every key that version 2
knows. A path is written with dots, and `[]` stands for every item of a list. A dash means "no such key".
The columns mean:

- **Reading version 1:** what the reader does. Empty means: the same key, with the same value.
- **Default:** the value of a missing key.
  - For a key of version 1, it is 0.1.0's default. Version 2 reads a missing key with the same
    default, except where the reader writes the value itself (column "Reading version 1"). For those
    keys, W3 may change the default for new sessions.
  - For a key that only version 2 has, it is version 2's default.
  - "below" means that a missing block reads as all its keys missing.
  - "left out" marks an optional key, which is written only when it has a value (decision X18).
- **Step:** the W3 step that switches the code over to this row. A dash means nothing to switch.

The file:

| Version 1 | Version 2 | Reading version 1 | Default | Step |
|---|---|---|---|---|
| `format` | `format` | `"outline-tracker-session"`, the same | required | - |
| `schema_version` | `schema_version` | `1` becomes `2` | required | 1 |
| `tool_version` | `tool_version` | kept: the version that made the session | the reading build's version | - |
| `student` | `label` | renamed; the same text | `""` | 3 |
| `notes` | `notes` | | `""` | - |
| `complete` | `complete` | | `true` | - |

The video:

| Version 1 | Version 2 | Reading version 1 | Default | Step |
|---|---|---|---|---|
| `video` | `video` | | below | - |
| `video.relpath` | `video.relpath` | | `null` | - |
| `video.abspath` | `video.abspath` | | `""` | - |
| `video.size` | `video.size` | | `0` | - |
| `video.sha256_first_64MiB` | `video.sha256_first_64MiB` | | `""` | - |
| `video.width` | `video.width` | | `0` | - |
| `video.height` | `video.height` | | `0` | - |
| `video.n_frames` | `video.n_frames` | | `0` | - |
| `video.fps_container` | `video.fps_container` | | `0.0` | - |

The clip and the time:

| Version 1 | Version 2 | Reading version 1 | Default | Step |
|---|---|---|---|---|
| `clip` | `clip` | | below | - |
| `clip.start` | `clip.start` | | `0` | - |
| `clip.end` | `clip.end` | | `0` | - |
| `clip.step` | `clip.step` | | `2` | - |
| `time` | `time` | | below | - |
| `time.fps_true` | `time.fps_true` | | `null` | - |
| `time.source` | `time.source` | `"manifest"` becomes `"typed"`; `"stopwatch"`, `"typed"`, `"tracker-export"` and any other text stay as they are. Version 2 adds `"file"` | `"typed"` | 4 |
| `time.manifest_path` | - | dropped; the kept version 1 file still holds it | `null` | 1 |
| `time.stopwatch` | `time.stopwatch` | the same object (`frame_a`, `time_a_s`, `frame_b`, `time_b_s`) or `null` | `null` | - |

The calibration and the axes:

| Version 1 | Version 2 | Reading version 1 | Default | Step |
|---|---|---|---|---|
| `calibration` | `calibration` | | below | - |
| `calibration.stick` | `calibration.stick` | the same object (`p1_px`, `p2_px`, `length_mm`) or `null` | `null` | - |
| `calibration.click_sigma_px` | `calibration.click_sigma_px` | | `0.5` | - |
| `calibration.check` | `calibration.check` | the same object (`p1_px`, `p2_px`, `true_mm`) or `null` | `null` | - |
| `calibration.tracker_fit` | `calibration.tracker_fit` | the same object (`mm_per_px`, `rms_mm`, `n_points`) | left out | - |
| `axes` | `axes` | | below | - |
| `axes.origin_px` | `axes.origin_px` | | `[0.0, 0.0]` | - |
| `axes.angle_deg` | `axes.angle_deg` | | `0.0` | - |

The boundary:

| Version 1 | Version 2 | Reading version 1 | Default | Step |
|---|---|---|---|---|
| `circle` | `boundary` | renamed; `null` stays `null`; an object gets `shape` as its first key | `null` | 2 |
| - | `boundary.shape` | new: `"circle"`, the only shape (decision 8) | `"circle"` | 2 |
| `circle.points_px` | `boundary.points_px` | | `[]` | 2 |
| `circle.center_px` | `boundary.center_px` | | `[0.0, 0.0]` | 2 |
| `circle.radius_px` | `boundary.radius_px` | | `0.0` | 2 |
| `circle.rms_px` | `boundary.rms_px` | | `0.0` | 2 |
| `circle.dish_mm` | `boundary.diameter_mm` | renamed; the same number or `null` | `null` | 2 |

The processing:

| Version 1 | Version 2 | Reading version 1 | Default | Step |
|---|---|---|---|---|
| `processing` | `processing` | | below | - |
| `processing.model` | `processing.model` | | `"edgetam"` | - |
| `processing.device` | `processing.device` | | `"auto"` | - |
| `processing.dish_crop` | `processing.boundary_crop` | renamed; the same value | `true` | 2 |
| `processing.outline_points` | `processing.outline_points` | | `128` | - |
| `processing.radial_step_deg` | `processing.radial_step_deg` | | `5` | - |
| `processing.shape_ok_min` | `processing.shape_ok_min` | | `20` | - |
| `processing.core_open_frac` | `processing.core_open_frac` | | `0.1` | - |
| `processing.jump_mm_s` | `processing.jump_mm_s` | the same number; written as `100.0` when missing. Version 2 also allows `null`: no `JUMP` flag (decision 13) | `100.0` | 5 |
| `processing.fine_window_factor` | `processing.fine_window_factor` | | `3.0` | - |
| `processing.shape_files_for_coarse` | `processing.shape_files_for_coarse` | | `false` | - |
| - | `processing.tracker_format` | new: always written as `true`, because 0.1.0 always wrote the Tracker-format files. A new session has `false` (decisions 7, 34) | `false` | 4 |

The probes and the tracks:

| Version 1 | Version 2 | Reading version 1 | Default | Step |
|---|---|---|---|---|
| `probes` | `probes` | | `[]` | - |
| `probes[].name` | `probes[].name` | written as `"LED1"` when missing (decision 9) | `"LED1"` | 5 |
| `probes[].rect_px` | `probes[].rect_px` | | `[0, 0, 0, 0]` | - |
| `tracks` | `tracks` | | `[]` | - |
| `tracks[].id` | `tracks[].id` | | `""` | - |
| `tracks[].color` | `tracks[].color` | | `""` | - |
| `tracks[].mode` | `tracks[].mode` | | `"coarse"` | - |
| `tracks[].fine_window_px` | `tracks[].fine_window_px` | | `null` | - |
| `tracks[].start_frame` | `tracks[].start_frame` | | `0` | - |
| `tracks[].head_px` | `tracks[].head_px` | | `null` | - |
| `tracks[].ended_at` | `tracks[].ended_at` | | `null` | - |
| `tracks[].prompts` | `tracks[].prompts` | | `[]` | - |
| `tracks[].prompts[].frame` | `tracks[].prompts[].frame` | | `0` | - |
| `tracks[].prompts[].frame_hash` | `tracks[].prompts[].frame_hash` | | `null` | - |
| `tracks[].prompts[].decoder` | `tracks[].prompts[].decoder` | | left out | - |
| `tracks[].prompts[].points_px` | `tracks[].prompts[].points_px` | | `[]` | - |
| `tracks[].prompts[].labels` | `tracks[].prompts[].labels` | | `[]` | - |

The runs and the corrections:

| Version 1 | Version 2 | Reading version 1 | Default | Step |
|---|---|---|---|---|
| `runs` | `runs` | | `[]` | - |
| `runs[].tracks` | `runs[].tracks` | | `[]` | - |
| `runs[].start_frame` | `runs[].start_frame` | | `0` | - |
| `runs[].mode` | `runs[].mode` | | `"coarse"` | - |
| `runs[].started` | `runs[].started` | | `""` | - |
| `runs[].finished` | `runs[].finished` | | `null` | - |
| `runs[].frames_done` | `runs[].frames_done` | | `0` | - |
| `runs[].seconds_per_frame` | `runs[].seconds_per_frame` | | `0.0` | - |
| `runs[].device` | `runs[].device` | | `""` | - |
| `runs[].model_id` | `runs[].model_id` | | left out | - |
| `runs[].weights_sha256` | `runs[].weights_sha256` | | left out | - |
| `corrections` | `corrections` | | `[]` | - |
| `corrections[].time` | `corrections[].time` | | `""` | - |
| `corrections[].tracks` | `corrections[].tracks` | | `[]` | - |
| `corrections[].action` | `corrections[].action` | | `""` | - |
| `corrections[].frame` | `corrections[].frame` | | `0` | - |
| `corrections[].prompts` | `corrections[].prompts` | each item as an item of `tracks[].prompts` | `[]` | - |

### Example 1: a session made in the window

A version 1 file as 0.1.0 writes it after a run and one correction. The manifest gave the frame rate, and the
user fitted a circle. Every length in it is consistent with the others:

- The stick is 600 px for 30 mm, so the scale is 0.05 mm per px.
- The tape is 400 px for 20 mm.
- The circle's three points lie 500 px from its center: 25 mm, for a diameter of 50 mm.
- The clip covers frames 120 to 2398, every second frame: 1,140 frames.

#### Version 1

```json
{
  "format": "outline-tracker-session", "schema_version": 1, "tool_version": "0.1.0",
  "student": "ana", "notes": "second try", "complete": true,
  "video": {"relpath": "../clip_tracker.mp4", "abspath": "/data/videos/clip_tracker.mp4", "size": 52428800,
            "sha256_first_64MiB": "0cab1c9617404faf2b24e221e189ca5945813e14d3f766345b09ca13bbe28ffc",
            "width": 1920, "height": 1080, "n_frames": 2400, "fps_container": 30.0},
  "clip": {"start": 120, "end": 2398, "step": 2},
  "time": {"fps_true": 239.6, "source": "manifest", "manifest_path": "/data/videos/data/manifest.csv",
           "stopwatch": null},
  "calibration": {"stick": {"p1_px": [400.5, 900.5], "p2_px": [1000.5, 900.5], "length_mm": 30.0},
                  "click_sigma_px": 0.5,
                  "check": {"p1_px": [400.5, 950.5], "p2_px": [800.5, 950.5], "true_mm": 20.0}},
  "axes": {"origin_px": [960.5, 540.5], "angle_deg": 0.0},
  "circle": {"points_px": [[460.5, 540.5], [960.5, 40.5], [1460.5, 540.5]], "center_px": [960.5, 540.5],
             "radius_px": 500.0, "rms_px": 0.0, "dish_mm": 50.0},
  "processing": {"model": "edgetam", "device": "auto", "dish_crop": true,
                 "outline_points": 128, "radial_step_deg": 5, "shape_ok_min": 20,
                 "core_open_frac": 0.1, "jump_mm_s": 100.0, "fine_window_factor": 3.0,
                 "shape_files_for_coarse": false},
  "probes": [{"name": "LED1", "rect_px": [100, 100, 140, 140]}],
  "tracks": [{"id": "A", "color": "#FFFF00", "mode": "coarse", "fine_window_px": null,
              "start_frame": 120, "head_px": [1010.5, 520.5], "ended_at": 2000,
              "prompts": [{"frame": 120,
                           "frame_hash": "sha256:5bc20c9482ae5e22878953010a304eca4c9fbd2c82ec1d9e833416a4240f0d9a",
                           "decoder": "opencv-4.12.0/darwin/arm64", "points_px": [[1000.5, 540.5]],
                           "labels": [1]}]}],
  "runs": [{"tracks": ["A"], "start_frame": 120, "mode": "coarse", "started": "2026-10-06T14:02:11+02:00",
            "finished": "2026-10-06T14:20:40+02:00", "frames_done": 1140, "seconds_per_frame": 0.97,
            "device": "mps", "model_id": "facebook/EdgeTAM",
            "weights_sha256": "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"}],
  "corrections": [{"time": "2026-10-06T14:31:05+02:00", "tracks": ["A"], "action": "end", "frame": 2000,
                   "prompts": []}]
}
```

#### Version 2

```json
{
  "format": "outline-tracker-session", "schema_version": 2, "tool_version": "0.1.0",
  "label": "ana", "notes": "second try", "complete": true,
  "video": {"relpath": "../clip_tracker.mp4", "abspath": "/data/videos/clip_tracker.mp4", "size": 52428800,
            "sha256_first_64MiB": "0cab1c9617404faf2b24e221e189ca5945813e14d3f766345b09ca13bbe28ffc",
            "width": 1920, "height": 1080, "n_frames": 2400, "fps_container": 30.0},
  "clip": {"start": 120, "end": 2398, "step": 2},
  "time": {"fps_true": 239.6, "source": "typed", "stopwatch": null},
  "calibration": {"stick": {"p1_px": [400.5, 900.5], "p2_px": [1000.5, 900.5], "length_mm": 30.0},
                  "click_sigma_px": 0.5,
                  "check": {"p1_px": [400.5, 950.5], "p2_px": [800.5, 950.5], "true_mm": 20.0}},
  "axes": {"origin_px": [960.5, 540.5], "angle_deg": 0.0},
  "boundary": {"shape": "circle", "points_px": [[460.5, 540.5], [960.5, 40.5], [1460.5, 540.5]],
               "center_px": [960.5, 540.5], "radius_px": 500.0, "rms_px": 0.0, "diameter_mm": 50.0},
  "processing": {"model": "edgetam", "device": "auto", "boundary_crop": true,
                 "outline_points": 128, "radial_step_deg": 5, "shape_ok_min": 20,
                 "core_open_frac": 0.1, "jump_mm_s": 100.0, "fine_window_factor": 3.0,
                 "shape_files_for_coarse": false, "tracker_format": true},
  "probes": [{"name": "LED1", "rect_px": [100, 100, 140, 140]}],
  "tracks": [{"id": "A", "color": "#FFFF00", "mode": "coarse", "fine_window_px": null,
              "start_frame": 120, "head_px": [1010.5, 520.5], "ended_at": 2000,
              "prompts": [{"frame": 120,
                           "frame_hash": "sha256:5bc20c9482ae5e22878953010a304eca4c9fbd2c82ec1d9e833416a4240f0d9a",
                           "decoder": "opencv-4.12.0/darwin/arm64", "points_px": [[1000.5, 540.5]],
                           "labels": [1]}]}],
  "runs": [{"tracks": ["A"], "start_frame": 120, "mode": "coarse", "started": "2026-10-06T14:02:11+02:00",
            "finished": "2026-10-06T14:20:40+02:00", "frames_done": 1140, "seconds_per_frame": 0.97,
            "device": "mps", "model_id": "facebook/EdgeTAM",
            "weights_sha256": "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"}],
  "corrections": [{"time": "2026-10-06T14:31:05+02:00", "tracks": ["A"], "action": "end", "frame": 2000,
                   "prompts": []}]
}
```

### Example 2: a session made by from-tracker

A version 1 file as 0.2.0a1 writes it after `from-tracker` has tracked two objects:

- The frame rate came from the export's time column.
- The scale was fitted to the export: 0.05 mm per px, from 24 rows.
- There is no stick, tape check or circle.
- The clip covers frames 0 to 2398, every second frame: 1,200 frames.

#### Version 1

```json
{
  "format": "outline-tracker-session", "schema_version": 1, "tool_version": "0.2.0a1",
  "student": "ben", "notes": "", "complete": true,
  "video": {"relpath": "../clip_tracker.mp4", "abspath": "/data/videos/clip_tracker.mp4", "size": 52428800,
            "sha256_first_64MiB": "0cab1c9617404faf2b24e221e189ca5945813e14d3f766345b09ca13bbe28ffc",
            "width": 1920, "height": 1080, "n_frames": 2400, "fps_container": 30.0},
  "clip": {"start": 0, "end": 2398, "step": 2},
  "time": {"fps_true": 240.0, "source": "tracker-export", "manifest_path": null, "stopwatch": null},
  "calibration": {"stick": null, "click_sigma_px": 0.5, "check": null,
                  "tracker_fit": {"mm_per_px": 0.05, "rms_mm": 0.012, "n_points": 24}},
  "axes": {"origin_px": [960.5, 540.5], "angle_deg": 90.0},
  "circle": null,
  "processing": {"model": "edgetam", "device": "cpu", "dish_crop": true,
                 "outline_points": 128, "radial_step_deg": 5, "shape_ok_min": 20,
                 "core_open_frac": 0.1, "jump_mm_s": 100.0, "fine_window_factor": 3.0,
                 "shape_files_for_coarse": false},
  "probes": [],
  "tracks": [{"id": "A", "color": "#FFFF00", "mode": "coarse", "fine_window_px": null,
              "start_frame": 0, "head_px": null, "ended_at": null,
              "prompts": [{"frame": 0,
                           "frame_hash": "sha256:40db78ae029df9f395c8fcd445f1625e2e7bf22e0bc0dc6182ae62fee82319f0",
                           "decoder": "opencv-4.12.0/linux/x86_64", "points_px": [[700.25, 300.75]],
                           "labels": [1]}]},
             {"id": "B", "color": "#FF00FF", "mode": "coarse", "fine_window_px": null,
              "start_frame": 0, "head_px": null, "ended_at": null,
              "prompts": [{"frame": 0,
                           "frame_hash": "sha256:40db78ae029df9f395c8fcd445f1625e2e7bf22e0bc0dc6182ae62fee82319f0",
                           "decoder": "opencv-4.12.0/linux/x86_64", "points_px": [[1200.5, 640.0]],
                           "labels": [1]}]}],
  "runs": [{"tracks": ["A", "B"], "start_frame": 0, "mode": "coarse", "started": "2026-10-09T09:15:00+00:00",
            "finished": "2026-10-09T09:43:00+00:00", "frames_done": 1200, "seconds_per_frame": 1.4,
            "device": "cpu", "model_id": "facebook/EdgeTAM",
            "weights_sha256": "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"}],
  "corrections": []
}
```

#### Version 2

```json
{
  "format": "outline-tracker-session", "schema_version": 2, "tool_version": "0.2.0a1",
  "label": "ben", "notes": "", "complete": true,
  "video": {"relpath": "../clip_tracker.mp4", "abspath": "/data/videos/clip_tracker.mp4", "size": 52428800,
            "sha256_first_64MiB": "0cab1c9617404faf2b24e221e189ca5945813e14d3f766345b09ca13bbe28ffc",
            "width": 1920, "height": 1080, "n_frames": 2400, "fps_container": 30.0},
  "clip": {"start": 0, "end": 2398, "step": 2},
  "time": {"fps_true": 240.0, "source": "tracker-export", "stopwatch": null},
  "calibration": {"stick": null, "click_sigma_px": 0.5, "check": null,
                  "tracker_fit": {"mm_per_px": 0.05, "rms_mm": 0.012, "n_points": 24}},
  "axes": {"origin_px": [960.5, 540.5], "angle_deg": 90.0},
  "boundary": null,
  "processing": {"model": "edgetam", "device": "cpu", "boundary_crop": true,
                 "outline_points": 128, "radial_step_deg": 5, "shape_ok_min": 20,
                 "core_open_frac": 0.1, "jump_mm_s": 100.0, "fine_window_factor": 3.0,
                 "shape_files_for_coarse": false, "tracker_format": true},
  "probes": [],
  "tracks": [{"id": "A", "color": "#FFFF00", "mode": "coarse", "fine_window_px": null,
              "start_frame": 0, "head_px": null, "ended_at": null,
              "prompts": [{"frame": 0,
                           "frame_hash": "sha256:40db78ae029df9f395c8fcd445f1625e2e7bf22e0bc0dc6182ae62fee82319f0",
                           "decoder": "opencv-4.12.0/linux/x86_64", "points_px": [[700.25, 300.75]],
                           "labels": [1]}]},
             {"id": "B", "color": "#FF00FF", "mode": "coarse", "fine_window_px": null,
              "start_frame": 0, "head_px": null, "ended_at": null,
              "prompts": [{"frame": 0,
                           "frame_hash": "sha256:40db78ae029df9f395c8fcd445f1625e2e7bf22e0bc0dc6182ae62fee82319f0",
                           "decoder": "opencv-4.12.0/linux/x86_64", "points_px": [[1200.5, 640.0]],
                           "labels": [1]}]}],
  "runs": [{"tracks": ["A", "B"], "start_frame": 0, "mode": "coarse", "started": "2026-10-09T09:15:00+00:00",
            "finished": "2026-10-09T09:43:00+00:00", "frames_done": 1200, "seconds_per_frame": 1.4,
            "device": "cpu", "model_id": "facebook/EdgeTAM",
            "weights_sha256": "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"}],
  "corrections": []
}
```

### How the tests use this page

The tests of the version 1 reader read this section:

- the two columns of keys, to check that every key of each version has its row;
- the column "Default", for the defaults;
- the four examples above. Each version 1 example must read as its version 2 example: equal as JSON,
  with the same keys in the same order. Each version 2 example must be read and written back unchanged.

A change of the format therefore starts with a change of this page.
