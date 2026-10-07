# Roadmap inventory: where the app and the tool are slow or wasteful

> Inventory for `docs/ROADMAP.md`: where the app is slow or wasteful. Read-only findings at commit `f73d29f` (2026-10-07).
> File and line pointers are true for that commit. "The ledger", "the build ledger" and "the
> scratch folder" are private working notes of the first build; they are not in the repository.
> The design note of the window is `docs/design/gui_design.md`.

Commit f73d29f, measured 2026-10-07 on an Apple M1 Max (10 cores, 64 GB), disk cache warm, another job running (load about 3). No real model was loaded and no real video was read.

- Stand-ins: `segmenter/fake.py` `ExactFake`, records from `measure.measure_mask`, clips from `synthetic.render`.
- "10x1200" = 10 tracks of 1,200 tracked frames (a 10 s clip at 240 fps, step 2), bodies 14 px long, frame 640 x 480.
- "flat" = `synthetic.dish_scene` at 1920 x 1080, 240 fps. The 2 min and 5 min clips are the 20 s clip repeated by stream copy. "textured" = a 1080p noise clip from ffmpeg, 271 kB per frame: an upper bound. A real phone clip lies between the two: not measured.
- GUI numbers: the window object was built on Qt's offscreen platform, never shown, factory None, as the GUI tests do. Only the work of the slots is timed. Painting is not.
- "Computed" = a measured unit cost times a size. It was not run.
- The scripts and their JSON outputs are in the session scratch folder, `perf/bench_{store,video,overlay,gui,misc}.py`. They are temporary.

## 1. Ranking

| rank | 10 s clip, 240 fps, 10 objects (12,000 records) | 5 min clip (72,000 frames) |
|---|---|---|
| 1 | P1: every frame shown and every click rebuild the tracks' arrays on the GUI thread | P2: results.npz is rewritten whole and re-read three times at each autosave; memory |
| 2 | P3 + P4: 5 s for the flags, 5 s again for the export | P1: one frame shown costs about 0.2 s per track (computed) |
| 3 | P2: the window stands still 0.4 s at each autosave | P5 + P6: minutes of decoding from frame 0; minutes-long jobs without progress or cancel |
| 4 | P7: start-up | P3 + P4: minutes per derive, done twice, every track in memory at once |
| 5 | P6: overlay 3 to 11 s behind a busy bar | P9, P8: bottom bar, opening |

Tracking itself is the model's time. docs/VALIDATION.md 1.2 and 4.4: 0.37 s per frame for one object and 2.2 to 2.3 s for ten on cpu, 0.11 to 0.15 s for one on mps. The app's own share per tracked frame is small: decode 0.9 to 4.0 ms, `measure_mask` 0.23 ms per object, P12 about 6 ms per object. Faster tracking is a question of models and backends, not of this list.

Where each suspect of the build is answered:

| suspect | answer |
|---|---|
| `flags_table` takes 4 to 5 s | confirmed, 5.2 s: P3 (derive, 3.3 to 4.0 s) and P4 (CONTACT, 1.8 s); loading is 0.1 s |
| computed twice between export, flags and CHECK lines | yes: derive runs twice, and `from-tracker` loads the results three times: P3 |
| `arrays()` rebuilds; `add_prompt` builds three times; no frames accessor | confirmed, and the GUI calls it 20 to 100 times per action: P1 |
| results.npz rewritten whole | confirmed, and re-read three times by the window: P2 |
| opening on the GUI thread | confirmed; 0.04 to 0.30 s on warm local files: P8 |
| torch before Qt, 4 s | 0.6 to 0.8 s warm; cold not measured: P7 |
| start frame decoded three times | confirmed by count; the cost is the read from frame 0: P5 |
| overlay, probes, export, model load: no progress, no cancel | confirmed: P6 |
| frame cache, preview copy, GUI work during a run, session saves | P10, section 3, P2, P14 |
| the test suite | section 4 |

## 2. Items

### P1. The store holds records; every reader converts a whole track
- Where: `results.py` `ResultsStore.arrays` (l.146), `_to_arrays` (l.247). GUI-thread callers: `gui/overlays.py:195` (every track at every frame shown), `gui/panels/objects_panel.py:71` `status_of` -> `tracking_plan.pending_from` (l.104), `gui/panels/track_panel.py:314-327`, `tracking_edit.add_prompt` (l.250-265), `_reach` (l.198), `tracking_plan.partial_tracks` (l.216, 220).
- Cost at 10x1200: one call 2.6 ms; 18.7 ms at 7,200 rows (linear). One tick of Play: 55 ms of slot work, 20 calls (47 ms), against a tick of 33 ms (`gui/navigation.py:34`). `controller.touch()`: 210 ms, 50 calls. A click (`Prompts.add_point`): 218 ms, 53 calls (`add_prompt` alone: 3 calls, 6.8 ms). Undo: 328 ms, 100 calls. Opening the session: 652 ms, 110 calls, 3 loads.
- Why: a track is a dict of `PixelRecord` objects. To read one row, or only the frame numbers, 23 arrays are built.
- Fix: keep each track as columns that grow; `arrays()` returns views. Add `frames(track_id)`, `has(track_id, frame)`, `last_frame(track_id)`. Compute the panels' status once per version of the store, not per refresh.
- First step, a few lines: keep a track's `TrackArrays` in the store, read-only, until `put` or `_keep` touches that track.
- Proof: a test that counts calls of the array builder: 0 for 100 frame changes, one click and one undo. A test that two `arrays()` calls share memory.

### P2. results.npz: whole rewrite at every autosave, three re-reads on the GUI thread
- Where: `tracking.py:58` `AUTOSAVE_EVERY = 200`, `_Runner._save` (l.364) -> `ResultsStore.save` (`results.py:174`). GUI side: `gui/worker_jobs.py:269` `_on_changed` (touch, then `saved`); three `ResultsOnDisk` (`gui/overlays.py:145`, `gui/prompts.py:115`, `gui/worker_jobs.py:152`; class in `gui/click_rules.py:112`); `ResultsStore.load` -> `_to_records` (`results.py:259`).
- Cost: save 48 ms and 27.0 MB at 10x1200 (2,252 bytes and 4 µs per record); 232 ms and 162 MB at 10x7200. Load 59 ms and 421 ms. A loaded store takes 3,471 bytes per record (42 MB at 10x1200). One autosave seen from the GUI thread: 389 ms (3 loads 212 ms, 30 `arrays()` calls). A job's end: 597 ms. A correction also saves the whole file (`tracking_edit._save`, l.121).
- Computed for 10 objects x 36,000 frames: 180 saves, about 2 min of saving, 73 GB written, last file 0.81 GB, about 12 s standstill per autosave near the end, 1.25 GB per loaded store and up to four stores (the worker's and the three GUI copies).
- Why: one uncompressed file is the only form of the results. The window learns what was tracked by reading it again.
- Fix: append-only blocks per run while tracking, results.npz written once at the end or on demand. One read-only store for the whole window, fed by the job (`tracking.Callbacks.frame_result`, l.135, exists; the GUI passes a no-op, `gui/worker_jobs.py:105`) or read off the GUI thread.
- First step: one `ResultsOnDisk` per window instead of three. That is one load per autosave, not three.
- Proof: a test with a counting `atomic_write`: bytes written per autosave do not grow with the frames already saved. A test that one autosave causes at most one `ResultsStore.load`, and none in the GUI thread.

### P3. Derive is slow per row and is done again for every consumer
- Where: `tracking_flags.flags_table` (l.22), `export.derive_all` (l.89), `derive.derive_track` (l.115), `derive_outline.outline_quantities` (l.182, loop per row l.212-218), `hull_area_and_feret` (l.106), `farthest_crossings` (l.61).
- Cost at 10x1200: `flags_table` 5.2 s = `derive_track` x 10: 3.3 to 4.0 s, `compute_flags` 1.8 s (P4), load 0.06 s, `arrays` 0.04 s. Inside derive: hull and Feret 1.9 s (12,000 times SVD + Qhull + pdist), rays 1.6 s, resampling 0.3 s. `derive_all` 5.2 s. `export_all` without overlay 5.5 s (CSV text 0.5 s); with shape files for every track 6.8 s (radial.csv 8.0 MB, outlines.npz 22 MB). 0.28 ms per row: computed 100 s for 360,000 rows.
- Memory: `derive_all` keeps every track's `DerivedTrack` at once. `outline_xy_mm`, `outline_xieta_mm` and `radial_mm` are about 4.7 kB per row at 128 points; `stored` and `polygons` add 8 kB per row while one track is derived (`derive.py:157-158`). Computed: 1.7 GB kept for 360,000 rows.
- Computed twice: yes. The review panel lists after a run (`gui/review_table.py:222`), Export all derives again (`export.py:164`). `from-tracker` derives in `export_all` (`from_tracker.py:208`) and again for the CHECK lines (l.279, with a second `ResultsStore.load`); `write_overlay` loads session and results a third time (`overlay.py:145-146`). `flags_table` keeps only the rows and drops its derived tracks.
- Fix: one cached `derive_all` result per (results file identity, calibration, fps_true, circle, processing, head clicks), shared by flags, export and CHECK lines; `export_all` returns its `ExportData`. Hull area, Feret, perimeter and area scale with k or k^2: compute them once in px. Replace the loop per row by array code.
- Proof: a call-count test: `derive_track` runs once per track for "list the flags, then export" and for one `from_tracker` run. A test that a new stick length calls no hull code.

### P4. CONTACT: exact outline comparison on every near frame
- Where: `qc_contact.contact_marks` (l.54), `_contact_rows` (l.84, 8 frames at a time), `_distances` (l.102), `_least_square` (l.116).
- Cost: 1.75 s at 10x1200 with one pair within the limit on all 1,200 frames (1.5 ms per near pair and frame). Every pair near on every frame: 64 s. Far pairs cost about 0.2 ms per pair; `tests/test_qc_contact.py::test_ten_tracks_over_1200_frames_take_well_under_a_second` covers only that case.
- Why: each near frame builds several 256 x 256 float64 arrays (point to edge both ways, edge crossings).
- Fix: decide most frames from the vertex-to-vertex distances with the point spacing as error bound, keep the inside test, run the exact code only in the band between.
- Proof: a test that counts rows given to `_distances` for two tracks 1 px apart and two tracks 50 px apart; the same marks as today on the synthetic contact scene.

### P5. Every start frame is reached by decoding from frame 0, two to four times
- Where: `video.iter_rgb_frames` (l.229), called by `tracking_guard.check_start_frames` (l.88), `tracking_fine.fine_starts` (l.247), `tracking._Runner.track` (l.305), `from_tracker.py:183`.
- Cost: counted for a start at frame 2,000: a coarse job reads from frame 0 twice, a fine job three times, `from-tracker` once more. Grab rate at 1080p: 5,500 frames per s (flat), 303 (textured). The last frame of the 5 min clip: 13 s per read (flat, measured), about 240 s (textured, computed).
- Why: frame k is defined by counting. `FrameSource` already proves a seek against the timestamp table (`frame_source.py:208-247`); tracking does not use it.
- Fix: start a run from a verified seek, fall back to counting. Decode a start frame once and hand it to guard, preview and run.
- Proof: a test with a counting capture: frames grabbed before the first tracked frame do not grow with the start frame; one decode of the start frame per job. Bit equality is `frame_source.check_seek`.

### P6. Long jobs without progress or cancel, all on the one worker thread
- Where: `overlay.write_overlay` (l.111, loop l.164), `probes.measure_probes` (l.27, loop l.68), `export.export_all` (l.142, called at `gui/panels/export_panel.py:119`), the flags listing (`gui/review_table.py:335`), `Worker.run` only while the model is ready (`gui/worker.py:213`), `Worker.stop` waits (l.345-347).
- Cost: overlay 2.6 ms per frame (flat, 3 tracks) to 9.0 ms (textured, 10 tracks): 3 to 11 s for 1,200 frames; computed 1.5 to 5.4 min for 36,000. Probes 1.1 to 3.7 ms per frame: computed 1.3 to 4.4 min for 72,000 frames. The 5.2 s listing of P3 runs on the worker thread: an outline after a click waits behind it. A model load cannot be interrupted either: its time was not measured (no model was loaded).
- Fix: give these functions `progress(done, total)` and `should_cancel()`, as `tracking.Callbacks` has. Run jobs that need no model on a second thread or process.
- Proof: tests in the style of the cancel tests of `tests/test_tracking_job.py`: a cancelled overlay leaves the old overlay.mp4 and returns within one frame. A GUI test: an outline is answered while a listing is parked on a gate.

### P7. Start-up: torch before the window
- Where: `gui/app.py:22` (`import torch` before Qt), `gui/app.py:39` (imports `from_tracker`), `geometry.py:29` (`import pandas`).
- Cost, warm, fresh processes: torch 0.60 to 0.84 s, PySide6 0.06 s, pyqtgraph 0.13 s, the package's modules 0.33 s (pandas 0.18 s, cv2 0.08 s), `QApplication` 0.03 s, `MainWindow()` 0.19 s: about 1.4 to 1.6 s before `show()`. With Qt first the window's modules are ready at 0.58 s. The 4 s seen at the build were not reproduced: probably a cold start (torch is 557 MB on disk). Cold start: not measured.
- Later, in the worker: torchvision 0.89 s, transformers 0.97 s, its Sam2Video classes 0.65 s, timm 0.18 s. Command line: `import outline_tracker.cli` 30 ms, `outline-tracker --version` 0.10 s.
- Why: SPEC 10.2 (torch after Qt failed on Windows). The CI step "Qt, then torch (information only)" in `.github/workflows/tests.yml` passes with the pinned torch (build ledger, task A01).
- Fix: keep torch first on Windows only, or drop it after a test on real Windows machines; elsewhere show the window, then import torch in the worker. Import pandas inside the functions that read CSV files.
- Proof: extend `tests/gui/test_import_order.py`: off Windows, `torch` is not in `sys.modules` when the window is shown; importing `outline_tracker.gui.main_window` in a fresh process does not import pandas.

### P8. A video is opened on the GUI thread
- Where: `gui/session_controller.py:131` (`_read` -> `FrameSource`), l.137 (`sha256_first_64mib`), l.173 (`locate_video` -> the hash), `frame_source.py:112` -> `video.frame_timestamps` (l.259, an ffmpeg process that reads every packet), `gui/main_window.py:240` (wait cursor).
- Cost, flat 1080p: `FrameSource(path)` 0.04 s (2 s clip), 0.05 s (20 s), 0.14 s (2 min), 0.30 s (5 min, 72,000 frames, 18 MB). The table is nearly all of it: 0.02, 0.04, 0.12, 0.28 s (3.7 µs per packet). Textured, 651 MB, 2,400 frames: open 0.27 s, table 0.13 s (0.2 s per GB, warm). Hash of 64 MiB: 32 ms. With results, the session's three loads come on top (P1, P2).
- Why it stays on the list: the table reads the whole file. A cold cache, a slow disk and a cloud placeholder were not measured.
- Fix: open in a thread with a line of progress; show frame 0 at once (counting needs no table) and use the table when it is there. Keep the table next to the session, keyed by size and hash.
- Proof: a GUI test with `frame_timestamps` parked on a gate: the window shows frame 0 and answers. A test that the second open of a file starts no ffmpeg process.

### P9. The bottom bar scans the whole grid; the slider shows every value
- Where: `gui/navigation.py:305` `_nearest` (a `min` over the grid with a lambda), `set_grid` (l.268), called at `gui/main_window.py:268` after every frame shown and at l.288 on every change of the session; the slider at `gui/navigation.py:124`.
- Cost: `set_frame` 0.15 ms with 1,200 grid frames, 3.9 ms with 36,000, 7.7 ms with 72,000. `set_grid` 9.8 ms with 72,000. Each slider value is shown before the next: a jump is 9 to 11 ms (flat) and 155 ms (textured).
- Fix: keep the grid as a `range`, compute the index. Show only the newest slider value when the last frame is done.
- Proof: a unit test with a grid of 10^6 frames that counts item reads in `set_frame`. A GUI test: 50 slider values in a row give fewer than 50 `FrameSource.get` calls.

### P10. Frame cache: 400 MB, never hit when playing forward
- Where: `frame_source.py:57-59`, `get` (l.146-171).
- Cost: 64 frames of 1080p are 398 MB; 4K gets 16 frames. A new frame is always a miss; the only hits while playing are the second `get` of the same frame (`gui/click_rules.py:107`). Forward `get`: 2.3 to 2.6 ms per frame (flat); 4.0 ms at step 1 and 7.3 ms at step 2 (textured). One step back outside the cache: 7 to 11 ms (flat), 176 ms (textured, GOP 24).
- Fix: a smaller default, set in bytes; on a step back decode the group of pictures once and keep it.
- Proof: a test on `FrameSource.stats`: 24 steps back outside the cache make 1 seek, not 24.

### P11. The overlay graphics are made anew at every frame
- Where: `gui/overlays.py:185-240`: `refresh` removes and makes two `PlotCurveItem`s and a `TextItem` per track.
- Cost: `_draw` 6.8 ms per tick for 10 tracks under the profiler (P1 hides it today). Painting: not measured.
- Fix: keep the items per track; `setData`, `setPos`, hide.
- Proof: a test that 100 frame changes add no item after the first (count `view.add_item`).

### P12. After the model: full-frame arrays per object
- Where: `segmenter/hf.py:248-260` `_results` (`post_process_masks` gives a full-frame float array per object), `segmenter/base.py` `crop_to_bbox` (`np.nonzero` on the full frame).
- Cost, stand-in without the model: `logits > 0` + `crop_to_bbox` 5.1 ms per object and frame at 1080p, 20.5 ms at 4K. Bilinear 256 x 256 to the frame with torch on the processor: 0.75 and 2.9 ms. Ten objects: about 60 ms per frame at 1080p, 230 ms at 4K. The real processor call: not measured.
- Fix: find the box on the 256 x 256 grid; upsample only the box with a margin.
- Proof: a test with a fake processor: the largest array made per object is the box, not the frame; the same positions as today.

### P13. `measure_mask` grows fast with the size of the mask
- Where: `measure.measure_mask` (l.129), `_opening` (l.245, a disk kernel of 0.1 x length), `_level_set` (l.269). On the GUI thread per outline: `gui/worker.py:92`, and again in `tracking_fine.fine_window` (l.77) through `gui/worker.py:97`, both from `gui/prompts.py:375`.
- Cost by diameter: 14 px 0.23 ms, 50 px 0.58 ms, 150 px 2.4 ms, 300 px 17 ms, 600 px 175 ms, 1,000 px 225 ms. Per object and frame in the worker; twice per object and outline on the GUI thread.
- Fix: do the opening with two distance transforms (it should give the same pixels for a disk). Measure once per outline, in the worker.
- Proof: old and new opening equal on random masks; a call-count test: one `measure_mask` per object and outline.

### P14. Small things on the GUI thread
- Lazy imports land there: `derive_outline.py:118` (scipy.spatial, at the first outline) and `geometry.py:274` (scipy.optimize, at the first circle fit). Warm 0.26 s and 0.09 s; the build ledger notes 0.6 to 4 s cold. Fix: import both in the worker after start. Proof: both are in `sys.modules` when the worker says ready.
- session.json: `Session.save` 0.46 ms (10 tracks, 5.9 kB), 0.75 s after a change (`gui/session_controller.py:38`). A locked target sleeps about 5 s in the GUI thread (`fileio.py:30`, `replace_with_retry` l.73): from the code, Windows only, not measured. Fix: no sleep there; retry from a timer. Proof: a rename that always raises: `_save` returns without `time.sleep`.
- The video check runs in its own thread (`gui/panels/video_panel.py:340`): 0.6 s (flat), 10 s (textured: 60 seeks, `video._ramp_ratio`, l.169). Closing the window waits for it (l.305). Fix: let it be cancelled.
- The flags list: 26,401 rows at 10x1200 (HEADGUESS is on every row of a track without a head click, `qc.py:95`). `gui/panels/review_panel.py:335` filters the whole list in the GUI thread. At 360,000 records: not measured.

## 3. Checked, cheap
- The copy of the frame for an outline (`gui/worker_engine.py:131`, `gui/click_rules.py:191`): 0.09 to 0.10 ms at 1080p.
- `video.frame_hash` of a 1080p frame: 2.6 ms. `store.put`, `Session.load` (0.3 ms), `fileio.append_block` on a run.log of 200 kB (0.3 ms).
- `copy.deepcopy(session)` once per job (`tracking.py:177`, `gui/worker_jobs.py:228`). `copy.deepcopy(store)` is 180 ms at 10x1200 but the GUI passes no store (`tracking_edit._save`, l.131).
- During a run the GUI thread gets one `Jobs.progress` per tracked frame: two calls on the progress bar (`gui/panels/track_panel.py:257`). The overlays read the file only when `saved` comes (`gui/overlays.py:159`): that is P2.
- `HFSegmenter._infer` (the processor's resize per frame, `segmenter/hf.py:272`): not measured.

## 4. The test suite
- One run of `uv run pytest -m 'not slow' -q --durations=25 -p no:cacheprovider`: 2,684 passed, 1 skipped, 13 xfailed, 34 deselected in 190 s (user 176 s, sys 47 s). It runs in one process (no pytest-xdist in `pyproject.toml`).
- By folder (junit times): `tests/` 2,015 tests, 113 s, 56 ms each. `tests/gui/` 683 tests, 72 s, 106 ms each. The median test takes 6 ms; 215 tests above 0.2 s take 108 s; 20 above 1 s take 33 s.
- Slowest tests: `tests/gui/test_video_view.py::test_the_click_test_passes_on_a_screen_scaled_to` 3.1 and 2.6 s; `tests/test_repo_rules.py::test_template_tests_pass_on_reference` 2.6 s; `tests/test_port_fidelity.py::test_selftest_clip_writes_the_files_the_reference_selftest_writes` 2.3 s; `tests/test_readme.py::test_the_example_output_is_what_the_commands_print` 2.0 s (setup); `tests/test_synthetic.py::test_true_solidity_of_the_closeup_shrimp_beats_at_9_hz_not_18` 1.8 s; then 11 entries of `tests/test_selftest.py` at 1.0 to 1.5 s.
- Slowest files: `test_selftest` 14.8 s (19 tests), `gui/test_video_view` 8.2 s (23), `gui/test_review` 7.8 s (26), `test_export_shapes` 6.1 s (16), `test_corrections_gaps` 5.8 s (61), `gui/test_export_panel` 5.3 s (21), `test_frame_source_jumps` 5.0 s (25).
- Causes seen: `selftest.selftest` writes a new 1080p clip in every test. Two tests start pytest again in a subprocess (`tests/gui/test_video_view.py:272`). `ExactFake._truth` builds the full-frame mask and logits per object and frame (`segmenter/fake.py:200-201`): 6.8 ms per object and frame at 640 x 480.
- Fixes: one selftest clip per test session; `ExactFake` works inside the object's window (`synthetic._window`); tracked run folders from session fixtures, copied per test; pytest-xdist (a new dependency: ask in docs/PLAN.md first, CLAUDE.md).
- CI at 4 to 6 min per platform: not verified.

## 5. A benchmark layout for the repository
A folder `benchmarks/`, outside pytest, run by hand or by a manual CI job. Each script prints a table and writes JSON with the commit, the machine and the sizes. A `--quick` size runs in under a minute.

| script | times | inputs |
|---|---|---|
| `make_inputs.py` | nothing | stores of N tracks x F frames from a `synthetic.Scene` and `measure_mask` (no video); flat clips from `synthetic.render` at 2 s and 20 s, longer ones by stream copy; one textured clip from ffmpeg |
| `bench_store.py` | `arrays`, `save`, `load`, memory per record, the sum over a run's autosaves | N in 1, 10, 100; F in 1,200, 7,200, 36,000 |
| `bench_derive.py` | `derive_track`, `compute_flags` (far pairs, one near pair, all pairs near), `flags_table`, `derive_all`, `export_all` | the same stores |
| `bench_video.py` | `FrameSource` open, `frame_timestamps`, the hash, forward `get`, jump, step back, grab rate to the last frame, `check_video` | clips of 2 s, 20 s, 2 min, 5 min, flat and textured |
| `bench_outputs.py` | `write_overlay`, `measure_probes`, ms per frame | a 1080p clip with 3 and 10 tracks |
| `bench_gui.py` | a Play tick, `touch`, a click, undo, an autosave seen from the GUI thread, opening a session; with counts of `arrays` and `load` calls | offscreen window, results 10x1200 and 10x7200 |
| `bench_start.py` | import stages in fresh processes, both import orders | none |

Wall times go to the benchmark report only. The proofs above are call-count and size tests in `tests/`, so they do not depend on the machine.

## 6. Not verified
- Anything with the real model: load time, `_infer`, per-frame cost on mps for ten objects.
- A real phone video: decode rate, file size, opening with a cold cache or from a cloud folder.
- Windows: start-up, the locked-file waits, the import order with other torch versions.
- Painting in the window (pyqtgraph `setImage` and the items) and how the GUI thread behaves while the worker holds the GIL.
- Sizes beyond 10x7200 records: every number for 36,000 or 72,000 frames is computed.
