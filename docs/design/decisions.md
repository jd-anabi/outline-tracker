# Decisions of the first build

These are the decisions the first build made where the specification of version 0.1.0 left a choice. They are copied unchanged from section 4 of `docs/PLAN.md` at commit `67908d8`. The decisions of this phase are in [section 3 of the roadmap](../ROADMAP.md#3-decisions-the-owner-makes-first).

The roadmap records these later changes to them:

- X2 and X3: ended in W1 step 5, which removed the class template (`tests/reference/`) and the port-fidelity tests (roadmap, section 4, W1).
- X7: changed in W0 by task C9. The model is made in a short-lived thread of its own and is used in the one worker thread, so that an export can run while a model loads (roadmap, section 4, W0).
- X8: confirmed, not changed, by decision 3 (roadmap, section 3).
- X17: confirmed, not changed, by decision 27 (roadmap, section 3).

These are the agent's defaults. J can overrule any of them.

| # | decision | why |
|---|---|---|
| X1 | Package folder `outline_tracker/` at the repo root, as §12 draws it; hatchling build; `requires-python >= 3.11` with numpy and scipy pinned per Python version | matches the spec's paths; the wheel ships only the package |
| X2 | `tests/reference/` is excluded from the normal test run. One test runs the 27 unmodified template tests against the unmodified reference copy in a subprocess | shows last week's code still passes with this week's libraries, so a failing ported test points at the port; avoids name clashes between the two test sets |
| X3 | Ported tests change only their import lines (see the table in section 8). The three end-to-end tests of `test_segment.py` keep their assertions on frames, times and positions, but look for the files in the §8.1 run folder | §8.1 and §8.3 move the output folder; nothing else changes |
| X4 | Exact random access (`FrameSource`) with OpenCV only, no PyAV. Tracking keeps the plain sequential decode. For display, a jump seeks, then **identifies the frame actually delivered** from its timestamp against a table of all frame timestamps (read once with the bundled ffmpeg, no decoding), then steps forward to the wanted frame. If anything does not match, it decodes forward from the start | §3.5 forbids unverified seeking, with reason: measured before coding, plain OpenCV seeking is exact on evenly timed clips (0 of 50 wrong) but silently 1–2 frames off when timestamps have gaps (10 of 50 wrong), and `convert` keeps the phone's timestamps. The verified method: 0 wrong in 1,518 reads on 7 clips, about 20 ms per jump at 1080p |
| X5 | `results.npz` also stores `core_fallback`, `second_fraction` and `core_r_px` per frame | `ORIENT` and `MULTI` (§9) cannot be computed from the §8.12 keys alone |
| X6 | `CONTACT` threshold per pair of tracks = `max(3 px, 2 · max(cell_i, cell_j))`, from each track's stored grid cell size | §9 says "2 coarse grid cells"; for two fine tracks that leaves 3 px |
| X7 | One worker thread (64 MiB stack) owns the loaded model and runs previews, runs, exports and probe passes one at a time. Preview is disabled while a run is active. After an Apple-GPU fallback the app stays on cpu | one copy of the weights; no concurrent use of the model; a default Qt thread has a 0.5 MiB stack on macOS, where the model has never run |
| X8 | Frame hash = SHA-256 of the decoded RGB frame's bytes (2.5 ms at 1080p), stored with a decoder tag (OpenCV version, platform, machine). Same tag and a different hash aborts the run. A different tag (the folder moved to another computer) recomputes the stored hash here and logs it | decoders on Windows and macOS may differ in the last bit, and §8.1 wants the run folder to be movable |
| X9 | fps_true for `from-tracker` and `probe`: `--fps`; else `data/manifest.csv` in the current folder (last week's rule); else the same file searched upward from the export's folder, then the video's folder (§4.1); else the export's `t` column, recorded as source `tracker-export` with a printed note | keeps last week's behavior and adds §4.1's search |
| X10 | Two hidden commands not in §11: `compare-tracks NEW_DIR OLD_DIR [--by-position]` (RMS difference in px per track on common frames) and `synth {dish,closeup} OUT.mp4` (writes a synthetic clip) | the go/no-go checks need them, also after `uv tool install` on Windows |
| X11 | New CSVs: UTF-8, `\n` line ends on every platform, NaN as an empty cell. The Tracker-format files keep the ported writer's bytes (so CRLF on Windows, as last week), written to a temporary file and renamed | §8.2 fixes the decimals only; §8.3 asks for last week's bytes |
| X12 | `radial.csv` and `outlines.npz` are always written; with no fine track they hold only the header or the `meta` key | §8.1 marks only `probes.csv` as optional and §14.3 wants every P0 file |
| X13 | `score` in `results.npz` is the model's object-presence logit | it is the only score the Hugging Face video model returns |
| X14 | Track colors follow last week's overlay (A is yellow), stored as RGB hex | students' overlays look the same as last week |
| X15 | `results.npz` also keeps each frame's mask crop, bit-packed (about 1 MB per 1,200-frame fine track) | `core_open_frac` is a setting (§8.10); without the masks, changing it would mean tracking again |
| X16 | Synthetic shapes are implicit functions; `ExactFake` returns the analytic signed distance to the shape as logits | §13.2 says "signed distance to the mask boundary"; a distance transform of the pixel mask gives a staircase outline (perimeter +2 to 3%, radial 3 to 5%) that cannot meet §13.1's 0.5% and 1% |
| X17 | Heading reference: an axis jump or a core fallback flags that frame `ORIENT` but still becomes the next reference; only frames with an undefined axis are skipped | read literally, §7.3 flags every frame after one large turn (measured: 470 of 470) |
| X18 | session.json gains optional keys (schema version stays 1): `calibration.tracker_fit = {mm_per_px, rms_mm, n_points}` for sessions made by `from-tracker` (no stick exists), `time.source = "tracker-export"`, `model_id` and `weights_sha256` in each run entry, and a decoder tag next to each frame hash | `export` must rebuild everything from the session without loading torch; X8 |
| X19 | Options beyond §11: `from-tracker --student` and `--no-overlay` (`--no-video` accepted), `probe --student`, hidden `check --seek` | §8.1 needs a name; the go/no-go 1 checks use the others |
| X20 | An off-grid frame snaps to the first grid frame at or after it (§3.4 does not give the direction) | start, re-track and new-piece all mean "from this frame on", so nothing before the typed frame is touched |
| X21 | The `Segmenter` protocol of §12 gains one method, `preview(image, prompts)`: masks for one frame, no tracking state kept | §5's preview and §6.3's window from the start-frame mask both need it, also with the stand-ins |
| X22 | The console `CHECK:` lines and `from-tracker`'s returned `flags` are last week's three checks only (lost, jump, size change), in last week's wording. All nine §9 codes go to the `flags` columns and to the QC summary in run.log | "the same console lines as last week"; every dish-scale track carries `LOWRES` and `HEADGUESS`, which are not alarms |

**Test conditions checked before coding (no action needed).** Every tolerance of §13.1, §13.2 and
§13.4 is reachable under these conditions, which the tasks state:

| check | measured | tolerance | condition |
|---|---|---|---|
| moments, ellipse a = 40, b = 15 | L1 0.30%, L2 1.04%, axis 0.32° | 2%, 2%, 1° | rasterize at pixel centers (a `cv2`-drawn ellipse gives 4.5%) |
| circle R = 50 perimeter, 256 points | −0.006% | 0.5% | analytic logits (distance-transform logits give +1.7 to 3.0%) |
| radial profile, ellipse | 0.11–0.22% | 1% | analytic center and heading (through the pipeline: 0.7–1.4%, so the expected value there is the exact ray–ellipse intersection from the measured center) |
| L-shape solidity | 0.50% at 120 px across (1.2% at 40 px) | 1% | L at least 120 px across |
| core, sideways rods | axis 0.7°, centroid 0.24 px, fraction 2.0% | 5°, 0.5 px, 5% | rods reach 30 px beyond the body, so the premise (lateral > axial moment) is true |
| `ThresholdFake` through H.264 | 0.12–0.18 px (R = 12, crf 10); 0.30 px (R = 6, crf 23) | 0.25 px | disks of radius 12 px, crf 10 |
| circle fit, 6 points, σ = 0.5 px | fails for 0.2% of seeds if evenly spread, 7.5% if random | 1.0 px | points spread evenly, fixed seed |
| §13.4 solidity signal | peak at 9.0 Hz; ideal-grid RMS 0.003 | ±0.5 Hz; 0.02 | true solidity from the analytic outline, not the pixel mask |
| overlay colors | a 1 px yellow line decodes as about (232, 243, 111) | none | colors are asserted on the drawn array, never on the decoded file |
