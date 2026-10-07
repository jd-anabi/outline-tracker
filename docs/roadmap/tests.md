# Roadmap inventory: the test suite and the debt in it

> Inventory for `docs/ROADMAP.md`: the test suite and the debt in it. Read-only findings at commit `f73d29f` (2026-10-07).
> File and line pointers are true for that commit. "The ledger", "the build ledger" and "the
> scratch folder" are private working notes of the first build; they are not in the repository.
> The design note of the window is `docs/design/gui_design.md`.

State of `main` at commit f73d29f, read on 2026-10-07. Read only. Nothing was run except test collection, greps, and one timing of clip rendering in a scratch folder. The fast suite and the slow tests were **not run** here. Paths are relative to the repository root. "ledger:N" is line N of `.superpowers/sdd/PLAN/progress.md`; "PLAN" is `docs/PLAN.md`; "VALIDATION" is `docs/VALIDATION.md`.

## 0. Totals

- `uv run pytest --collect-only -q -p no:cacheprovider`: **2732 tests collected**. With `-m "not slow"`: 2698. With `-m slow`: 34.
- Node ids by folder: `tests/` top level 2015, `tests/gui/` 683, `tests/slow/` 34.
- `git grep -c 'def test_' -- tests` sums to **1680** test functions: 1081 top level, 546 gui, 28 slow, 25 in `tests/reference/` (not collected, `pyproject.toml:50`; they run as 27 tests in a subprocess, see 5).
- 116 test files (67 + 41 + 8) and 24 helper modules; 36,187 lines without `tests/reference/`.
- Last recorded fast run on macOS: 2682 passed, 1 skipped, 13 xfailed (ledger:463). Not re-run. That is 2696 node ids; 2698 are collected now, so 2 were added since.
- xfail marks: **18**, all `strict=True`: 15 in the fast suite, 3 in `tests/slow/` (5 node ids). Two more grep hits only use the word (`tests/gui/test_finish.py:144`, `tests/test_repo_rules.py:96`). The root `conftest.py` has none.

## 1. Tests marked xfail

The build rule (CLAUDE.md; `docs/DEVELOPER.md:134`): a test that seems wrong is never edited or deleted; it is marked and waits for the owner. Each mark is also described in PLAN under "Raised during the work" (lines 203 to 361). Every successor below was looked up and exists; none of them is marked, except where said.

### 1a. Superseded on every system, with a successor: 13

| # | Node id (line of the mark) | Reason in one line | Successor (line) |
|---|---|---|---|
| 1 | `tests/gui/test_controller.py::test_a_session_file_is_only_named_until_sessions_can_be_opened` (295) | C2: sessions open now; a missing .json gives a message | `tests/gui/test_session_saving.py::test_a_session_file_that_is_not_there_gives_a_message_and_names_only_the_file` (495) |
| 2 | `tests/gui/test_menus.py::test_the_menu_bar_has_file_and_help_with_the_specs_items` (36) | C6: File has six items now (Export) | `tests/gui/test_export_panel.py::test_the_file_menu_has_export_between_save_session_as_and_quit` (470) |
| 3 | `tests/gui/test_menus.py::test_quickstart_opens_the_repositorys_readme_in_the_browser` (161) | C8b: Help > Quickstart opens `#quickstart`, not `#readme` | `tests/gui/test_finish.py::test_help_quickstart_opens_the_quickstart_section_of_the_readme` (108) |
| 4 | `tests/gui/test_navigation.py::test_the_bar_is_laid_out_as_the_design_note_says` (78) | C8a: Play sits between −1 and +1 | `tests/gui/test_play.py::test_the_row_has_play_between_the_steps_back_and_the_steps_forward` (60) |
| 5 | `tests/gui/test_panel.py::test_rows_added_to_the_body_follow_the_hint_with_the_notes_spacing` (95) | C2: panel 1 has controls under the hint | `tests/gui/test_session_panels.py::test_a_row_added_to_a_body_follows_what_is_there_with_the_notes_spacing` (289) |
| 6 | `tests/gui/test_session_panels.py::test_open_video_and_open_session_are_the_file_menus_actions` (398) | C8a: File has Save session as now | `tests/gui/test_menus.py::test_open_video_and_open_session_are_the_file_menus_items_and_panel_1s_buttons` (55); the other successor named is row 2, itself marked |
| 7 | `tests/gui/test_shell.py::test_the_video_area_says_how_to_start` (178) | C1: the area is two labels around a button | `tests/gui/test_controller.py::test_the_empty_video_area_says_how_to_start_and_offers_open_video` (200) |
| 8 | `tests/gui/test_video_view.py::test_one_to_one_draws_one_video_pixel_on_one_screen_pixel_and_fit_goes_back` (142) | C3: the axes cross the marked block | `tests/gui/test_tools.py::test_one_to_one_draws_one_video_pixel_on_one_screen_pixel_beside_the_axes` (488) |
| 9 | `tests/test_tracking_job.py::test_a_fine_object_is_not_tracked_yet` (269) | A15 placeholder; A16 built fine mode | `tests/test_tracking_fine.py::test_a_fine_object_is_not_also_tracked_in_coarse_mode` (297) |
| 10 | `tests/test_tracking_fine_cases.py::test_an_empty_preview_mask_gives_a_96_px_window_and_the_run_still_completes` (162) | A16 review: the fallback window must not be stored | same file, `..._window_that_is_not_stored_and_the_run_still_completes` (206) |
| 11 | `tests/test_tracking_guard.py::test_hashes_made_by_another_decoder_are_not_compared_but_logged` (101) | asserts the opposite of decision X8 | same file, `test_hashes_made_by_another_decoder_are_made_anew_here_and_logged` (123) |
| 12 | `tests/test_export_pipeline_gaps.py::test_a_frame_without_a_record_is_written_as_a_lost_frame_is` (76) | A18b review: no lost rows in the Tracker-format file (SPEC 8.3) | same file, `..._is_a_lost_row_in_positions_and_shapes_and_no_row_in_the_tracker_file` (94) |
| 13 | `tests/test_export_pipeline_gaps.py::test_the_grid_of_the_lost_rows_is_the_clips_start_and_step` (126) | same cause as 12 | same file, `test_the_lost_rows_are_on_the_clips_grid_and_the_tracker_file_has_the_frames_with_a_record` (151) |

- Safe to delete once the owner says so. Rows 11, 12 and 13 carry an alternative in their reason text ("or amend X8"; "or say that the Tracker-format file should have the lost rows too"). Deleting them confirms X8 and SPEC 8.3 as built.
- After the deletion the docstring of `tests/test_export_pipeline_gaps.py:11-12` still describes the two marked tests, and the PLAN notes stay open until they are closed by hand.

### 1b. Superseded on some systems only, with a successor: 2

| Node id | Condition | Reason | Successor |
|---|---|---|---|
| `tests/gui/test_shell.py::test_the_dock_stays_at_the_right_and_is_400_px_wide` (101) | `sys.platform.startswith("linux")` | at 960 px the dock is 393 px in the Linux font, since the Play button | same file, `..._where_the_window_has_room` (123), at 1440 px |
| `tests/test_from_tracker_cli.py::test_folders_with_spaces_and_other_alphabets` (116) | `sys.platform != "darwin"` | a 1 px limit that was not derived; 1.1 and 1.3 px off on Linux and Windows | same file, `..._with_the_disk_clip` (154), 0.25 px |

So the fast suite expects 13 xfailed on macOS, 14 on Windows, 15 on Linux. CI has no macOS runner (`.github/workflows/tests.yml:24,43`), so the second test passes only on a developer's Mac.

### 1c. Real findings about the model (slow): 3 marks, 5 node ids

- `tests/slow/test_fine_mode.py::test_solidity_spectrum_peaks_within_half_a_hz_of_9_hz[cpu|mps]` (151) and `::test_solidity_is_within_0_02_rms_of_the_true_solidity[cpu|mps]` (156). Asked (SPEC 13.4): on the synthetic close-up shrimp, the solidity signal peaks at 9 ± 0.5 Hz and is within 0.02 RMS of the truth. Measured from one click on the body: the model outlines the body without the antennae; peak 0.56 Hz, RMS 0.4293, the same on cpu and mps (reason text `test_fine_mode.py:51-61`; VALIDATION 4.2, lines 311-322). With a click on each antenna, tried once and not a test: peak 9.00 Hz, RMS 0.0526, of which 0.052 is a constant offset (VALIDATION lines 338-346). Open question: is 0.02 meant for the absolute value or the variation (PLAN, note of Wed 06:45).
- `tests/slow/test_coarse_lowres.py::test_every_frame_with_a_mask_is_flagged_lowres[B]` (88). Asked: every frame with a mask of a 14.5 px body is `LOWRES`. Measured: B is lost on frames 6 to 34; on frame 36 its mask has 2 stray pixels, so `px_along_major` reads 134.8 px and the frame is not `LOWRES` (reason text `test_coarse_lowres.py:42-50`; VALIDATION 4.3). Open question: take the size from the largest piece of the mask (a change of SPEC 7.8).
- These are not superseded. They need a decision on the rule, not a deletion. They carry one laptop's numbers (PLAN:1222): with another torch or model version a strict mark can turn into an unexpected pass.

### 1d. Anything else

No other xfail. Two passing tests are superseded but unmarked: `tests/gui/test_theme.py::test_the_theme_draws_the_window_without_a_qt_message` (206, light and dark) and `::test_a_state_change_is_drawn_at_once` (275). Each has a successor right below it (233, 288) that reads the badge's fill as its most frequent colour. The old ones read one pixel (see 6); the ledger lists them for the tidy-up (ledger:386).

## 2. Skips and platform conditions

- `tests/test_fileio.py:251` `skipif(sys.platform != "win32")`: the real locked-file test runs on Windows only. This is the "1 skipped" of the recorded macOS run (ledger:463); the condition skips it on Linux too.
- `tests/test_geometry.py:659` `skipif(win32)` (POSIX permissions), and a runtime skip at `:668` when permissions are not enforced (a root user).
- `tests/test_provenance.py:30` `needs_git` on four tests; `:154` also needs a `.git` folder. A source archive without git skips them.
- Slow: `tests/slow/test_memory.py:92` `skipif(win32)`. Eight node ids skip without an Apple GPU: `test_fine_mode.py:105` (3), `test_real_model.py:289,478` (2), `test_gui_worker_real_model.py:105`, `test_regression_reference.py:424`, `test_selftest_real.py:69`.
- Platform branches inside tests, no skip: `tests/gui/test_badge.py:52` (a font fixture for Windows), `tests/gui/test_prompts.py:117-118` (Control and Command keys), `tests/gui/test_model_choice.py:115` (devices of this system), `tests/slow/memory_child.py:61` (units of `ru_maxrss`).
- Set-up for every run, root `conftest.py`: offscreen Qt (line 7), Windows fonts for the offscreen platform (13-14), torch imported before Qt on Windows (18-22), `pytest_plugins = ["helpers"]` (26).
- CI (`.github/workflows/tests.yml`): Ubuntu and Windows, `uv run pytest -m "not slow"` (lines 40, 70). No macOS job. Pushes that change only `**.md` or `docs/**` start no run (line 7), although tests read `README.md`, `docs/DEVELOPER.md` and `docs/OUTPUTS.md`.

## 3. The slow tests (`tests/slow/`, 34 node ids)

Every file sets `pytestmark = pytest.mark.slow`. CI never runs them (`tests.yml:3`, and the `-m "not slow"` steps). They need torch and transformers; most need the converted EdgeTAM in the local cache, `~/.cache/shrimp-models/edgetam` or `$SHRIMP_MODEL_CACHE/edgetam` (`outline_tracker/segmenter/edgetam_convert.py:82-85`). The first run downloads 56 MB (`test_regression_reference.py:10`). Times are from VALIDATION, one Apple-silicon laptop.

| File | Ids | Covers | Time (VALIDATION line) |
|---|---|---|---|
| `test_regression_reference.py` | 9 | new backend against the template's segmenter: 4 without weights, 4 on cpu, 1 on mps | 60 s; 96 s with the download (27) |
| `test_real_model.py` | 8 | several points, negative click, preview, fall back from the Apple GPU: 5 without weights, 3 with | 30 s (101) |
| `test_regression_pipeline.py` | 2 | `from_tracker` against the template's `track_video`, files within 0.01 px | 59 to 74 s (265) |
| `test_fine_mode.py` | 6 | fine mode on the close-up clip, cpu and mps; 4 ids xfail | 200 s cpu, 73 s mps (309) |
| `test_coarse_lowres.py` | 3 | `LOWRES` at dish scale; 1 id xfail | 36 s (378) |
| `test_memory.py` | 2 | 10 objects at 1080p under 3 GB and flat, in a child process (`memory_child.py`); 1 id needs no model | 232 to 245 s (420-423) |
| `test_selftest_real.py` | 2 | the `selftest` function on cpu and mps | 30 s (489) |
| `test_gui_worker_real_model.py` | 2 | the Track button's job through the window's worker | 26 s and 14 s (553) |

- The sum is about 13 minutes. A run of the whole folder was not timed after the last files were added: not verified.
- Ten ids need no weights. `test_memory.py::test_growth_is_not_moved_by_single_readings_and_sees_a_slow_leak` (64) is arithmetic only and says so in its docstring. CI installs torch (`tests.yml:37`), so these could run there; not verified that they pass on Linux or Windows.
- Long runs cancel pytest's 120 s stack dump with `pipeline_helpers.expect_minutes()` (`tests/slow/pipeline_helpers.py:30`; `pyproject.toml:53`).

## 4. Structure

- **Files over 400 lines: 21.** `test_hf_helpers.py` 802, `test_session.py` 770, `test_geometry.py` 751, `test_port_fidelity.py` 679, `test_schema.py` 639, `test_schema_docs.py` 639, `test_corrections.py` 577, `gui/test_session_panels.py` 547, `test_readme.py` 545, `slow/test_real_model.py` 527, `gui/test_session_saving.py` 523, `gui/test_tracking_panel.py` 523, `gui/test_model_choice.py` 509, `gui/test_tools.py` 507, `gui/test_export_panel.py` 499, `gui/test_worker_jobs.py` 495, `gui/test_review.py` 475, `test_synthetic.py` 467, `slow/test_regression_reference.py` 435, `gui/test_prompts.py` 409, `test_tracking_coarse.py` 407 (`wc -l`). The ledger lists several for a split (ledger:86, 281, 316, 419).
- **Helper modules and the number of files that import each** (grep of import lines): `tracking_helpers.py` 33, `helpers.py` 26 (also loaded as a plugin), `export_helpers.py` 16, `analytic_shapes.py` 11, `from_tracker_helpers.py` 7, `overlay_helpers.py` 7, `results_helpers.py` 7, `conftest.py` 5 (`java_sci`), `derive_helpers.py` 4, `qc_helpers.py` 3, `cli_probe_helpers.py` 3, `cli_compare_helpers.py` 2, `cli_export_helpers.py` 2. In `tests/gui/`: `gui_helpers.py` 30, `prompt_helpers.py` 18, `session_helpers.py` 16, `track_helpers.py` 14, `finish_helpers.py` 8, `calibration_helpers.py` 4, `last_controls_helpers.py` 3, `export_panel_helpers.py` 2, `review_helpers.py` 2. In `tests/slow/`: `pipeline_helpers.py` 4.
- **No conftest below `tests/`.** `tests/conftest.py` is the template's file and is pinned by its hash (`tests/test_repo_rules.py:84-86`), so shared fixtures live in `tests/helpers.py` (root `conftest.py:24-26`). GUI fixtures are therefore imported by name into test modules, with 16 `# noqa: F401` or `F811` lines (for example `tests/gui/test_badge.py:25`, `tests/gui/test_menus.py:18`).
- **Test modules import from test modules**, 11 lines: `test_tracker_io` gives `tracker_map` to five files (`tests/from_tracker_helpers.py:22`, `test_from_tracker.py:26`, `test_from_tracker_port.py:23`, `test_geometry.py:19`, `slow/test_regression_pipeline.py:33`); also `test_frame_source_jumps.py:21-22`, `test_frame_source_times.py:22`, `gui/test_badge.py:25`, `slow/test_regression_pipeline.py:32`, `test_measure_port_guard.py:17`. `test_tracker_io.py` cannot be changed while the fidelity test pins its text (see 5).
- **Duplicated helpers.**
  - The simulation of a locked file (a patch of `os.replace` that raises) is written in 11 files: `tests/test_fileio.py:34`, `test_session.py:444`, `test_results_file.py:66`, `test_overlay.py:230,250`, `test_tracking_job.py:198`, `test_corrections_gaps.py:259`, `test_export_tracker_folder.py:46,237`, `test_cli_probe.py:229`, `cli_export_helpers.py:69`, `gui/export_panel_helpers.py:101`, `gui/test_session_saving.py:310`. Each patches the one `os.replace` of the process for the length of its test (`fileio.os` is the `os` module); the ledger notes this for one of them (ledger:191).
  - The same name with the same body (docstring aside, compared by syntax tree) in several files, 23 cases. The largest: `silent` in 5 export test files, `run_folder` in 4, `tracked_folder` in 3 (`test_corrections.py`, `test_corrections_gaps.py`, `test_tracking_edit.py`), `error_line` in 3, `fit_of` in 3 (`gui/calibration_helpers.py`, `gui/prompt_helpers.py`, `gui/test_video_view.py`), `measure` in 3. `never_blocking` is a fixture in `gui/finish_helpers.py:64`, `gui/test_confirm.py:18` and `gui/test_controller.py:334`.
  - Analytic shapes exist twice: `tests/analytic_shapes.py` (175 lines, signed-distance functions: `disk`, `ellipse`, `capsule`, `body_with_rods`, ...) and `outline_tracker/synthetic_shapes.py` (176 lines, classes `Disk`, `Ellipse`, `Shrimp`, `Straight`, `Arc`). The ledger asks to join them (ledger:141). Whether the two give the same shapes was not compared: not verified.
  - QSettings are redirected twice: a session fixture (`tests/helpers.py:129`) and a per-test autouse fixture that does not restore the path (`tests/gui/session_helpers.py:25-32`).
- **Session-scoped fixtures.** Clips: `dish_clip`, `closeup_clip`, `disk_clip`, `shapes_clip`, `gapped_clip` (`tests/helpers.py:19-82`), `wide_clip` (`tests/gui/calibration_helpers.py:26`), `hd_scene_clip` (`tests/gui/session_helpers.py:35`). Rendering each took 0.04 to 0.10 s here, 0.5 s for all seven (timed once with `synthetic.render`). So the scope is about sharing one read-only file, not about time. `sequential` (`tests/test_frame_source.py:85`) keeps every decoded frame of two clips for the whole run, about 55 MB by arithmetic (240 frames of 320 x 240 x 3 bytes). `dish_run` (`tests/gui/review_helpers.py:74`) tracks the dish clip once with a stand-in; cost not measured. The fast tests have 29 module-scoped fixtures, many of which track a clip into a run folder (`git grep -n 'scope=' -- tests`); cost not measured.

## 5. Tests tied to the class's template

What is there:
- `tests/reference/`: ten files copied byte for byte from the class template (`tests/reference/README.md`), on the import path as package `shrimp` (`pyproject.toml:49`).
- `tests/test_port_fidelity.py` (679 lines, 70 node ids). It pins **source text**: every ported function or class equals the template's text (`VERBATIM`, lines 43-66), or differs only by listed replacements (`ADAPTED`, 136-168); constants by value (`CONSTANTS`, 172-177); methods (`VERBATIM_METHODS`, 181-190); one moved block (`MOVED_BLOCKS`, 195-206); ported test files that may differ in the import line only (`PORTED_TESTS`, `SPLIT_TESTS`, 210-249); three rewritten end-to-end tests (`REWRITTEN_TESTS`, 280-315); one copied helper (`COPIED_HELPERS`, 319-321).
- `tests/test_measure_port_guard.py` (8 ids): guards the ported test in `tests/test_measure.py`.
- `tests/test_port_equivalence.py` (7 ids): new and template functions side by side on 200 random inputs each: `fit_calibration`, `write_tracker_file`, `make_plan`, `read_tracker_export`, `mask_center` (lines 83-297).
- `tests/test_repo_rules.py::test_reference_copies_unmodified` (76) and `::test_template_tests_pass_on_reference` (88: the template's 27 tests in a subprocess, on every fast run).
- Single tests that call or read the template: `test_from_tracker_port.py::test_the_tracker_format_files_are_last_weeks_bytes` (110, 3 ids, bytes of the Tracker-format files); `test_overlay.py::test_a_frame_without_tracks_is_last_weeks_overlay_frame` (89, 3 ids); `test_export_tracker_folder.py:71,169` (reads the tool's files with the template's reader); `test_from_tracker.py:84` and `test_tracking_edit.py:73` (track colours); `test_fakes_threshold.py:54,73` (the stand-in against the template's `DiskFinder`); `test_cli_probe.py:104-106` (one message is in the template's script); `test_readme.py:293,299` (the README's table of options against the template's script). By name only, no call: `test_hf_helpers.py:385` pins the cache folder `shrimp-models` and `SHRIMP_MODEL_CACHE`.
- Slow: `test_regression_reference.py` runs the template's segmenter in `test_session_equals_the_one_the_reference_builds` (164), `test_per_object_logits_give_the_reference_masks` (197), the `selftest_runs` fixture (265) and `test_three_objects_match_the_reference` (364). `test_regression_pipeline.py` compares both of its tests with the template's `track_video` (`_track_both_ways`, 56).
- 24 collected test names contain `last_week`; 27 contain `template` or `reference`.

What goes when `tests/reference/` is removed and the wording is made general:
- Goes entirely: `tests/reference/`, `tests/test_port_fidelity.py`, `tests/test_measure_port_guard.py`, the two reference tests of `test_repo_rules.py` with `REFERENCE_SHA256` (27-38), the `shrimp` rows of the import scan (`test_repo_rules.py:22,135,158,166,171`), `pyproject.toml:47-50` (path and `norecursedirs`), `.gitattributes:2`, `test_cli_probe.py:104-106`, the two `DiskFinder` tests, CLAUDE.md's port rule and `docs/DEVELOPER.md:123-124`.
- Then `tests/conftest.py` is free to change. It can take the fixtures of `tests/helpers.py`, and `tests/gui/` can get its own conftest, which removes the imports of fixtures by name.
- Then the ported tests become ordinary tests: `test_video.py`, `test_convert.py`, `test_tracker_io.py`, the first test of `test_measure.py`, the first two of `test_from_tracker_port.py`, the first of `test_selftest.py`. Keep them. Their wording is the class's ("shrimp", "last week").
- Must change with the wording: `test_readme.py:293,299` (the "same as last week" table), the 55 quoted texts with "shrimp" in tests (for example `test_from_tracker_cli.py:99`, `test_from_tracker_fps.py:120`, `slow/test_selftest_real.py:56-58`), the fixture and scene names with `dish` (946 lines), `tests/gui/test_dish_panel.py`.

Value to keep in another form:
- **Numeric regression of the real model.** Today it is "equal to the template's code within 0.01 px" (VALIDATION 1.1, 4.1). Store the numbers instead: the tracked positions of the selftest clip (20 rows) and of the three-ellipse clip (60 rows), with the weights' SHA-256 (VALIDATION:17-19) and the library versions. They must be written **before** the reference is deleted, from a run in which both sides still agree. Whether the cpu numbers repeat on another machine or system is not verified; the limit may need to be per device.
- **Bytes of the Tracker-format file** with the stand-in model (3 cases, `test_from_tracker_port.py:82-106`): store three small golden files.
- **Behaviour of the calibration, the writer, the reader and the plan** (`test_port_equivalence.py`): rewrite as property tests against known maps and round trips. `test_tracker_io.py` already has such tests for single cases.
- **The track colours** (two tests): a literal list.
- **The selftest criterion** (3 px against the true centers, `test_regression_reference.py:305,418`; `slow/test_selftest_real.py`) does not depend on the template. Keep as it is.
- The checks without weights in `test_regression_reference.py:142-229` compare with the template's session and masks. Keep the half that checks the new backend against given logits; the comparison side goes.

## 6. Tests that depend on fonts, pixels, the screen or timing

- **One pixel of the badge.** `tests/gui/test_theme.py:224-226` and `:285` read `QPoint(4, 10)` of a 20 px badge and expect the fill, so the digit must not reach that pixel. This failed on the Windows CI machine until the set-up gave it fonts (ledger:378-386; root `conftest.py:9-14`). `tests/gui/test_badge.py` checks the same by geometry and is the sturdier form.
- **Pixels of panels that may be off screen.** The ledger notes that seven theme and badge tests read pixels of lower panels in a 600 px high window without scrolling (ledger:419). Which seven was not checked: not verified.
- **Other grabs of the drawn window:** `gui/gui_helpers.py:153` (`drawn`), `gui/calibration_helpers.py:108` (`lit`), `gui/test_fill.py:78`, `gui/test_fill_label.py:30`, `gui/test_calibration_panel.py:278`. `gui/test_video_view.py::test_the_click_test_passes_on_a_screen_scaled_to` (271) starts pytest inside pytest with `QT_SCALE_FACTOR` 1.5 and 2 and matches the summary line `3 passed`.
- **The bottom row at the smallest window (960 px).** `gui/test_shell.py:101` (the Linux xfail of 1b), `gui/test_navigation.py:106-119` (exact padding of each button from the font's metrics), `gui/test_play.py:92-106` and `:109-131`, `gui/test_status_bar.py:168-178`, `gui/test_tool_line.py:27`. The ledger warns that this row is nearly full and font-dependent (ledger:394, 433, 445-446).
- **Timing.** `docs/DEVELOPER.md:126` says no test sleeps. Exceptions: `threading.Timer(0.2, gate.open)` opens a gate from another thread while the GUI thread waits in `close()` (`gui/test_export_panel.py:326`, `gui/test_worker_jobs.py:157,397`); a stand-in check sleeps 0.2 s (`gui/test_session_panels.py:196`); mouse drags wait 20 ms per step because pyqtgraph drops faster moves (`gui/gui_helpers.py:185`). These are bounded waits, not races, as far as read; not run under load.
- **Measured time and memory** are asserted only in the slow tests (`slow/test_memory.py`: limits of 3 GB and 50 MB, with a known risk of a false alarm, VALIDATION 4.4 "Uncertain").

## 7. Rules as tests

`tests/test_repo_rules.py` (272 lines, 8 tests):
1. `test_version_flag` (62): `python -m outline_tracker.cli --version` prints `outline-tracker 0.1.0` and loads neither Qt nor torch.
2. `test_reference_copies_unmodified` (76): the ten template files and `tests/conftest.py` by SHA-256; no other `.py` in `tests/reference/`.
3. `test_template_tests_pass_on_reference` (88): exactly `27 passed`.
4. `test_import_boundaries` (143): Qt only under `gui/`; torch, torchvision, transformers, timm only under `segmenter/` and in `gui/app.py`; `shrimp` nowhere. The rule is keyed on the first path part (lines 130-131).
5. `test_import_scan_flags_each_rule` (150): the scan on a made-up package that breaks each rule once.
6. `test_core_does_not_load_torch` (207): importing every module outside `gui/` and `segmenter/` loads no heavy module, in a subprocess.
7. `test_no_personal_paths_or_big_files` (234): no tracked `.mp4 .mov .npz .pt .safetensors`; no home-folder path in any tracked text file.
8. `test_home_path_pattern_examples` (253): the pattern on examples.

Other rule tests:
- Import order: `tests/gui/test_import_order.py` (3 tests: torch before PySide6, pyqtgraph last; the gui package imports without torch and makes no Qt object; five module names at 128-129), `tests/gui/test_app.py` (2), `tests/gui/test_launch.py:37` (the script entry `outline_tracker.launch:main`) and `:118`.
- "Loads neither torch nor Qt" probes by module name, in subprocesses: `test_cli.py:178`, `test_cli_export.py:106,187`, `test_cli_compare.py:290`, `test_cli_probe.py:364`, `test_export.py:222`, `test_fakes.py:319`, `test_hf_helpers.py:48`, `test_schema.py:637`, `test_schema_docs.py:77`, `test_selftest.py:307`, `gui/test_about.py:80`.
- Pages: `tests/test_readme.py` (32 tests: commands parse with the real parser, options exist, example output, ten Quickstart steps, links, spelling), `tests/gui/test_finish.py:134-146` (DEVELOPER.md names every module path; required phrases), `tests/gui/test_finish.py` (every control the Quickstart marks is a real button or menu item), `tests/test_schema_docs.py` (38 tests; `docs/OUTPUTS.md` equals the generated text).
- File-size and layout rules: `tests/test_schema_docs.py:111` (two modules under 420 lines), `tests/gui/test_model_choice.py:414` (worker files under 400 lines and `_Engine` identity), `tests/gui/test_panel_modules.py` (the table of nine panel modules).

Must be rewritten when modules move into subpackages:
- Rules 4 and 6 above: the allowed places are path prefixes. A new model backend outside `segmenter/` fails rule 4, and its libraries must be added to `MODEL_MODULES` (`test_repo_rules.py:21`). Re-key these two tests first, as one table of "which subpackage may import what".
- `tests/gui/test_finish.py:137` (four literal module paths) and `docs/DEVELOPER.md` itself; `tests/gui/test_import_order.py:128-129`; the line-count tests; `test_schema_docs.py` tests of re-exports (`test_schema_re_exports_readme_text`, `test_every_public_name_of_schema_is_still_importable`); the `-m outline_tracker.<module>` calls (`cli` twice, `launch` once, `schema_docs` twice); `gui/test_model_choice.py:411` (a logger name).
- Scale of the plain edits: 440 import lines in 133 test files name `outline_tracker` modules; 18 distinct dotted module names appear as strings. 134 `monkeypatch.setattr` lines patch names on module objects (for example `fileio.os`, `video.check_video`, `menus.QDesktopServices`): after a move, a patch on a module that only re-exports the name silently does nothing. Each must be checked.
- Pinned lists that the planned features change: the two models in `gui/test_model_choice.py:113`, the File menu items (`gui/test_export_panel.py:470`), the nine panels, the ten Quickstart steps (`test_readme.py`).

## 8. Proposed order of cleanup steps (one sitting each)

1. **Get the owner's word on the 15 fast marks and delete those tests** (1a, 1b), with the three alternatives of rows 11 to 13 answered. Delete the two unmarked predecessors in `test_theme.py` (1d) in the same sitting if the owner agrees. Fix the texts that mention the marks. Goal: 0 xfailed in the fast suite on all three systems.
2. **Decide the two model findings** (1c): the size rule of SPEC 7.8 and the fine-mode criterion. Then rewrite those three tests to the decided rule. Run only `tests/slow/test_fine_mode.py` and `test_coarse_lowres.py`.
3. **Freeze reference numbers while the template is still there**: run the two slow regression files once, store the position tables, the three Tracker-format golden files and the weights' hash under `tests/data/`. Add tests that read them; keep the old comparisons until they agree.
4. **Remove the template coupling** (section 5, "goes entirely"), rewrite `test_port_equivalence.py` as property tests, replace the single uses of `reference`. Update CLAUDE.md and DEVELOPER.md in the same commit.
5. **Give the tests normal conftest files**: fold `tests/helpers.py` into `tests/conftest.py`, add `tests/gui/conftest.py`, drop the fixture imports and the `noqa` lines, end the test-to-test imports.
6. **Join the duplicated helpers**: one lock simulation as a fixture, then `silent`, `run_folder`, `tracked_folder`, `error_line`, `fit_of`, `never_blocking`; decide between `tests/analytic_shapes.py` and `outline_tracker/synthetic_shapes.py`.
7. **Harden the fragile GUI tests**: replace the timer threads by an event the window raises when it starts to wait; decide whether 960 px stays the smallest window and test the row by rule, not by font; correct `docs/DEVELOPER.md:126`.
8. **Re-key the rule tests for the new layout before any module moves** (section 7), then move modules one subpackage per sitting, checking every `monkeypatch` target.
9. **Split the test files over 400 lines** along the new module layout (not before step 8, or they are split twice).
10. **Rename the class's words in tests together with the app's renames** (fixture names with `dish`, the quoted texts, `SHRIMP_MODEL_CACHE`), one concept per sitting.
11. **CI**: add a macOS job, run the ten no-weights slow tests in CI (move the arithmetic one to the fast suite), let documentation-only pushes run the page tests, and consider a scheduled slow job with the weights cached.
