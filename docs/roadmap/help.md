# Roadmap inventory: help inside the app (tooltips, "?" help, terms)

> Inventory for `docs/ROADMAP.md`: help inside the app. Read-only findings at commit `f73d29f` (2026-10-07).
> File and line pointers are true for that commit. "The ledger", "the build ledger" and "the
> scratch folder" are private working notes of the first build; they are not in the repository.
> The design note of the window is `docs/design/gui_design.md`.

Question: what can a user read in the window about each control, and what is missing.

How it was checked. I read every file named below. I also ran a scratch script (session scratch folder, `help_inventory/walk.py`). It builds `MainWindow(segmenter_factory=None)` on Qt's `offscreen` platform, never shows it, opens no video, loads no model (torch is not imported), and keeps the settings in the scratch folder. It walks the widget tree and prints `toolTip()`, `statusTip()`, `whatsThis()` and the accessible name of each widget. Numbers marked "walk" come from it, in the state "no video open". Nothing was looked at on a screen: how a text looks is not verified.

Short names (all under `outline_tracker/gui/`): mw = main_window.py, nav = navigation.py, p1 = panels/video_panel.py, p2 = panels/time_panel.py, p3 = panels/calibration_panel.py, p4 = panels/dish_panel.py, p6 = panels/objects_panel.py, p7 = panels/track_panel.py, p8 = panels/review_panel.py, p9 = panels/export_panel.py, sw = stopwatch_dialog.py, rt = review_table.py. Design note = `docs/design/gui_design.md`.

## 1. Every control, and what explains it

### 1.1 Counts

| kind | in all | explained in the window | not explained |
|---|---|---|---|
| Widgets the user operates (buttons, check boxes, combo boxes, number boxes, text fields, slider, tables), main window and Stopwatch dialog (walk) | 76 | 66 have a tooltip | 10: the 9 panel title buttons and Cancel of the Stopwatch dialog |
| Menu items (menus.py:40-52) | 8 | 0 have a tooltip that describes them | 8 |
| Keys (1 to 9 counted once, up and down counted once) | 14 | 11 | 3: Esc, Ctrl/Cmd+Z, 1 to 9 |
| Table columns (p6:43, rt:46) | 9 | 0 header tooltips | 9 |
| Combo box entries (mode 2, model 2, device 3) | 7 | 0 entry tooltips; 1 tooltip per box | 7 |
| Mouse gestures (list in 1.4) | 10 | 5 | 5 |

- Status tips: 0 of 76 widgets, 0 of 8 menu items. What's This texts: 0. Accessible names and descriptions: 0 (walk). A grep for `setStatusTip`, `setWhatsThis`, `setAccessible` in `outline_tracker/` finds nothing.
- `setToolTip(` is called on 49 lines in `outline_tracker/gui`; the texts are literals or module constants next to the widgets. There is no table of help texts.
- Only 4 widgets have an object name (mw:111, mw:127, mw:144, nav:113). A help table keyed by control needs a name for each.
- The 76 leave out the 3 pop-up lists that Qt makes inside the combo boxes. The walk also found a QAction "Export..." that is pyqtgraph's own, not the app's. Whether a user can reach it is not verified (the view's menu is off: video_view.py:39).

### 1.2 Controls outside the panels

| control | tooltip (exact text) | pointer | other text that explains it |
|---|---|---|---|
| Open video (start page) | "Choose the video to track" | mw:178 | "Open a video to start." mw:42 |
| Fill (button that stays down) | "Fill each tracked outline with its color, so that you see what it covers" | mw:197 | none |
| Fit / 1:1 | "Show the whole frame" / "One video pixel per screen pixel" | mw:199-200 | none |
| Slider | "Move through the frames of the clip" | nav:123 | none |
| First, −10, −1, +1, +10, Last | "First frame of the clip (Home)", "Back 10 steps (Shift+←)", "Back 1 step (←)", "Forward 1 step (→)", "Forward 10 steps (Shift+→)", "Last frame of the clip (End)" | nav:44-51 | none. "step" is not defined here |
| Play | "Play or pause (Space)" | nav:36 | none |
| Frame box | "Go to a frame. A frame that is not on the frame grid goes to the nearest one." | nav:161 | label "Frame" (nav:164), no tooltip |
| 9 panel titles (click opens or closes) | none | panel.py:51-62, 92 | only the arrow shows it |
| Stopwatch: 2 frame boxes | "First moment: the number of a frame that shows the stopwatch" (and "Second …") | sw:71 | dialog text sw:34-35 |
| Stopwatch: 2 time boxes | "First moment: the time the stopwatch shows on that frame, in s" | sw:75 | same |
| Stopwatch: Use frame shown (×2) | "Take the number of the frame that the video view shows" | sw:39, 76 | none |
| Stopwatch: OK / Cancel | "Use this frame rate as fps_true" / none | sw:93-94 | none |

Read-outs with a tooltip (not controls): the tool line (its own text again, mw:194, 297), the zoom (mw:204), t (nav:167: "Time of this frame: frame / fps_true"), Pixel, Position, Gray, Device in the status bar (status_bar.py:33-36), the model state (its tooltip is the worker's last message, p6:325), the two run folder labels (the whole path, p1:364-365, p9:340-341). Read-outs without one: the panel state word and number badge (panel.py:46, 64), Scale, Measured, Center, Radius, RMS, 2R, Origin (`value_label`, p3:141-148), Source (p2:83), the progress lines (p7:137, p9:165).

### 1.3 Controls in the panels

"Hint" = does the panel's first hint line (mw:50-60) name the control.

| panel | control | tooltip (exact text, or its start) | pointer | hint |
|---|---|---|---|---|
| 1 | Name (text field) | "Your name. The run folder is named after you." | p1:220 | yes |
| 1 | Open video / Open session | "Choose the video to track" / "Choose the session.json of a run folder, to go on with it" | p1:223-224 | yes / no |
| 1 | Start frame, End frame, Step | "First frame of the clip", "Last frame of the clip", "Every how many frames one is tracked" | p1:232 | no (the done hint shows the values, p1:51) |
| 1 | "The file says" (read-out) | "The frame rate written in the file. Times come from fps_true (panel 2), never from this." | p1:229 | no |
| 2 | fps_true (text field, placeholder "frames per second") | "The true frame rate of the recording in frames per second (fps_true). The time of a frame is its number divided by fps_true." | p2:80-82 | yes |
| 2 | Stopwatch… / Choose manifest | "Measure fps_true from two frames that show a stopwatch" / "Read fps_true from a manifest.csv file that lists this video" | p2:87, 90 | yes / no |
| 3 | Stick / Tape (tool buttons) | "Click the two ends of a known length on the ruler" / "Check the scale on two other ruler marks" | p3:210-211 | yes / no |
| 3 | Redo (×2), also panel 4 | "Remove the points and click them again." | p3:36, 212; p4:45 | no |
| 3 | Length / True distance | "The true length between the two ends of the stick" / "The true distance between the two ruler marks" | p3:215-216 | yes / no |
| 4 | Circle / Axes | "Click points on the inner wall of the dish" / "Click where the origin of the axes goes" | p4:44, 51 | yes / no |
| 4 | Origin to Center | "Put the origin at the center of the circle" | p4:52 | no |
| 4 | Dish diameter / Angle | "The dish's inner diameter, if you know it: 2R should be close to it" / "The direction of +x: counterclockwise from the picture's right" | p4:49, 53 | no |
| 4 | Crop to dish for tracking (check box) | "Track only inside the square around the dish: faster, and sharper on the animals" | p4:54-55 | no |
| 5 | none | – | – | the hint names "Add a box" and "Measure" (mw:55), which do not exist |
| 6 | Objects table (click a row) | "The objects to track. Click a row to choose the object that gets the next point." | p6:107 | no |
| 6 | Mode (combo: coarse, fine) | "coarse: tracked with the others on the frame. fine: tracked alone, in a window that follows the object." | p6:111-113 | no |
| 6 | Fine window (number box) / Auto | "Side of the window of a fine object. auto: three times the object's length." / "Let the window follow from the object's length again" | p6:121, 124 | no |
| 6 | Positive, Negative, Head (tool buttons) | "Click on the animal: a positive point", "Click on what is not the animal: a negative point (also a right click)", "Click on the head of the animal, on the frame its track starts on" | p6:58-60 | no |
| 6 | Add / Remove | "Add an object. Then click on the animal in the video." / "Remove the selected object, with its points and its results" | p6:61, 162 | yes / no |
| 7 | Model / Device (combos) | "The model that finds the outlines. EdgeTAM is the default." / "What the model runs on. auto takes the graphics processor if it works, else the processor (cpu)." | p7:70-71 | no |
| 7 | Track / Cancel | "Track the objects that have points (panel 6)" / "Stop after the current frame. The tracked frames are kept." | p7:72-73 | the hint is the estimate, or why Track is off (p7:312-336) |
| 8 | Flags table (click a row; up, down) | "The flags of the tracked frames. Click a row to go to its frame (up and down: the row before, the next)." Each cell: "CODE: meaning" | rt:54, 112-113 | yes |
| 8 | Show shape flags too / Only the selected object | "Also list LOWRES, ORIENT and HEADGUESS: they are on nearly every frame of a small object" / "List the flags of the object selected in panel 6 only" | p8:101-105 | no |
| 8 | Next flag / Previous flag | "Go to the next flag of the selected object" / "Go to the previous flag of the selected object" | p8:64-65 | no |
| 8 | Re-track from here, End track here, Continue as new track | "Track the selected object again from this frame to the end of the clip", "Remove the results of the selected object after this frame", "Add the next track of this animal, starting on this frame" | p8:66-68 | no |
| 9 | Change folder | "Go on in another run folder: the session and the results are copied there (File > Save session as)" | p9:58 | no |
| 9 | Export all / Open folder | "Write the CSV files, the overlay video, the log and README.txt to the run folder" / "Show the run folder in the system's file browser" | p9:56-57 | yes / no |

Panel 5 has no controls: `panels/__init__.py:19-20` names `probes_panel`, the file does not exist, and `build_bodies` skips it (`panels/__init__.py:34-35`). Task D1 is open (docs/PLAN.md:1648).

### 1.4 Menu items, keys, mouse

- Menu items: File > Open video, Open session, Save session, Save session as, Export, Quit; Help > Quickstart, About (menus.py:40-62). None has a status tip or a What's This text (walk). Three get a tooltip that is their own name, or "Tracking is running." during a run (menus.py:92-93). Quickstart opens a fixed GitHub address in the browser (menus.py:28, 107-110): it needs the internet and names one repository. About lists versions and model folders (about.py:27-40). No item leads to the docs or to a list of keys.
- Keys explained in the window: ←, →, Shift+←, Shift+→, Home, End and Space in the tooltips of their buttons (nav:36, 44-51); Open, Save and Quit beside their menu items (menus.py:41, 44, 49; how each system shows them is not verified); up and down in the flags table's tooltip (rt:54).
- Keys with no text in the window: Esc returns to Pan (video_view.py:83-85); Ctrl/Cmd+Z takes back the last point (prompts.py:132-134; README.md:85 says it); 1 to 9 select an object (mw:164-167). Esc and 1 to 9 are in no user document (grep of README.md).
- Mouse, explained: drag moves the picture and the wheel zooms (the Pan line, mw:44, shown only while no tool is chosen; precise tools add "Scroll to zoom in for a precise click." under 100%, tool_items.py:39, 198); a click places a point (each tool's line: prompts.py:51-52, tools.py:176-177, 216-217, 288-289, 381-382); a right click is a negative point (only in the Negative button's tooltip, p6:59); a click on a table row (p6:107, rt:54).
- Mouse, not explained anywhere: Alt/Option-click and Control-click are negative points (click_rules.py:47-60); the second click of a double click is dropped (click_rules.py:73-94, tool_items.py:200-206); a third click with Stick or Tape starts again (tools.py:95-117); the wheel changes a number box only while it has the keyboard (p1:130-144, p3:106-110); a click on a panel title opens or closes it (panel.py:92).
- How to leave a tool is not said. Circle and the three point tools stay chosen after a click (tools.py:331-335, prompts.py:197-218). The ways out are Esc or a second click on the tool's button, and the Pan line that would say so is replaced by the tool's line (mw:293-297).
- Dialog buttons (Close, Cancel, and the action's name: dialogs.py:43, 64-65) are not counted. Their dialog text explains them.

### 1.5 Findings

1. Tooltips are nearly complete (66 of 76), one sentence each, as the design note asks (§6, "Tooltips: every control has one sentence, with its key in brackets"). They are the only help layer.
2. A control that is off loses its description. Its tooltip becomes the reason: p1:360, p6:309 and 314-315, p7:374 and 376, p8:384 and 389, p9:344 and 360, menus.py:93. Before a video is open, 11 controls show only a reason (walk): Add, Model, Device, Track, Next flag, Previous flag, the three corrections, Export all, Open folder. That is the moment a newcomer explores.
3. A tooltip needs a mouse that rests on the control. Keyboard and touch users get nothing. No widget has an accessible name (walk), so a screen reader reads a number box as a bare number.
4. One sentence is too short for the hard controls: Mode, Fine window, Step, Model, the corrections, the flags. There is no place for a longer text.
5. Hint lines say the next step of the work, not what each control means. Of the 46 controls inside the panels (titles not counted), the start hints name 10 (column "hint" above).
6. Table headers, combo box entries and the object status words "no prompts, ready, tracked, ended" (p6:67-75) have no explanation at all.
7. Panel 5's hint describes controls that are not there.
8. Full stops are mixed: most tooltips have none, some have (p3:36, nav:161, p2:81-82). The design note's rule (§8.3) covers buttons, labels, hints and messages but not tooltips.

## 2. Terms a newcomer does not know

| term | where it shows without an explanation | what explains it today | missing |
|---|---|---|---|
| fps_true | field label (p2:100), Stopwatch result (sw:36-37), t tooltip (nav:167), export refusal (p9:63), warning (p2:43) | field tooltip (p2:81-82), hint (mw:52) | why the file's frame rate is not the true one; "manifest" is a course word (p2:88-90) |
| clip | tooltips of Start, End, slider, First, Last (p1:232, nav:44-51, 123) | nothing | a definition: start frame, end frame, step |
| step | "Step" label (p1:259), "Back 1 step" (nav:46-49) | "Every how many frames one is tracked" (p1:232) | that the buttons and the slider move by it; that times do not shift (SPEC.md §3.3) |
| frame grid, grid frame | frame box tooltip (nav:161) | nothing; a moved click says "is not a frame of the clip" (tracking_edit.py:211-212) | a definition (SPEC.md §3.4), and one word for it |
| coarse, fine | Mode entries and table cells (p6:111, 349) | Mode tooltip (p6:112-113) | when to choose fine; cost (a run of its own, estimate.py:6-7); the mode cannot change after tracking (p6:54-55). README says "close-up window" for the same thing (README.md:106) |
| fine window | "Fine window", "Window" column, "auto" (p6:43, 117, 134) | tooltip (p6:121) | the dotted square on the picture is this window (prompts.py:315-320) |
| the nine flags | codes in the table (rt:111), in a tooltip (p8:102-103) | cell tooltip and the line under the table, for the chosen row (rt:65-67, 112-113; p8:377) | the conditions and limits (schema.py:172-192 has them; the window shows only `meaning`); what to do; no list of all nine in the window |
| flags on positions, shape flags | hints and a check box (p8:47-49, 101) | check box tooltip, in part | the two groups (rt:49-50) and why three are hidden by default |
| shape_ok | not in the window (grep); in shapes.csv and in LOWRES's condition (schema.py:186) | docs/OUTPUTS.md | a link from LOWRES to it |
| positive, negative point | tool buttons, tool lines (p6:58-59, prompts.py:51-52) | tooltips | when to add a second positive point (only in a later hint, p6:52-53); the Alt and Control clicks |
| head click | Head button (p6:60) | tooltip says where, not why | that it sets the heading's side, is optional, and removes HEADGUESS (README.md:86, schema.py:191) |
| Fill | button (mw:195) | tooltip (mw:197) | enough |
| dish crop | check box (p4:54) | tooltip (p4:55) | that it needs the circle; that EDGE then means the crop's border (schema.py:181-182) |
| tape check, "Passes" | group label and result (p3:226, 288-300) | Tape tooltip (p3:211), note (p3:37) | the 1% limit shows only after the check (p3:38); what to do when it does not pass shows only then |
| scale uncertainty "± 0.08%" | Scale value (p3:271) | nothing | where it comes from (the click's uncertainty, tools.py:172-174) |
| RMS, 2R, Radius | labels (p4:59-60) | "2R should be close to it" (p4:49) | what RMS measures and what value is good |
| run folder | labels (p1:260, p9:176), many messages | "The run folder is named after you." (p1:220) | what it is and where it is (README.md:71); the tooltip gives only the path |
| session | menu items, Open session (menus.py:42-45) | Open session tooltip (p1:224) | what a session holds; that it saves by itself (README.md:89) |
| re-track, end track, continue as new track | buttons (p8:113-116) | tooltips (p8:66-68), the questions before the first two (p8:59-62), "The new track is A2." (p8:57) | when to use which; that the tooltips vanish while the buttons are off |
| object status words | Status column (p6:67-75) | nothing | all four |
| model, device | combos (p7:68-69), status bar | tooltips (p7:70-71, status_bar.py:36) | how the two models differ; that a change loads again and forgets the timing (p7:208) |
| s/frame, time left (ETA), the estimate | progress line (estimate.py:73-74), hint (p7:74-75) | "time left (ETA)" | where the estimate comes from (p7:34-37); how to get one when there is none (click on an object) |

## 3. What Qt offers, and what fits

Verified in the project's environment: PySide6 6.11.2 (Essentials only, pyproject.toml:19), with `QWhatsThis` (`createAction`, `showText`, `enterWhatsThisMode`), `QToolTip`, `QTextBrowser.setMarkdown`; no web view (`QtWebEngineWidgets` is not installed). The style is Fusion (theme.py:168).

| mechanism | effort | how it reads for a novice | test without pixels |
|---|---|---|---|
| Tooltips (`setToolTip`, `ToolTipRole` for table headers and combo entries) | small: fill 10 gaps, add headers and entries, keep the description while a control is off | known to everyone; hover only, one sentence, goes away | walk the tree: every interactive widget has a tooltip that starts with the table's short text |
| Status tips (`setStatusTip`) | small | does not fit: Qt shows a status tip with the status bar's message and clears it on leaving, and this window keeps its last message there until the next one (status_bar.py:5-6, design note §7 "Status line") | – |
| What's This (`setWhatsThis`, Shift+F1, `QWhatsThis.createAction()` in Help) | small once the table exists: one loop | holds a long text and works from the keyboard; the mode "click ?, then click a control" is little known, least on a Mac. Whether it answers on a control that is off is not verified | every interactive widget and menu item has `whatsThis()`; the Help item puts `QWhatsThis.inWhatsThisMode()` on |
| Small "?" buttons that open a popover | medium: one button class, one popover (`QWhatsThis.showText`, or a `QFrame` with `Qt.Popup`), one per panel header (panel.py:69-74 has the row) and one beside about 8 hard labels | best: visible, a click, stays open, room for 3 to 6 sentences and a link. The dock is 340 px at its smallest with a 120 px label column (mw:37, p1:42): room for a 20 px button | each `Panel` has a help button whose key is in the table; a click calls one function `help.show(key, anchor)` that a test replaces, as tests replace the dialogs (tests/gui/gui_helpers.py, `record_dialogs`); a second test reads the real popover's text |
| Help area that follows the focused or hovered control | medium to large: a strip in the dock or under the video, focus and hover plumbing (`QApplication.focusChanged` is already used, nav:154), a way to hide it | no click needed; costs height in a dock that already scrolls (design note §10 point 4); text that changes under the mouse distracts | give each widget the focus; the area's text equals the table's long text |
| Help menu items | small to medium: "Controls and terms" and "Keys and mouse" in a `QTextBrowser` dialog (works offline), "What's This?", Quickstart from package metadata, not a literal (pyproject.toml has no `[project.urls]` yet) | the place people look first; today it has 2 items | each item exists and is connected; the page shown equals the generated docs page |

Fit: tooltips (complete, and kept while off) plus "?" popovers plus Help menu pages, with What's This as the keyboard path for the same long texts. No status tips. A following help area only if the "?" buttons prove too few.

Design note rules a help system should follow:
- §8.1: sentences of 15 words at most, present tense, active; no contractions, idioms, "please", "invalid", "failed".
- §8.2: one word for one thing. The term list there is the glossary's word list. The rename to a general tool (Dish to Boundary, animal to object) then happens in one place.
- §8.3, §8.5, §8.6: sentence case; a number has its unit; a panel is named "panel 3 (Calibration)"; a file by its name, never a path.
- §8.7: a flag code is in capitals and its meaning shows wherever the code shows, in the words of SPEC.md §9.
- §6: one sentence per tooltip with its key in brackets; a control that is off keeps its place and says why.
- §5: the hint line stays, one text per state. Help adds to it and never replaces it.
- §7: the last message stays in the status bar; inline messages stay where they are.
- Two rules need a ruling from the owner. §6 "Icons: … no symbol used as an icon" speaks against a "?" button. §7 "A dialog is allowed only for …" has no case for a help page or a popover.

## 4. Proposal

One source of truth: a table without Qt, `outline_tracker/help_texts.py`, outside `gui` so that the docs and the command line can use it (CLAUDE.md: nothing outside `gui` imports Qt). One entry per control, menu item, key, gesture, table column and combo entry: `key` (the widget's object name, for example `time.fps_true`), `label`, `kind`, `short` (one sentence: the tooltip), `long` (2 to 6 sentences: What's This, the popover, the docs), `terms` (links into a glossary in the same module), `panel`. The flags are not copied: their entries are built from `schema.FLAG_INFO` (schema.py:159-194), condition and meaning both.

Pieces to build:
1. The table and the glossary (section 2 is the list of terms; sections 1.2 to 1.4 are the list of keys).
2. An object name on every control. The "Parts:" lists in the panels' docstrings are the names (for example p6:83-86).
3. `gui/help.py`: `apply(window)` sets tooltip, What's This text and accessible name from the table after `panels.build_bodies` (mw:168); `tip(key, reason)` gives "what it is. why it is off." for the places of finding 2 (six files); the "?" button and its popover; `show(key, anchor)`.
4. Help menu: Controls and terms, Keys and mouse, What's This?, Quickstart, About.
5. A generated page `docs/CONTROLS.md`, made from the table as `docs/OUTPUTS.md` is made from the schema (schema_docs.py:365, 408).
6. Remove the 49 literal `setToolTip` calls as each panel moves to the table.

Tests that keep it complete (written first; expected values come from the table and SPEC.md, not from the running window):
- Completeness: build the window with a stand-in and walk the tree. An interactive widget (`QAbstractButton`, `QAbstractSpinBox`, `QLineEdit`, `QComboBox`, `QSlider`, `QAbstractItemView`; not the inner parts of a combo or spin box) or a menu item fails the test without an object name in the table, a tooltip that starts with `short`, and a What's This text. The same for the Stopwatch dialog, each table header and each combo entry. The scratch walk shows the walk is about 30 lines and finds 76 widgets and 8 menu items today.
- No orphans: every entry of the table is found in the tree. Every `QShortcut` of the window (19 today, walk) has a key entry, and the entry's key text is derived from the shortcut, not typed.
- Off state: without a video, and during a run, every control that is off still starts its tooltip with `short` and ends with the reason.
- Wording (no Qt): §8 of the design note as a lint over the table: sentence length, the banned words, panel references that match `mw.PANELS` (mw:50-60), every term that a text links exists, every glossary term is linked at least once.
- Docs: `docs/CONTROLS.md` equals the generator's output, as tests/test_schema_docs.py:598-602 does for `docs/OUTPUTS.md`; the page the Help menu shows is the same text.

Open questions for the owner: the two rulings in section 3; whether the "?" buttons sit only on panel headers or also beside hard labels; whether Quickstart should open an offline page.
