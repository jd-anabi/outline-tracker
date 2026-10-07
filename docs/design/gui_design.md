# Outline Tracker: GUI design note (tasks C0 to C8, D1 to D12)

> The design note that the window of version 0.1.0 was built from (2026-10-07): spacing, type, color
> tokens, track colors, panel anatomy, buttons, feedback, wording and the video view. Where it and the
> code differ, the code is what was shipped; `docs/ROADMAP.md` (workstream W4, decision 21) lists the
> two rules that a help system needs changed.

Scope: how the window looks and reads. SPEC §10 fixes what is in it (video view, right dock with panels 1 to 9, bottom bar, status bar, menus File and Help, tool modes, keys). Nothing here renames, reorders, merges or drops any of that. Sources: SPEC §4 to 6, §9, §10; PLAN §4 and §7; the impeccable and ui-ux-pro-max plugins as far as they carry over to Qt widgets.

Checked for this note: every text pair with the WCAG formula; the track colours with a colour-blind simulation (Machado 2009, CIEDE2000); the QSS skeleton and the video overlays rendered offscreen with PySide6 6.11.2 and pyqtgraph 0.14.0 (Fusion, light and dark, scale 1, 1.5 and 2). Not checked: a real window on Windows 11 or macOS. All sizes are Qt's device-independent px; Qt 6 scales them at 150 % and 200 %, so never multiply by the device pixel ratio.

## 1. Layout metrics

Spacing scale (px): **2, 4, 8, 12, 16, 24**. No other values.

| step | use | step | use |
|---|---|---|---|
| 2 | slider to flag strip | 12 | panel padding (left, right, bottom); label column to field; between status-bar items |
| 4 | between the frame buttons; icon to its text | 16 | between two groups in one panel (Stick, Tape check); in front of a destructive button |
| 8 | the default: rows in a panel, buttons in a row, between panels, dock margins, bottom-bar padding | 24 | dialog margins (Stopwatch…, About) |

| element | value |
|---|---|
| Window | minimum 960 × 600; opens maximized when the screen's available area is under 1440 × 900, else 1440 × 900 centered. At 1280 × 720 (a 1920 × 1080 laptop at 150 %) the video view is about 874 × 484, so a 16:9 frame shows at 860 × 484 |
| Central area | top to bottom: view bar 32 (tool text at the left; Fill, Fit, 1:1 and the zoom % at the right), video view (all the rest), bottom bar 76. Status bar 24 |
| Dock (right) | default width 400, minimum 340, maximum 520; resizable by its edge; not closable, not floating, not movable, no title bar. Inside: one `QScrollArea` (`widgetResizable`, no horizontal bar, vertical bar always on so the width never jumps), margins 8, 8 between panels |
| Panel | 1 px border, radius 6; header row 32 high; body padding 12 (left, right, bottom); 8 between rows; label column 120 wide, fields fill the rest |
| Controls | height 28 (buttons, fields, spin boxes, combo boxes; `setMinimumHeight(28)` on the Fusion-drawn ones); primary button 32; table and check-box rows 24. Corner radius 4 (buttons, messages), 6 (panels), 10 (the number badge, a 20 px circle) |
| Tables | Objects shows 4 rows, Flags shows 6 rows; fixed height; they scroll inside |
| Bottom bar | 76 = 8 + slider 20 + 2 + flag strip 6 (P1; keep the space from C1 on) + 4 + button row 28 + 8 |
| Button row | First, −10, −1, Play/Pause, +1, +10, Last (text buttons, 28 high, 4 apart), the flags read-out (§9; takes the free width), the "Frame" box (96 wide), "t = 1.234 s" |

- **What scrolls:** only the dock's panel column, the two tables and the log. Never the window, the video view (it zooms and pans) or the bottom bar. No horizontal scroll bar anywhere: long paths are elided in the middle, full text in the tooltip.
- **Wheel guard:** spin boxes and combo boxes in the dock react to the wheel only when they have keyboard focus (`Qt.StrongFocus` plus an event filter that ignores `QEvent.Wheel` otherwise). Scrolling the dock must never change a value.

## 2. Type

Base size B = `QApplication.font()` (Segoe UI 9 pt = 12 px on Windows, the system font at 13 pt on macOS). Never set a family or a px size; set `pointSizeF` relative to B. Nothing is smaller than B.

| role | size, weight | colour | notes |
|---|---|---|---|
| Panel title | B + 1 pt, DemiBold | text | sentence case |
| Group label in a panel (Stick, Tape check); primary button | B, DemiBold | text; onAccent | other buttons: B, Normal |
| Field label | B, Normal | text | no colon; the unit sits with the number, not in the label |
| Value (a result: scale, R, error) | B, DemiBold | text | tabular digits, right-aligned |
| Hint line, table header | B, Normal | muted | hints wrap; two lines at most at width 400 |
| Frame box and time in the bottom bar | B + 1 pt, DemiBold | text | tabular digits |
| Status line | B, Normal | read-outs muted, last message text | tabular digits, fixed widths |
| Log | the system fixed font at B (`QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)`) | text | the only monospace text |

- **Tabular digits:** one helper `numeric_font(font)` calls `font.setFeature(QFont.Tag("tnum"), 1)`. Use it wherever a number changes while the student watches: status read-outs, frame, time, progress line, values, numeric table columns, spin boxes. (The macOS system font needs it; Segoe UI digits already have one width.) Every live read-out also has a fixed minimum width from its longest text (`fontMetrics().horizontalAdvance("0000.0, 0000.0 px")`), so nothing moves when digits change.
- **Alignment:** numbers right-aligned (tables, values, spin boxes); text left-aligned; a column header is aligned like its column. Centered: only the badge number and the frame buttons.
- **Number formats:** px 1 decimal; mm 2; µm/px 2; s 3; fps 2; s/frame 2; degrees 1 with "°"; percent as the spec writes it ("0.08%", no space); frames are integers without group separators; minus is "−" (U+2212). `QLocale.setDefault(QLocale.c())` at startup, so the decimal sign is "." as in the CSV files.

## 3. Colour tokens

`app.setStyle("Fusion")` on both systems: it draws the same on Windows and macOS and follows the palette. `gui/theme.py` holds the two token tables, builds one `QPalette` and one QSS string, and applies them at startup and on `app.styleHints().colorSchemeChanged` (dark when `colorScheme()` is `Qt.ColorScheme.Dark`, else light). No Qt object at import time. **From the system:** only the light or dark choice, the font, native file dialogs and Qt's standard icons. **Fixed:** every colour below; the system accent is not used, because a user-chosen accent cannot be checked for contrast. Windows contrast themes are not handled in v0.1.

| token | light | dark | use (contrast: light and dark) |
|---|---|---|---|
| window | `#F1F3F5` | `#1E2126` | window, dock background, status bar, zebra rows |
| panel | `#FFFFFF` | `#272B31` | panel surface, secondary button, tooltip |
| field | `#FFFFFF` | `#1A1D21` | inputs, tables, log |
| control | `#F7F8FA` | `#30353C` | palette `Button`: spin-box arrows, combo boxes, table headers |
| hover / pressed | `#E8EBEF` / `#DCE0E6` | `#30353C` / `#3A4048` | button states |
| border | `#D0D5DC` | `#3A4048` | panel outline, separators (decoration only) |
| borderStrong | `#7C8591` | `#7D8793` | outline of buttons and of the badge (3.7:1 and 3.9:1 on panel) |
| text | `#1B1F24` | `#E7EAEE` | all normal text (16.6:1 and 11.8:1 on panel; 14.9:1 and 13.4:1 on window) |
| muted | `#57606A` | `#A3ADB8` | hints, table headers, read-outs (6.4:1 and 6.3:1 on panel; 5.7:1 and 7.1:1 on window) |
| disabled | `#8C959F` | `#6E7781` | disabled text (3.0:1 and 3.1:1 on panel; disabled text is exempt from AA) |
| accent | `#1F5FBF` | `#7AB0FF` | primary button, selection, focus ring, progress, quiet-button text (6.1:1 and 6.4:1 on panel) |
| onAccent | `#FFFFFF` | `#0D1B2E` | text on accent (6.1:1 and 7.8:1) |
| accentHover / accentPressed | `#1A52A6` / `#164690` | `#94C0FF` / `#5F9BEE` | primary button states (onAccent stays at 6.1:1 or more) |
| accentSoft | `#DCE8FA` | `#253B5C` | checked tool button, table rows of the current frame (text on it 13.4:1 and 9.4:1) |
| success / successBg | `#17753A` / `#E3F4E7` | `#5FD080` / `#1C3324` | done, a check passes (5.8:1 and 7.3:1 on panel; 5.0:1 and 7.0:1 on its Bg) |
| warning / warningBg | `#8A5A00` / `#FFF3D1` | `#F2BD4B` / `#3A2E12` | needs attention (5.9:1 and 8.2:1 on panel; 5.4:1 and 7.7:1 on its Bg) |
| problem / problemBg | `#B42318` / `#FDE7E4` | `#FF8E84` / `#40211F` | problems, destructive buttons (6.6:1 and 6.4:1 on panel; 5.6:1 and 6.5:1 on its Bg) |
| canvas / onCanvas | `#1B1D20` / `#E7EAEE` | the same | around the video, in both themes (14.0:1) |

- Every text pair is 4.5:1 or more (WCAG AA); message text is `text` on the tint (11:1 or more). Colour is never the only cue: a state always has an icon or a shape, and a word (§5, §7).
- Palette roles: Window = window; WindowText, Text, ButtonText = text; Base = field; AlternateBase = window; Button = control; Highlight, Accent, Link = accent; HighlightedText = onAccent; ToolTipBase = panel; ToolTipText = text; PlaceholderText = muted; the three text roles of the Disabled group = disabled.

## 4. Track colours

One list for the GUI, `session.json` and `overlay.mp4`. It is core data: define it once outside `gui/` (suggested `schema.TRACK_COLORS`) and import it. A stays yellow and the hue order of last week is kept (PLAN X14; see §10). `overlay.py` (built, tested) draws 1 px lines without a casing; the GUI tasks do not change it.

| # | id | hex | name | # | id | hex | name |
|---|---|---|---|---|---|---|---|
| 1 | A | `#FFFF00` | yellow | 6 | F | `#23EBFA` | cyan |
| 2 | B | `#D71EA5` | magenta | 7 | G | `#FA9B8C` | pink |
| 3 | C | `#D3FFB7` | light green | 8 | H | `#CD4B46` | red |
| 4 | D | `#5AB4F5` | sky blue | 9 | I | `#2F6FFF` | blue |
| 5 | E | `#E66E00` | orange | 10 | J | `#F0B946` | gold |

Smallest difference between any two (CIEDE2000): 20.5 with normal vision, 14.4 protan, 12.4 deutan, 7.3 tritan. Last week's list has 4.1 (yellow and green; green and light green) and 4.7 (magenta and azure). Black text on every colour is 4.6:1 or more. Against mid-gray video the luminance contrast of magenta, red, blue and orange is as low as 1.1:1, so the casing below is required.

Drawing on the video (pyqtgraph pens are cosmetic: widths are screen px at any zoom):
- **Outline:** first a casing, black at alpha 180, width 4; on it the track colour, width 2, solid. Antialiasing on.
- **Fill** (the view bar's Fill box; off by default): the track colour at alpha 64 inside the outline; the outline stays.
- **Centroid:** a 6 px dot in the colour with a 1 px black edge. **Head mark:** a 9 px diamond, filled when the head was clicked, edge only when it is a guess (`HEADGUESS`).
- **Id label:** black DemiBold text on a box in the track colour with a 1 px black border, 6 px above and right of the outline: `pg.TextItem(id, color="#000000", fill=colour, border="#000000", anchor=(0, 1))`. In the object table: a 14 × 14 swatch with a 1 px borderStrong edge next to the id text.
- **Selected track:** colour width 3 on a casing of width 5; fill alpha 96 when Fill is on; the label box gets a 2 px white border. Other tracks do not change.
- **Lost on this frame** (`LOST`, no outline exists): a dotted ring (`Qt.DotLine`, width 2, diameter 24, with casing) at the last known centroid and the label "A · LOST".
- **11th track and later:** colour = list[(n − 1) mod 10]; tracks 11 to 20 are dashed (`Qt.DashLine`), 21 to 30 dash-dot. n counts letters: a piece (A2, A3) has the colour of its letter. The id label is always shown.

## 5. Panel anatomy

- `Panel(QFrame)` has `role="panel"` and `state` = `todo`, `done` or `attention`. Header row, left to right: number badge `QLabel[role="badge"]`; header `QToolButton[role="panelHeader"]` (checkable, takes the free width, text = title, `setArrowType(Qt.RightArrow)` when collapsed, `Qt.DownArrow` when expanded); state word `QLabel[role="state"]`.
- **State = shape and word, not only colour.** todo: outlined badge, "Not started" in muted (panel 5: "Optional"). done: badge filled with success, "Done". attention: badge with the warning tint and edge, panel edge in warning, "Needs attention" in DemiBold.
- **Collapsed:** the header row only. **Expanded:** header and body. No animation. At start the first panel that is not done is expanded and the others are collapsed. A click on a header toggles that panel only. Nothing opens or closes by itself later.
- **Body order:** hint line, controls, inline message (if any), button row (primary at the left, then secondary; quiet and destructive at the right after a stretch).
- **Hint line:** the first row of the body, `QLabel[role="hint"]`, always there, one text per state (§8). It also gives the reason when the main button is disabled.
- A dynamic property needs a re-polish: `w.setProperty(name, value); w.style().unpolish(w); w.style().polish(w)` (one helper). QSS has no variables: `theme.py` replaces each `@name` below with the hex of the active theme.

```css
#Dock { background: @window; }
QFrame[role="panel"] { background: @panel; border: 1px solid @border; border-radius: 6px; }
QFrame[role="panel"][state="attention"] { border-color: @warning; }
QToolButton[role="panelHeader"] { border: none; background: transparent; color: @text; font-weight: 600; text-align: left; padding: 6px 4px; }
QToolButton[role="panelHeader"]:hover { background: @hover; border-radius: 4px; }
QToolButton[role="panelHeader"]:focus { border: 2px solid @accent; border-radius: 4px; padding: 4px 2px; }
QLabel[role="badge"] { min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px; border-radius: 10px; border: 1px solid @borderStrong; color: @text; font-weight: 600; }
QLabel[role="badge"][state="done"] { background: @success; border-color: @success; color: @panel; }
QLabel[role="badge"][state="attention"] { background: @warningBg; border-color: @warning; color: @warning; }
QLabel[role="state"], QLabel[role="hint"] { color: @muted; }
QLabel[role="state"][state="done"] { color: @success; }
QLabel[role="state"][state="attention"] { color: @warning; font-weight: 600; }
QLabel[role="value"] { font-weight: 600; }
QFrame[role="msg"] { border-radius: 4px; border: 1px solid @border; background: @window; }
QFrame[role="msg"][kind="success"] { background: @successBg; border-color: @success; }
QFrame[role="msg"][kind="warning"] { background: @warningBg; border-color: @warning; }
QFrame[role="msg"][kind="problem"] { background: @problemBg; border-color: @problem; }
QFrame[role="msg"] QLabel { background: transparent; border: none; color: @text; }
QPushButton { min-height: 26px; padding: 0 12px; border: 1px solid @borderStrong; border-radius: 4px; background: @panel; color: @text; }  /* 26 + border = 28 */
QPushButton:hover { background: @hover; }
QPushButton:pressed { background: @pressed; }
QPushButton:checked { background: @accentSoft; border: 2px solid @accent; padding: 0 11px; }
QPushButton:focus { border: 2px solid @accent; padding: 0 11px; }
QPushButton[kind="primary"] { min-height: 30px; background: @accent; border-color: @accent; color: @onAccent; font-weight: 600; }
QPushButton[kind="primary"]:hover { background: @accentHover; border-color: @accentHover; }
QPushButton[kind="primary"]:pressed { background: @accentPressed; border-color: @accentPressed; }
QPushButton[kind="primary"]:focus { border: 2px solid @text; }
QPushButton[kind="quiet"] { background: transparent; border-color: transparent; color: @accent; }
QPushButton[kind="quiet"]:hover { background: @hover; }
QPushButton[kind="quiet"]:focus { border: 2px solid @accent; }
QPushButton[kind="destructive"] { border-color: @problem; color: @problem; }
QPushButton[kind="destructive"]:hover { background: @problemBg; }
QPushButton:disabled, QPushButton[kind]:disabled { background: @window; border-color: @border; color: @disabled; }  /* keep this rule last */
QAbstractSpinBox[check="error"], QLineEdit[check="error"] { background-color: @problemBg; }
QAbstractSpinBox[check="warn"], QLineEdit[check="warn"] { background-color: @warningBg; }
```

## 6. Buttons and inputs

| kind (`kind` property) | look | used for |
|---|---|---|
| primary | accent fill, onAccent text, DemiBold, 32 high | the next step of the panel; one per panel at most, and only while that step is open |
| secondary (no property) | panel fill, borderStrong edge, 28 high | everything else, and all tool buttons |
| quiet | no edge, accent text | Redo, Show details |
| destructive | problem text and edge | Remove, End track here; at the right, 16 px from other buttons |

- **Never primary:** Re-track from here, End track here, Remove, Cancel, and Export all once the run folder holds an export (then Open folder is the primary). Track is primary but slow: no button in the main window reacts to Enter (`setAutoDefault(False)`); in a dialog the default button is the safe one.
- **Tool buttons** (Stick, Tape, Circle, Axes, Probe, Positive, Negative, Head): checkable, one checked at most in the whole window; none checked = Pan. Checked look: accentSoft fill, 2 px accent edge.
- **States:** hover = `hover` fill; pressed = `pressed` fill; keyboard focus = 2 px accent edge (on a primary button: 2 px `text` edge), with 1 px less padding so the size stays; disabled = window fill, `border` edge, `disabled` text. A disabled button keeps its place, and the hint line says why. While its action runs a button is disabled, and the progress line says what runs.
- **Inputs are drawn by Fusion from the palette.** No QSS border on `QLineEdit`, spin boxes, combo boxes, check boxes, sliders, progress bars or tables: with a QSS border Qt draws the arrows itself and they come out wrong without images (tested). Known limit: Fusion's field outline has about 2.2:1 (light) and 1.2:1 (dark) against the panel (measured), under the 3:1 guideline; so every field has a visible label at its left.
- **Spin boxes with units:** the unit is the suffix (`setSuffix(" mm")`, also " px", " s", " fps", "°"), text right-aligned, `setKeyboardTracking(False)`, decimals as in §2, minimum width 104.
- **Out of range:** the widget's own range is only a wide hard limit; it must never change a typed value silently. A value that cannot be used (stick length 0 mm, end frame before start frame, fps_true 0 or less, step 0) is checked on Enter or on leaving the field: set `check="error"` (problemBg fill), put a problem message right under the field ("The stick length must be more than 0 mm."), set the panel to attention, disable its main button.
- **Soft limit** (fps_true under 100 fps, stick under 300 px, a typed frame that is not on the frame grid): `check="warn"` and a warning message; the action stays possible.
- **Combo texts:** Model: "EdgeTAM", "SAM 2.1 tiny". Device: "auto", "cpu", "mps (Apple GPU)", "cuda (NVIDIA GPU)". **Tooltips:** every control has one sentence, with its key in brackets: "Back 1 step (←)", "Play or pause (Space)", "Show the whole frame", "One video pixel per screen pixel".
- **Icons:** only `QStyle.standardIcon`, 16 px, only in messages. Every button has text: no icon-only button, no emoji, no symbol used as an icon.

## 7. Feedback

- **Progress (panel 7).** Before a run, the hint line gives the estimate. During a run: a `QProgressBar` from 0 to the frames of all runs, no text in the bar; under it one line in tabular digits: "Run 1 of 2 · frame 240 of 600 · 0.42 s/frame · time left (ETA) 2 min 30 s", updated at most 4 times a second. Track stays where it is, disabled; it never turns into Cancel. **Cancel** is a secondary button beside it, enabled only during a run; after a click it is disabled and the line reads "Stopping after the current frame."
- **No invented progress.** While the model loads or is downloaded, the bar is in busy mode (`setRange(0, 0)`) with a text line. The preview shows `Qt.BusyCursor` over the video and no bar. Export uses the same bar and line in panel 9.
- **Status line** (`QStatusBar`): at the left a 16 px icon and the last message; it stays until the next message, no timeout. At the right, permanent, in this order: "Pixel 1024.5, 512.5 px", "Position 12.34, −5.67 mm", "Gray 128", "Device cpu", "Model ready" (or "Model loading"). Cursor outside the picture: "Pixel –". No scale yet: "Position – (no scale)".
- **Log area:** in panel 7 under the progress line. A quiet button "Show details" / "Hide details" opens a read-only `QPlainTextEdit` in the fixed font, 6 lines high, 500 lines at most, newest at the bottom. It shows the `log(msg)` lines of runs and exports. The full log is `run.log` in the run folder. Never a stack trace.
- **Inline message:** `QFrame[role="msg"]` with `kind`, holding a 16 px icon and a wrapped text label; margins 8, 6, 8, 6. Success: `SP_DialogApplyButton`. Warning: `SP_MessageBoxWarning`. Problem: `SP_MessageBoxCritical`. It sits under the control it is about, else above the button row; it stays until its cause changes; one per panel (problem before warning before success); the same text goes to the status line.
- **A dialog is allowed only for:** (1) an action the student asked for that could not start or was stopped (a video or session cannot be opened, tracking stopped by an error, a file cannot be written): the title says what happened, the text says what to do next, one button "Close"; (2) a confirmation before tracked results are deleted or replaced (Re-track from here, End track here, Remove for an object with results): title "Re-track A from frame 412?", text "The results of A from frame 412 to the end are replaced. Other tracks do not change.", buttons named by the action ("Re-track from here", "Keep results"), Enter and Esc take the safe one; (3) what the spec names: file dialogs, Stopwatch…, About, Quickstart. Never a dialog for success, a warning, a field error, progress or flags. All dialogs go through the plan's one helper.

## 8. Wording

1. Short sentences (15 words at most), present tense, active voice. No contractions ("cannot", "do not"), no idioms, no humour, no exclamation marks, no "please", "sorry", "oops", "successfully", "invalid", "failed".
2. One word for one thing: **video** (the file), **clip** (start frame, end frame, step), **frame**, **step**, **scale**, **stick**, **tape check**, **dish**, **circle**, **origin**, **axes**, **probe** (a box), **object** (an animal you clicked), **track** (its outline and position over the frames: A, B, A2), **point** (positive, negative, head), **outline** (not "mask"), **flag**, **run folder**, **session**, **model**, **device**. The object table keeps the spec's status words: no prompts, ready, tracked, ended.
3. Sentence case. No colon after a label. No full stop on buttons and labels; full stops in hints and messages.
4. A button is a verb and says what happens. The spec's labels are used letter for letter: Open video, Convert for tracking, Stopwatch…, Stick, Tape, Redo, Circle, Origin to Center, Crop to dish for tracking, Measure, Add, Remove, Positive, Negative, Head, Track, Cancel, Re-track from here, End track here, Continue as new track, Export all, Open folder, Fit, 1:1. "…" only where the spec writes it. Under "Continue as new track" a line names the id: "The new track is A2." The tooltip of Redo: "Remove the points and click them again."
5. A number always has its unit (px, mm, s, fps, s/frame, %).
6. A message says what happened, then what to do next. It names a panel by number and title: "panel 3 (Calibration)". It shows a file name, never a long path or a code.
7. Flag codes stay in capitals as in the CSV files. Wherever a code shows, its meaning shows too (tooltip, and a line under the Flags table), in the words of SPEC §9, column "meaning for the student", with "outline" for "mask".

| # | title | main button | hint: not started | hint: done | hint: needs attention |
|---|---|---|---|---|---|
| 1 | Student and video | Open video | Type your name. Then open your video (a _tracker.mp4 file). | {file} · {W} × {H} px · {N} frames · clip {a} to {b}, step {k}. | This video has {n} warnings. Read them below before you continue. |
| 2 | Time | Stopwatch… (secondary) | Type the true frame rate (fps_true), or measure it with the stopwatch. | fps_true = 240.00 fps (from the manifest). | fps_true is under 100 fps. This usually means a re-timed copy of the video. Check the value. |
| 3 | Calibration | Stick, then Tape | Click Stick. Click the two ends of a known length on the ruler. Type the length. | Scale: 32.40 µm/px ± 0.08%. Tape check: 0.4%, passes (limit 1%). | The tape check does not pass: 1.8% (limit 1%). Click Redo and place the stick again. |
| 4 | Dish and axes | Circle, then Origin to Center | Click Circle. Click 6 or more points on the inner wall of the dish, spread around it. | Dish: R = 16.20 mm (500.0 px), RMS 0.6 px. The origin is at the center. | The circle needs 3 or more points. Click more points on the dish wall. |
| 5 | Probes | Add box, then Measure | Optional. Add a box over the LED. Then click Measure. | {n} boxes are measured on {N} frames. | Box {name} has one corner. Click the opposite corner in the video. |
| 6 | Objects | Add | Click Add. Then click on one animal in the video. An outline appears. | {n} objects are ready to track. | Object {id} has no points. Click on the animal, or remove the object. |
| 7 | Track | Track | Estimated time: about {m} min for {N} frames and {n} objects. | Tracking is complete: {n} objects, {N} frames. | Tracking was cancelled at frame {k}. The tracked frames were kept. |
| 8 | Review and fix | Next flag | Track first. The flags appear here. | No flags on positions. Play the video once and look at the outlines. | {n} flags on {m} tracks. Click a row to go to its frame. |
| 9 | Export | Export all | Export all writes the CSV files, the overlay video, the log and README.txt to the run folder. | Exported at {hh:mm}. Export all replaces these files. | {file} could not be written. See the message below. |

- Sources in panel 2, in the spec's words: "from the manifest", "from the stopwatch", "typed in". Reasons for a disabled Track, in the hint line: "Type your name in panel 1 first." "Add an object in panel 6 first." "The model is loading. Track is ready when the model is ready."
- State rules: 1 done = name and video; 2 = fps_true set; 3 = scale set (attention: stick under 300 px, or the tape check over 1%); 4 = circle fitted or origin placed; 5 = measured; 6 = every object has a positive point; 7 = every object tracked to the clip end; 8 = no position flag (attention: any `LOST`, `JUMP`, `SIZE`, `CONTACT`, `EDGE`); 9 = exported after the last change.

| case | kind and place | text |
|---|---|---|
| Video cannot be opened | problem, dialog | Title: "The video could not be opened". Text: "{file} could not be read. Check that the file is on this computer, not only in a cloud folder, and that it is a _tracker.mp4 file. For a .MOV file, run “outline-tracker convert” on it first." |
| No scale set | warning, panel 9 (Export all disabled) | "No scale is set. Place the stick in panel 3 (Calibration). Export needs the scale to write positions in mm." |
| Name missing | problem, under the name field | "Type your name in panel 1 first. The run folder is named after you." |
| Model is being downloaded | busy bar and line, panel 7 | "The model is being downloaded ({size} MB). This is needed once and uses the internet. You can do panels 1 to 6 while you wait." |
| Tracking cancelled | warning, panel 7 | "Tracking was cancelled at frame {k}. The {n} tracked frames were kept. To track the rest, go to frame {k}, click on each animal, then click Re-track from here (panel 8)." |
| A file is open in another program | problem, dialog | Title: "A file could not be written". Text: "{file} is open in another program, for example Excel. Close it there. Then click Export all again." |
| An object was lost | warning, panel 8, after a run | "Object {id} was lost on {N} frames, first at frame {k}. Go to that frame. If you can see the animal, click on it and click Re-track from here." |
| Tape check (the spec's green or red) | success or problem, panel 3 | "Tape check: 20.08 mm measured, 20.00 mm true, error 0.4%. Passes (limit 1%)." / "… error 1.8%. Does not pass (limit 1%)." |

The GUI checks these cases before it calls the core. Any other core error is shown with the core's own message in the dialog of §7 (1); the trace goes to `run.log`.

## 9. Video view

| mode | cursor | a click | text at the left of the view bar |
|---|---|---|---|
| Pan (default; Esc returns here) | `OpenHandCursor`; `ClosedHandCursor` during a drag | nothing | "Pan: drag to move the picture. Scroll to zoom." |
| Stick, Tape | `CrossCursor` | places an end | "Stick: click the first end on the ruler." then "Stick: click the second end." (Tape: "ruler mark" for "end") |
| Circle | `CrossCursor` | adds a wall point | "Circle: click points on the inner wall of the dish. {n} placed; 6 or more is best." |
| Axes | `CrossCursor` | places the origin | "Axes: click the new origin." |
| Probe | `CrossCursor` | places a corner | "Probe: click one corner of the box." then "Probe: click the opposite corner." |
| Positive, Negative, Head | `CrossCursor` | adds a point to the selected object | "Positive: click on animal {id}." "Negative: click on what is not animal {id}." "Head: click on the head of {id}." |
| a point tool while the preview runs | `BusyCursor` | is kept; the newest preview wins | "The outline is being updated." |
| a point tool while tracking runs | `ForbiddenCursor` | nothing | "Wait until tracking has stopped." |

- **In every mode** a drag moves the picture and the wheel zooms at the cursor, so the student can zoom in without leaving a tool (pyqtgraph already tells a click from a drag). A click outside the picture does nothing. A right click is a negative point in the three point tools (spec) and nothing elsewhere. Over a draggable handle (D3): `SizeAllCursor`.
- **Zoom and pan hints:** the view bar shows the zoom ("100%"), **Fit**, **1:1** and the **Fill** box at the right. Above 100 % pixels are not smoothed. For Stick, Tape and Circle the tool text ends with "Scroll to zoom in for a precise click." while the zoom is under 100 %.
- **No video yet:** centered on the canvas in onCanvas: "Open a video to start." with a primary button "Open video", and under it "Then follow panels 1 to 9 at the right."
- **Tool graphics:** white `#FFFFFF`, width 2, on the black casing (width 4, alpha 180); no track is white. Stick solid, tape dashed, both with `crosshair` ends (15 px); circle points as 7 px hollow squares, the fitted circle solid at width 1.5; axes as two arrows labelled "x" and "y"; a probe box solid with its name; the fine-mode window dotted in the track colour. Tool labels ("Stick 30.00 mm", "Tape 20.08 mm", "R = 16.20 mm", "LED1"): white text on black at alpha 180 (8.7:1 on any video).
- **Points** (`pg.ScatterPlotItem`, `pxMode=True`, 1.5 px black edge), told apart by shape: **positive** = a 12 px disc `o` in the track colour with a black 7 px `+` on it; **negative** = a 14 px cross `x` in the track colour; **head** = a 13 px white diamond `d`. Never green for positive and red for negative.
- **Flags of the current frame**, in three places. (1) On the video: a track with a position flag on this frame (`LOST`, `JUMP`, `SIZE`, `CONTACT`, `EDGE`) gets it in its label: "A · CONTACT". Shape flags (`MULTI`, `LOWRES`, `ORIENT`, `HEADGUESS`) are never drawn on the video. (2) In the bottom bar's button row: "Flags: A CONTACT · C LOST" with the warning icon, or "Flags: none" in muted; for the selected track its shape flags follow in muted. (3) In the Flags table the rows of the current frame have the accentSoft fill.
- **Flag strip (D4):** position flags of the selected track as marks 6 px high in `warning`; `MULTI` and `ORIENT` 3 px high in `muted`; `LOWRES` and `HEADGUESS` are not drawn (they are on nearly every frame).

## 10. Conflicts and open points

Where SPEC §10 goes against plugin advice (the spec wins; what softens it):
1. **"Redo" clears tool points.** Plugins: a button says what it does, and "Redo" usually means "do again what was undone". Kept; a tooltip explains it.
2. **Tape error in green or red.** Plugins: never red and green as the only cue. Kept; the words "Passes" or "Does not pass" and an icon are added.
3. **Error dialogs (§10.2).** Plugins: a modal is the last choice, and errors sit next to the field. Kept for errors that stop an action; field errors, warnings and success are inline (§7).
4. **Nine panels in one scrolling dock, with tables inside.** Plugins: no nested scroll areas, show less at once. Kept; panels collapse, tables have a fixed height, the wheel guard protects values.
5. **Right click is the negative point, and the context menu is off.** Plugins: no hidden gesture as the only way; keep platform behaviour. Kept; the Negative tool button is the visible way.
6. **Spec labels that are not plain words or not sentence case:** fps_true, ETA, s/frame, 1:1, mps, cuda, Origin to Center, the flag codes, and no "…" on Open video. Kept letter for letter; a tooltip or a few added words explain each ("time left (ETA)", "mps (Apple GPU)").
7. **Remove, End track here and Re-track from here cannot be undone** (undo beyond prompts is P2). Plugins: prefer undo to a confirmation. Kept; a confirmation dialog with the safe button as default.

Not a conflict: the numbers on the panels (the plugins allow them when the order carries information; here it is the workflow). Plugin advice not carried over to Qt: 16 px body text and 44 px targets (web and touch; here the system font and 28 px controls), custom fonts, icon sets, toasts, skeleton loaders, motion.

Questions a designer would still ask, with the default used here:
1. **Track colours against PLAN X14** ("colours follow last week's overlay"). Last week's list has pairs that look the same with red-green colour blindness. Default: the list of §4 (A stays yellow, same hue order), used by the GUI, `session.json` and `overlay.mp4`.
2. **Title of panel 1.** The spec lists its content but gives no title. Default: "Student and video".
3. **Colour of a piece (A2, A3).** Default: the colour of its letter, so one animal keeps one colour.
4. **Shape flags in the Flags table.** `LOWRES` and `HEADGUESS` sit on nearly every frame of a dish-scale track (PLAN X22) and would bury the rest. Default: the table shows position flags; a check box "Show shape flags too" (off) adds the others.
5. **Export all a second time.** It replaces the exported files. Default: no confirmation dialog; the button is no longer primary and the hint line says that the files are replaced.
