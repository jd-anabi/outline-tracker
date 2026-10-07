# Outline Tracker

Outline Tracker follows animals in a video. For each animal it finds the outline in every frame, with the model EdgeTAM. It writes the positions and the shapes to files that your analysis notebook can read.

You need `uv` and `git`; you have both from last week. You do not need to install Python. It runs on Windows 11 and on Macs with an Apple chip. Intel Macs are not supported.

The tool is an app with a window: the [Quickstart](#quickstart) below takes you through it in ten steps. It also has commands without a window. They are your fallback if the app does not open. You type every command of this page in the terminal: on a Mac, use Terminal (zsh); on Windows, use PowerShell. The commands are the same on both.

## Install

```shell
uv tool install git+https://github.com/jd-anabi/outline-tracker
```

The first install takes a few minutes. Then check that it worked:

```shell
outline-tracker --version
```

You see one line like this. The commit is different on your computer:

```text
outline-tracker 0.1.0 (commit 4a1c9e7)
```

To get a newer version, install again with `--force`:

```shell
uv tool install --force git+https://github.com/jd-anabi/outline-tracker
```

To remove the tool:

```shell
uv tool uninstall outline-tracker
```

## Check the installation

```shell
outline-tracker selftest
```

This tracks a made-up shrimp in a made-up video and compares the result with the truth. It takes about a minute. The first time it also downloads the model, about 56 MB, so you need the internet. If you ran last week's selftest on this laptop, most of the model is already there, but you still need the internet once for one small file.

The last two lines are the result. Your numbers are different:

```text
OK: edgetam followed the test shrimp within 0.4 pixels (should be under 3). 2.00 s per frame here.
Estimate for 10 s at step 2 (1,200 frames): one shrimp about 40 min; 10 shrimp together about 220 min.
```

- `OK:` means the installation works. The estimate tells you how long a run takes on your laptop.
- `PROBLEM:` means the tool lost the test shrimp, or was 3 pixels or more away from it. Run the test once more on the processor (below). If it says `PROBLEM:` again, send the whole text of the window to your instructor.
- `ERROR:` means the test could not run at all. See Troubleshooting below.

```shell
outline-tracker selftest --device cpu
```

## Quickstart

The window shows your video in the middle. At the right are nine numbered panels: work through them from top to bottom. Each panel has one line of text that says what to do next. To open the window, type this in the terminal:

```shell
outline-tracker
```

1. Start the app with the command outline-tracker, as shown above. The window opens after a few seconds, and the model loads while you do the first panels.
2. In panel 1 (**Student and video**), type your name, then click **Open video** and choose the \_tracker.mp4 copy of your video. Your name is part of the name of the run folder, where all your files go.
3. In panel 2 (**Time**), check the true frame rate (fps_true): the app reads it from your manifest if it finds one. If the field is empty, type the value, or measure it with **Stopwatch…**.
4. In panel 3 (**Calibration**), click **Stick**, click the two ends of a known length on the ruler, and type the length. Zoom in with the mouse wheel before you click, and check the scale on two other ruler marks with **Tape**: the error must be under 1%.
5. In panel 4 (**Dish and axes**), click **Circle** and click 6 or more points on the inner wall of the dish, spread around it. Then click **Origin to Center**, or place the origin yourself with **Axes**.
6. In panel 6 (**Objects**), click **Add**, click on one animal, and check that the outline follows it: do this for each animal. If the antennae matter, click on each antenna too, because one click on the body leaves them out.
7. In panel 7 (**Track**), read the estimated time, then click **Track**. You can look at other frames while it runs, and **Cancel** stops it and keeps the frames that are tracked.
8. In panel 8 (**Review and fix**), look at the flags: each flag names a track and a frame where something may be wrong. Click a row to go to its frame: if the outline is on the wrong animal, click on the right one, then click **Re-track from here**.
9. In panel 9 (**Export**), click **Export all**. It writes the CSV files, the overlay video, the log and README.txt to the run folder.
10. Your files are in the run folder, next to your video: **Open folder** in panel 9 shows it. The parts "Where the files are" and "Load the tracks in your notebook" below say what each file holds and how to read the tracks.

More about step 6:

- Click well inside the animal. To add a point to an object, click its row in the table, then click **Positive**.
- If an outline also takes in a neighbor, click on the neighbor with **Negative** (or with a right click). Keep that point away from your positive point: a negative point very near a positive one can empty the outline.
- Ctrl+Z (Cmd+Z on a Mac) takes back the last point. **Remove** deletes the selected object. If it has tracked frames, the app asks first.
- **Head** marks the head of the animal, so that the heading points the right way. It is optional.
- **Fill**, above the video, paints each tracked outline in its color, so that you see what it covers.

Your work is saved by itself in the run folder. To go on another day, start the app, click **Open session** and choose the file session.json in your run folder. If a run is slow, look at the estimate in panel 7 before you start: a smaller clip (the start frame and the end frame in panel 1) or a larger step takes less time.

## If the app does not open

This is last week's work with this week's files. You need the same two things as last week: the `..._tracker.mp4` copy of the video that you opened in Tracker, and the file that you exported from Tracker (File > Export > Data) with the columns frame, x, y, pixelx and pixely. It holds one track, or one point mass for each animal, marked on the same frame.

Last week you ran `python -m shrimp.segment VIDEO EXPORT`. This week you run `outline-tracker from-tracker VIDEO EXPORT`. You do not need the long `uv run --with` start any more.

| What you want | Last week | This week |
|---|---|---|
| Start a run | `python -m shrimp.segment VIDEO EXPORT` | `outline-tracker from-tracker VIDEO EXPORT` |
| Track only the first S seconds | `--seconds S` | `--seconds S` |
| Use every K-th frame | `--step K` | `--step K` |
| Give the real frame rate | `--fps F` | `--fps F` |
| Choose the processor | `--device D` | `--device D` |
| Choose the folder for the files | `--out FOLDER` | `--out DIR` |
| Do not write the overlay video | `--no-video` | `--no-overlay` |
| Track in a close-up window | not there | `--fine A,C` |
| Put your name in the folder name | not there | `--student sam` |

Open the terminal in your repository folder, the one that holds `data/manifest.csv`. The real frame rate (fps_true) comes from that file, as last week. You can also give it with `--fps 239.6`. To see first that the files are right, add `--seconds 2` to the end of the command: it tracks only 2 seconds. Add `--fine A,C` to track the animals A and C in a close-up window that follows them. Use the names as they are in your export. Their outlines are sharper.

```shell
outline-tracker from-tracker "path/to/video_tracker.mp4" "path/to/sam/extra/start.csv"
```

You see the plan, the folder for the files, the progress, and at the end the files that were saved. Your numbers are different:

```text
edgetam: 2 shrimp (A, B), frames 120-1320 every 2 (601 frames, 5.0 s at fps_true = 239.6); scale 32.40 um per pixel
  run folder: .../video_tracker_outline_sam
  frame 50/601: 1.20 s per frame, about 11 min left
  saved .../edgetam/A.csv, .../edgetam/B.csv and .../overlay.mp4
```

Lines that start with `CHECK:` are last week's three checks. They say that an animal was lost, that it jumped faster than 100 mm/s, or that its outline changed size by more than 2 times. Look at the overlay video at that time.

### Where the files are

All files of one run are in one new folder, the run folder: `<video folder>/<video stem>_outline_<student>/`. For the example above it is `path/to/video_tracker_outline_sam/`. The name `sam` is the name of the folder of your export (the folder above it, if it is called `extra`). Use `--student NAME` for another name. Run again with the same name and the new results replace the old ones.

Open this folder in Finder (Mac) or in File Explorer (Windows). It holds these files:

- `positions.csv`: where each animal is, frame by frame, in mm and in pixels.
- `shapes.csv`: the size, the axes, the heading and the solidity of each animal.
- `radial.csv`: the radius of the outline at 72 angles (close-up animals only).
- `outlines.npz`: the outlines, 128 points per frame (close-up animals only).
- `edgetam/`: one `<id>.csv` for each animal, in the format of last week's files. Nothing else is in this folder.
- `overlay.mp4`: your video with the outlines drawn on it, for your slides.
- `run.log`: what was run, and a summary of the quality checks.
- `session.json`: the settings of the run: scale, frame rate, start points.
- `results.npz`: the tracking results in pixels. The tool reads it; you do not.
- `README.txt`: the description of every file and column.

Every column and every unit is described in [docs/OUTPUTS.md](docs/OUTPUTS.md). Do not put `results.npz` and `overlay.mp4` in git; they are large. Share them through Drive.

### Load the tracks in your notebook

Last week your notebook read the tracks with `load_tracks(folder)`. It reads every `.csv` file in a folder, one file for each animal. The folder `edgetam` inside the run folder is such a folder. Give that folder to `load_tracks` and keep the rest of the call as it is:

```python
tracks = load_tracks("path/to/video_tracker_outline_sam/edgetam")
```

### The scale or the frame rate was wrong

You do not need to track again. Open `session.json` in a text editor. Change the number after `"fps_true"` (frames per second) or after `"mm_per_px"` (millimeters per pixel of the video). Change only the number. Save the file. Then write all files again:

```shell
outline-tracker export "path/to/video_tracker_outline_sam/session.json"
```

Nothing is tracked and the model is not loaded. You see the folder and one line for each file:

```text
Exporting the run folder path/to/video_tracker_outline_sam
wrote positions.csv: 85.2 kB
NOTE: overlay.mp4 was not regenerated and is as it was. Add --overlay to make it again from the video.
```

The overlay video stays as it is. Add `--overlay` to make it again. That decodes the whole video, so it takes a while.

## Check and convert a video

`outline-tracker check VIDEO` shows what the file says about itself, and warnings. A line that starts with `WARNING:` says what is wrong: read it before you use the file.

```shell
outline-tracker check "path/to/video.MOV"
```

```text
  the file says: 240.00 fps, 2400 frames, 10.00 s of file time
  note: Real time comes from your stopwatch clip (fps_true in data/manifest.csv), not from this file.
  OK: no problems found. Now measure fps_true from your stopwatch clip.
```

`outline-tracker convert VIDEO` makes a copy that Tracker can open, with every frame kept, in the same order. It writes `<name>_tracker.mp4` next to the original. A minute of 240 fps video takes a few minutes.

```shell
outline-tracker convert "path/to/video.MOV"
```

```text
Converting path/to/video.MOV (a minute of 240 fps video takes a few minutes) ...
  wrote path/to/video_tracker.mp4: 2400 frames, 1920 x 1080 px, 240.00 fps in the file
  Open this copy in Tracker. Real time still comes from fps_true (your stopwatch clip).
```

## Troubleshooting

### A file is open in another program

On Windows, a program such as Excel locks a CSV file that is open in it. The tool cannot replace the file, so it writes the new data next to it, for example in `positions.new.csv`, and tells you. Close the program. Then write the files again with `outline-tracker export SESSION.json`, as shown above.

### The model does not download

The first run downloads the model, about 56 MB. If it fails, you see a line that starts with `ERROR:` and says that the first run needs the internet once. Connect to the internet, and run the same command again. If it fails again, try another network, for example the hotspot of your phone. Some networks block the download.

### The Apple GPU fails (Mac)

You see a line that starts with `The Apple GPU failed` and ends with `using the processor instead`. This is not a problem. The run goes on with the processor and it is slower. To use the processor from the start, add `--device cpu`.

### Windows cannot find outline-tracker

Close PowerShell and open a new window. The install changes the list of places where Windows looks for commands, and only a new window sees the change. If it still fails, run this line, close PowerShell and open a new window again:

```powershell
uv tool update-shell
```

### The folder holds other .csv or .txt files

`from-tracker` does not write into a folder with your Tracker files, because your notebook would read the new files as tracks. Last week `--out` named the folder of the track files. This week `--out` names the run folder: give a new folder, or leave out `--out`.

## Getting the original video off your phone

Tracker cannot read HEVC (H.265), the format that iPhones use for 240 fps slow motion. Follow these steps:

1. Take the ORIGINAL video file from the phone, not a copy that an app made. A slow-motion original says 120 to 240 fps.
2. Run `outline-tracker check VIDEO` on it. If it says "The file says" a frame rate below 100 fps, the file is probably a re-timed "compatible" copy made by the phone or an app. Transfer the ORIGINAL file again. If it says that motion per frame is larger at the start and at the end, the slow motion was rendered. Do not analyze this file.
3. Run `outline-tracker convert VIDEO` on the original. Open the `_tracker.mp4` copy in Tracker, and use the same copy for `from-tracker`.
4. Real time comes from your stopwatch clip (fps_true in `data/manifest.csv`), never from the frame rate in the file.
