"""The main window (SPEC 10.1): the video in the middle, frame navigation under it, and at the
right a dock with the nine numbered panels in the order of the work.

For now it is the empty frame of the window. The video area and the bottom bar are placeholders,
and each panel is its header and its hint line; later tasks put the video view, the navigation and
the panels' controls in. Lengths are Qt's device-independent px (Qt scales them on a 150% or 200%
screen); nothing here is in video px or mm.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QDockWidget, QFrame, QLabel, QMainWindow, QScrollArea, QVBoxLayout, QWidget

from outline_tracker.gui.panel import Panel

TITLE = "Outline Tracker"
MINIMUM_SIZE = (960, 600)        # width, height of the window
START_SIZE = (1440, 900)         # on a screen that offers at least this much; else maximized
DOCK_WIDTHS = (340, 400, 520)    # smallest, at start, largest
DOCK_MARGIN = 8                  # around the panels, and between two of them
BOTTOM_BAR_HEIGHT = 76           # slider, flag strip and button row of the frame navigation

NO_VIDEO = "Open a video to start.\nThen follow panels 1 to 9 at the right."

# Number, title, and the hint while nothing is done. The estimate of panel 7 needs a clip and
# objects; without them its hint is the first reason why tracking cannot start.
PANELS = (
    (1, "Student and video", "Type your name. Then open your video (a _tracker.mp4 file)."),
    (2, "Time", "Type the true frame rate (fps_true), or measure it with the stopwatch."),
    (3, "Calibration", "Click Stick. Click the two ends of a known length on the ruler. Type the length."),
    (4, "Dish and axes", "Click Circle. Click 6 or more points on the inner wall of the dish, spread around it."),
    (5, "Probes", "Optional. Add a box over the LED. Then click Measure."),
    (6, "Objects", "Click Add. Then click on one animal in the video. An outline appears."),
    (7, "Track", "Type your name in panel 1 first."),
    (8, "Review and fix", "Track first. The flags appear here."),
    (9, "Export", "Export all writes the CSV files, the overlay video, the log and README.txt to the run folder."),
)
OPTIONAL_PANELS = {5}  # "Optional" in place of "Not started"


def start_geometry(available: QRect) -> QRect | None:
    """Where the window opens on a screen whose usable area is `available` (device-independent px,
    in the desktop's coordinates): 1440 x 900 in the middle of it, or None, which means maximized,
    when the area is less than 1440 wide or less than 900 high."""
    width, height = START_SIZE
    if available.width() < width or available.height() < height:
        return None
    return QRect(available.left() + (available.width() - width) // 2,
                 available.top() + (available.height() - height) // 2, width, height)


class MainWindow(QMainWindow):
    """The window. `segmenter_factory(model, device)` makes the segmenter that tracking will use
    (the real model in the application, a stand-in in tests); it is kept as `segmenter_factory` for
    the worker. Parts: `video_area`, `bottom_bar`, `dock`, `scroll` (the scroll area in the dock) and
    `panels`, the nine `Panel`s in order."""

    def __init__(self, segmenter_factory=None, parent: QWidget | None = None):
        super().__init__(parent)
        self.segmenter_factory = segmenter_factory
        self.setWindowTitle(TITLE)
        self.setMinimumSize(*MINIMUM_SIZE)

        self.video_area = QLabel(NO_VIDEO)
        self.video_area.setObjectName("VideoArea")
        self.video_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bottom_bar = QFrame()
        self.bottom_bar.setObjectName("BottomBar")
        self.bottom_bar.setFixedHeight(BOTTOM_BAR_HEIGHT)
        central = QWidget()
        rows = QVBoxLayout(central)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(0)
        rows.addWidget(self.video_area, 1)
        rows.addWidget(self.bottom_bar)
        self.setCentralWidget(central)

        self.panels = [Panel(number, title, hint, **({"todo_word": "Optional"} if number in OPTIONAL_PANELS else {}))
                       for number, title, hint in PANELS]
        self.panels[0].set_expanded(True)  # the first panel that is not done; the others stay closed
        column = QWidget()
        column.setObjectName("DockColumn")
        stack = QVBoxLayout(column)
        stack.setContentsMargins(DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN, DOCK_MARGIN)
        stack.setSpacing(DOCK_MARGIN)
        for panel in self.panels:
            stack.addWidget(panel)
        stack.addStretch(1)

        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)  # the column is as wide as the dock: only its height scrolls
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)  # so the width never jumps
        self.scroll.setWidget(column)

        smallest, at_start, largest = DOCK_WIDTHS
        self.dock = QDockWidget()
        self.dock.setObjectName("Dock")
        self.dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)  # not closable, movable, floating
        self.dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.dock.setTitleBarWidget(QWidget())  # an empty widget: no title bar
        self.dock.setMinimumWidth(smallest)
        self.dock.setMaximumWidth(largest)
        self.dock.setWidget(self.scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.resizeDocks([self.dock], [at_start], Qt.Orientation.Horizontal)

        self.statusBar()  # made here, so that it shows from the start; empty until there is a message

    def open_path(self, path: Path) -> None:
        """Take the video or session file the window was started with. For now its file name (never
        its folder) is shown in the status bar."""
        self.statusBar().showMessage(Path(path).name)

    def show_at_start(self) -> None:
        """Show the window as it opens for the user: 1440 x 900 px in the middle of its screen, or
        maximized on a smaller screen (`start_geometry`)."""
        place = start_geometry(self.screen().availableGeometry())
        if place is None:
            self.showMaximized()
        else:
            self.setGeometry(place)
            self.show()
