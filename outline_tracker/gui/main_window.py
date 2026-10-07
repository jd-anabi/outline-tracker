"""The main window (SPEC 10.1): the video in the middle, frame navigation under it, and at the
right a dock with the nine numbered panels in the order of the work.

The window owns the parts and connects them: the `SessionController` (the open video and its
session), the `VideoView`, the `NavigationBar`, and the nine `Panel`s. The controls of a panel are
put in by its own module (outline_tracker/gui/panels). Until a video is open the video area says
how to start; then it shows the view under a bar with the tool's line, Fit, 1:1 and the zoom.
Lengths are Qt's device-independent px (Qt scales them on a 150% or 200% screen); frames are video
frame numbers counted from 0; nothing here is in video px or mm.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QDockWidget, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from outline_tracker.geometry import grid_frames
from outline_tracker.gui import dialogs, panels
from outline_tracker.gui.navigation import NavigationBar
from outline_tracker.gui.panel import Panel
from outline_tracker.gui.session_controller import SessionController
from outline_tracker.gui.video_view import VideoView

TITLE = "Outline Tracker"
MINIMUM_SIZE = (960, 600)        # width, height of the window
START_SIZE = (1440, 900)         # of the window's content, on a screen with room for it; else maximized
FRAME_ALLOWANCE = (16, 40)       # room for the system's frame around that: its edges, its title bar
DOCK_WIDTHS = (340, 400, 520)    # smallest, at start, largest
DOCK_MARGIN = 8                  # around the panels, and between two of them
VIEW_BAR_HEIGHT = 32             # above the picture: the tool's line, Fit, 1:1, the zoom
CONTROL_HEIGHT, PRIMARY_HEIGHT = 28, 32  # a button; the button of the next step

START_TITLE = "Open a video to start."
START_HINT = "Then follow panels 1 to 9 at the right."
PAN_TEXT = "Pan: drag to move the picture. Scroll to zoom."
NOT_OPENED = "The video could not be opened"  # the heading of the message; what to do follows it
VIDEO_FILTERS = ["Videos (*.mp4 *.m4v *.mov *.avi *.mkv)", "All files (*)"]

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
    """The room for the window on a screen whose usable area is `available` (device-independent px,
    in the desktop's coordinates), or None, which means maximized.

    The room is the content's 1440 x 900 plus `FRAME_ALLOWANCE` for the frame the system draws around
    it (Windows 11: 8 px at the left, at the right and below, and a title bar of 31 px; macOS: a title
    bar of 28 px), in the middle of `available`. The frame's top left corner goes to the room's top
    left corner. An area with less than this room in width or in height gets None: a window put
    there would have its title bar, with the close and maximize buttons, partly above the screen.
    """
    width, height = (size + frame for size, frame in zip(START_SIZE, FRAME_ALLOWANCE))
    if available.width() < width or available.height() < height:
        return None
    return QRect(available.left() + (available.width() - width) // 2,
                 available.top() + (available.height() - height) // 2, width, height)


class MainWindow(QMainWindow):
    """The window. `segmenter_factory(model, device)` makes the segmenter that tracking will use
    (the real model in the application, a stand-in in tests); it is kept as `segmenter_factory` for
    the worker.

    What a panel's module works with: `controller` (the open video and its session), `view` (the
    video view), `navigation` (the bottom bar) and `panels` (the nine `Panel`s in order). Other
    parts: `video_area` (the start page, or the view under its bar), `bottom_bar` (the navigation),
    `dock`, `scroll` (the scroll area in the dock); on the start page `start_title`, `open_button`
    and `start_hint`; in the bar above the picture `tool_text`, `fit_button`, `one_to_one_button`
    and `zoom_label`; in the File menu `open_action` and `quit_action`."""

    closing = Signal()  # the window is about to close; the session and the video are still open

    def __init__(self, segmenter_factory=None, parent: QWidget | None = None):
        super().__init__(parent)
        self.segmenter_factory = segmenter_factory
        self._closed = False
        self.setWindowTitle(TITLE)
        self.setMinimumSize(*MINIMUM_SIZE)

        self.controller = SessionController(self)
        self.view = VideoView()
        self.navigation = self.bottom_bar = NavigationBar()
        self.video_area = QStackedWidget()
        self.video_area.setObjectName("VideoArea")
        self.video_area.addWidget(self._start_page())
        self._picture_page = self._picture_page_with_bar()
        self.video_area.addWidget(self._picture_page)
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

        file_menu = self.menuBar().addMenu("File")
        self.open_action = QAction("Open video", self)
        self.open_action.setShortcut(QKeySequence(QKeySequence.StandardKey.Open))
        self.open_action.triggered.connect(self.choose_video)
        self.quit_action = QAction("Quit", self)
        self.quit_action.setShortcut(QKeySequence(QKeySequence.StandardKey.Quit))
        self.quit_action.setMenuRole(QAction.MenuRole.QuitRole)
        self.quit_action.triggered.connect(self.close)
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        self.controller.video_opened.connect(self._video_opened)
        self.controller.session_changed.connect(self._session_changed)
        self.navigation.frame_requested.connect(self.show_frame)
        self.view.zoom_changed.connect(lambda zoom: self.zoom_label.setText(f"{round(100 * zoom)}%"))
        self.view.tool_changed.connect(self._tool_changed)
        panels.build_bodies(self)  # last: a panel's module finds every part above

    def _start_page(self) -> QWidget:
        """What the video area shows until a video is open: how to start, around the button."""
        self.start_title, self.start_hint = QLabel(START_TITLE), QLabel(START_HINT)
        self.open_button = QPushButton("Open video")
        self.open_button.setProperty("kind", "primary")
        self.open_button.setFixedHeight(PRIMARY_HEIGHT)
        self.open_button.setToolTip("Choose the video to track")
        self.open_button.clicked.connect(self.choose_video)
        page = QWidget()
        column = QVBoxLayout(page)
        column.setSpacing(12)
        column.addStretch(1)
        for part in (self.start_title, self.open_button, self.start_hint):
            column.addWidget(part, 0, Qt.AlignmentFlag.AlignHCenter)
        column.addStretch(1)
        return page

    def _picture_page_with_bar(self) -> QWidget:
        """The view under its bar: the tool's line at the left; Fit, 1:1 and the zoom at the right."""
        self.tool_text = QLabel(PAN_TEXT)
        self.fit_button, self.one_to_one_button = QPushButton("Fit"), QPushButton("1:1")
        self.fit_button.setToolTip("Show the whole frame")
        self.one_to_one_button.setToolTip("One video pixel per screen pixel")
        self.fit_button.clicked.connect(self.view.fit)
        self.one_to_one_button.clicked.connect(self.view.one_to_one)
        self.zoom_label = QLabel()
        self.zoom_label.setToolTip("How large the picture is drawn: 100% is one video pixel per screen pixel")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.zoom_label.setFixedWidth(self.zoom_label.fontMetrics().horizontalAdvance("00000%"))  # nothing jumps
        bar = QWidget()
        bar.setFixedHeight(VIEW_BAR_HEIGHT)
        row = QHBoxLayout(bar)
        row.setContentsMargins(DOCK_MARGIN, 0, DOCK_MARGIN, 0)
        row.setSpacing(DOCK_MARGIN)
        row.addWidget(self.tool_text, 1)
        for part in (self.fit_button, self.one_to_one_button):
            part.setFixedHeight(CONTROL_HEIGHT)
            row.addWidget(part)
        row.addWidget(self.zoom_label)
        page = QWidget()
        rows = QVBoxLayout(page)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(0)
        rows.addWidget(bar)
        rows.addWidget(self.view, 1)
        return page

    def choose_video(self) -> None:
        """Ask which video to open; the chosen file goes to `open_path`."""
        dialogs.open_file(self, "Open video", VIDEO_FILTERS, self.open_path)

    def open_path(self, path: Path) -> None:
        """Open the video at `path`: it replaces the video that was open, and its first frame shows.
        A file that is no video the tracker can read gives a message that says what to do, and what
        was open stays. The status bar names the file (never its folder). A session file
        (`.json`) is only named there for now: a later task opens sessions."""
        path = Path(path)
        if path.suffix.lower() == ".json":
            self.statusBar().showMessage(path.name)
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # a long video takes a moment to open
        try:
            self.controller.open_video(path)
        except ValueError as refused:
            self.statusBar().showMessage(f"{path.name} could not be read.")
            dialogs.message(self, "problem", f"{NOT_OPENED}\n{refused}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"{path.name} is open.")

    def show_frame(self, frame: int) -> None:
        """Show video frame `frame` (counted from 0) in the view, and put the bottom bar on it. A
        frame the video does not have is reported in the status bar; the frame shown then stays."""
        try:
            self.view.show_frame(frame)
        except IndexError:
            self.statusBar().showMessage(f"Frame {frame} cannot be read. The video ends before this frame.")
        if self.view.frame is not None:
            self.navigation.set_frame(self.view.frame)

    def _video_opened(self) -> None:
        """Show the new video from its first frame, wherever the bar was in the video before it."""
        self.view.set_source(self.controller.source)
        self.navigation.set_grid([])  # off the old video's frame: the new grid then starts at its first one
        self.video_area.setCurrentWidget(self._picture_page)
        self.view.setFocus()  # not a button: Space or Enter must not press one by accident
        self._session_changed()

    def _session_changed(self) -> None:
        """Put the bottom bar on the session's clip and fps_true, and show the frame it is on."""
        session = self.controller.session
        self.navigation.set_grid(grid_frames(session.clip.start, session.clip.end, session.clip.step))
        self.navigation.set_fps(session.time.fps_true)
        if self.navigation.frame != self.view.frame:
            self.show_frame(self.navigation.frame)

    def _tool_changed(self) -> None:
        tool = self.view.tool
        self.tool_text.setText(PAN_TEXT if tool is None else getattr(tool, "text", ""))

    def closeEvent(self, event) -> None:
        """Closing the window tells its parts first (`closing`: a panel saves, a worker stops), while
        the session and the video are still there, and then releases the video file. The parts are
        told once, however often the window is closed."""
        if not self._closed:
            self._closed = True
            self.closing.emit()
        self.controller.close()
        super().closeEvent(event)

    def show_at_start(self) -> None:
        """Show the window as it opens for the user: its content 1440 x 900 px, in the middle of its
        screen with the title bar inside the screen's usable area, or maximized where the screen has
        no room for that (`start_geometry`)."""
        room = start_geometry(self.screen().availableGeometry())
        if room is None:
            self.showMaximized()
        else:
            self.resize(*START_SIZE)
            self.move(room.topLeft())  # move() places the frame; setGeometry() would place the content
            self.show()
