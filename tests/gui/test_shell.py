"""The empty main window (SPEC 10.1, task C0): what is where, before any panel is filled.

The expected values are the spec's (the order and the titles of the nine panels, SPEC 10.1) and the
design note's layout numbers: minimum window 960 x 600, dock 400 wide (340 to 520), 8 px around and
between the panels, bottom bar 76 high, and the hint each panel shows while nothing is done. Sizes
are Qt's device-independent px. The window comes from the `window` fixture (tests/helpers.py), which
closes it.
"""

import pytest
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtWidgets import QApplication, QDockWidget, QWidget

import helpers
from outline_tracker.gui import main_window
from outline_tracker.gui.panel import Panel

PANELS = [
    (1, "Student and video"),
    (2, "Time"),
    (3, "Calibration"),
    (4, "Dish and axes"),
    (5, "Probes"),
    (6, "Objects"),
    (7, "Track"),
    (8, "Review and fix"),
    (9, "Export"),
]

# What each panel says while nothing is done. Panel 7's estimate needs a clip and objects, so an
# empty window gives the first reason why Track cannot start.
HINTS = {
    1: "Type your name. Then open your video (a _tracker.mp4 file).",
    2: "Type the true frame rate (fps_true), or measure it with the stopwatch.",
    3: "Click Stick. Click the two ends of a known length on the ruler. Type the length.",
    4: "Click Circle. Click 6 or more points on the inner wall of the dish, spread around it.",
    5: "Optional. Add a box over the LED. Then click Measure.",
    6: "Click Add. Then click on one animal in the video. An outline appears.",
    7: "Type your name in panel 1 first.",
    8: "Track first. The flags appear here.",
    9: "Export all writes the CSV files, the overlay video, the log and README.txt to the run folder.",
}


def shown(window, qtbot):
    """Show the window (offscreen) and wait until it is on the screen."""
    with qtbot.waitExposed(window):
        window.show()
    return window


def test_the_window_opens_and_closes_offscreen(window, qtbot):
    assert not window.isVisible()
    shown(window, qtbot)
    assert window.isVisible()
    assert window.windowTitle() == "Outline Tracker"
    assert window.close()
    assert not window.isVisible()


def test_the_minimum_window_size_is_960_by_600(window):
    assert window.minimumSize() == QSize(960, 600)


def test_the_nine_panels_are_in_the_dock_in_workflow_order(window, qtbot):
    shown(window, qtbot)
    panels = window.panels
    assert [(panel.number, panel.title) for panel in panels] == PANELS
    assert [(panel.badge.text(), panel.header.text()) for panel in panels] == [(str(n), title) for n, title in PANELS]
    column = window.scroll.widget()
    assert window.dock.widget() is window.scroll
    assert all(panel.parentWidget() is column for panel in panels)
    assert set(window.findChildren(Panel)) == set(panels) and len(panels) == 9  # no other panel anywhere
    # top to bottom, 8 px around the column and 8 px between two panels
    assert panels[0].geometry().top() == 8
    gaps = [below.geometry().top() - (above.geometry().top() + above.height())
            for above, below in zip(panels, panels[1:])]
    assert gaps == [8] * 8
    assert {panel.geometry().left() for panel in panels} == {8}
    assert {column.width() - (panel.geometry().left() + panel.width()) for panel in panels} == {8}


def test_each_panel_shows_that_nothing_is_done_yet_and_what_to_do(window):
    for panel in window.panels:
        assert panel.state == "todo"
        assert panel.state_label.text() == ("Optional" if panel.number == 5 else "Not started")
        assert panel.hint.text() == HINTS[panel.number]


def test_at_start_only_the_first_panel_is_open(window, qtbot):
    shown(window, qtbot)
    assert [panel.is_expanded() for panel in window.panels] == [True] + [False] * 8
    assert [panel.hint.isVisible() for panel in window.panels] == [True] + [False] * 8
    assert all(panel.header.isVisible() and panel.badge.isVisible() for panel in window.panels)


def test_the_dock_stays_at_the_right_and_is_400_px_wide(window, qtbot):
    shown(window, qtbot)
    dock = window.dock
    assert window.dockWidgetArea(dock) == Qt.DockWidgetArea.RightDockWidgetArea
    assert dock.allowedAreas() == Qt.DockWidgetArea.RightDockWidgetArea
    assert dock.features() == QDockWidget.DockWidgetFeature.NoDockWidgetFeatures  # not closable, movable, floating
    assert not dock.isFloating()
    assert (dock.minimumWidth(), dock.maximumWidth()) == (340, 520)
    assert dock.width() == 400
    assert window.scroll.geometry().top() == 0  # no title bar above the panels


def test_the_dock_scrolls_vertically_only(window, qtbot):
    scroll = window.scroll
    assert scroll.widgetResizable()
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOn  # so the width never jumps
    window.resize(window.minimumSize())
    shown(window, qtbot)
    for panel in window.panels:
        panel.set_expanded(True)
    tall = QWidget()
    tall.setFixedHeight(2 * window.height())  # a body that cannot fit in the window
    window.panels[8].body.addWidget(tall)
    column, bar = scroll.widget(), scroll.verticalScrollBar()
    qtbot.waitUntil(lambda: bar.maximum() > 0)
    assert bar.maximum() == column.height() - scroll.viewport().height()  # all of the column can be reached
    # At the dock's smallest width nothing is pushed out at the right, whatever the font and however
    # long a title is: the column stays as wide as what shows of it (the title is cut instead).
    window.panels[0].set_state("attention")  # the longest state word
    window.panels[1].header.setText("A title that is far too long for a dock of 340 px. " * 4)
    window.resizeDocks([window.dock], [340], Qt.Orientation.Horizontal)
    qtbot.waitUntil(lambda: window.dock.width() == 340)
    QApplication.processEvents()  # a layout that would widen the column has done so by now
    assert column.width() == scroll.viewport().width()
    assert all(panel.width() == column.width() - 16 for panel in window.panels)  # 8 px at each side


def test_the_video_area_is_above_the_bottom_bar_and_left_of_the_dock(window, qtbot):
    shown(window, qtbot)
    central, video, bar = window.centralWidget(), window.video_area, window.bottom_bar
    assert video.parentWidget() is central and bar.parentWidget() is central
    assert bar.height() == 76
    assert video.geometry().top() == 0
    assert video.geometry().top() + video.height() == bar.geometry().top()  # nothing between them
    assert bar.geometry().top() + bar.height() == central.height()
    assert video.width() == bar.width() == central.width()
    assert central.geometry().left() + central.width() <= window.dock.geometry().left()
    status = window.statusBar()
    assert status.isVisible()
    assert status.geometry().top() >= central.geometry().top() + central.height()
    assert status.currentMessage() == ""


def test_the_video_area_says_how_to_start(window):
    assert window.video_area.text().splitlines() == [
        "Open a video to start.",
        "Then follow panels 1 to 9 at the right.",
    ]


def test_the_window_keeps_the_segmenter_factory_for_the_worker(window):
    assert window.segmenter_factory is helpers.stand_in_segmenter


def test_a_path_given_at_start_is_named_in_the_status_bar_without_its_folder(window, tmp_path):
    path = tmp_path / helpers.ODD_FOLDER / "dish_tracker.mp4"
    window.open_path(path)
    message = window.statusBar().currentMessage()
    assert "dish_tracker.mp4" in message
    assert helpers.ODD_FOLDER not in message and str(tmp_path) not in message


@pytest.mark.parametrize(
    "available, expected",
    [
        # 1440 x 900 in the middle of what the screen offers: (1920 - 1440) / 2 = 240, (1080 - 900) / 2 = 90
        (QRect(0, 0, 1920, 1080), QRect(240, 90, 1440, 900)),
        # a menu bar of 24 px above: 24 + (1416 - 900) / 2 = 282; (2560 - 1440) / 2 = 560
        (QRect(0, 24, 2560, 1416), QRect(560, 282, 1440, 900)),
        (QRect(0, 0, 1440, 900), QRect(0, 0, 1440, 900)),
        # under 1440 x 900 in either direction: maximized
        (QRect(0, 0, 1280, 720), None),
        (QRect(0, 0, 1438, 900), None),
        (QRect(0, 0, 1440, 898), None),
        (QRect(0, 0, 800, 800), None),
    ],
)
def test_the_window_starts_at_1440_by_900_centered_or_maximized_on_a_small_screen(available, expected):
    assert main_window.start_geometry(available) == expected


def test_show_at_start_places_the_window_on_its_screen(window, qtbot):
    with qtbot.waitExposed(window):
        window.show_at_start()
    wanted = main_window.start_geometry(window.screen().availableGeometry())
    if wanted is None:
        assert window.isMaximized()
    else:
        assert not window.isMaximized() and window.geometry() == wanted
