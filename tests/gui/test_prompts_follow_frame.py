"""The click points and the model's outline belong to the frame they were made on (task C4), and
they follow the frame on the screen whoever put it there: the bottom bar, a key, or another part
of the window that calls `MainWindow.show_frame` (a row of the flags table, the Stopwatch dialog).

`VideoView.frame_changed` is the one signal for that. Frames are video frame numbers.
"""

from prompt_helpers import clicked_object


def test_the_outline_of_frame_0_goes_when_another_part_shows_another_frame(window, qtbot, disk_clip):
    panel, _ = clicked_object(window, qtbot, disk_clip)  # object A, clicked on frame 0
    prompts = panel.prompts
    qtbot.waitUntil(lambda: "A" in prompts.outlines)

    window.show_frame(4)  # not through the bottom bar
    assert window.view.frame == 4
    assert prompts.outlines == {}  # frame 4 has no clicks: nothing of frame 0 stays on the picture

    window.show_frame(0)
    qtbot.waitUntil(lambda: "A" in prompts.outlines)  # back on the clicked frame, its outline is back
