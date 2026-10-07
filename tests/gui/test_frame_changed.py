"""`VideoView.frame_changed`: the one signal that says which frame is on the screen now.

Parts that draw on the picture for the frame shown (outlines of results, click points) listen to
it, whoever asked for the frame: the bottom bar, a key, a row of the flags table, or a new video.
Frames are video frame numbers, counted from 0.
"""

import pytest


def test_every_frame_put_on_the_screen_is_told_once(window, dish_clip):
    seen = []
    window.view.frame_changed.connect(seen.append)
    window.open_path(dish_clip.path)
    assert seen == [0]  # a new video shows its first frame
    window.show_frame(4)
    window.navigation.step(1)  # the clip's step is 2: frame 6
    assert seen == [0, 4, 6] and window.view.frame == 6


def test_a_frame_the_video_does_not_have_is_not_told(window, dish_clip):
    window.open_path(dish_clip.path)
    seen = []
    window.view.frame_changed.connect(seen.append)
    with pytest.raises(IndexError):
        window.view.show_frame(100_000)
    assert seen == [] and window.view.frame == 0
