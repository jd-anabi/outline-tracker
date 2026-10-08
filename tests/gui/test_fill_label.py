"""The Fill switch in the bar above the picture is a button like Fit and 1:1 next to it, and it
shows whether it is on: found in a picture of the window, where a check box on the dark bar had a
label that could not be read.

Colors are read as the most frequent color inside the button (its fill; the text and the outline
are a minority of its pixels), so nothing depends on a font or on where a letter is drawn.
"""

from collections import Counter

import pytest
from gui_helpers import show
from PySide6.QtWidgets import QApplication, QPushButton

from outline_tracker.gui import theme


def fill_of(button) -> str:
    """The most frequent color drawn inside `button`, as `#RRGGBB` in capitals."""
    image = button.grab().toImage()
    colors = Counter(image.pixelColor(x, y).name().upper()
                     for x in range(image.width()) for y in range(image.height()))
    return colors.most_common(1)[0][0]


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_the_fill_switch_is_a_button_like_its_neighbors_and_shows_that_it_is_on(dark, window, qapp, qtbot,
                                                                              dish_clip, look):
    theme.apply(qapp, dark)
    window.open_path(dish_clip.path)  # the bar above the picture shows once a video is open
    show(window, qtbot)
    colors = theme.tokens(dark)
    switch = window.fill_box
    assert isinstance(switch, QPushButton) and switch.isVisible()
    assert switch.height() == window.fit_button.height()
    assert fill_of(switch) == fill_of(window.fit_button) == colors["panel"]  # off: a button like Fit
    switch.click()
    QApplication.processEvents()
    assert switch.isChecked() and fill_of(switch) == colors["accentSoft"]  # on: the look of a chosen tool
    assert fill_of(window.fit_button) == colors["panel"]
    switch.click()
    QApplication.processEvents()
    assert fill_of(switch) == colors["panel"]
