"""The `Panel` widget (task C0; the design note's panel anatomy): number, title, state, hint, body.

A panel is tested inside the main window, which the `window` fixture makes and closes. Expected
values are the note's: the three states and their words, header row 32 px high, body padding 12 px,
8 px between rows, the title one point larger than the application's font and DemiBold.
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from gui_helpers import shown


def test_the_parts_carry_the_roles_the_style_sheet_selects(window):
    panel = window.panels[0]
    parts = (panel, panel.badge, panel.header, panel.state_label, panel.hint)
    assert [part.property("role") for part in parts] == ["panel", "badge", "panelHeader", "state", "hint"]


@pytest.mark.parametrize("state, word", [("todo", "Not started"), ("done", "Done"), ("attention", "Needs attention")])
def test_a_state_is_a_word_and_a_property_for_the_style_sheet(window, state, word):
    panel = window.panels[2]
    panel.set_state("done" if state != "done" else "todo")
    panel.set_state(state)
    assert panel.state == state
    assert panel.state_label.text() == word
    assert [part.property("state") for part in (panel, panel.badge, panel.state_label)] == [state] * 3


def test_the_probes_panel_is_optional_until_it_is_done(window):
    probes = window.panels[4]
    assert probes.state_label.text() == "Optional"
    probes.set_state("done")
    assert probes.state_label.text() == "Done"
    probes.set_state("todo")
    assert probes.state_label.text() == "Optional"


def test_an_unknown_state_is_refused_and_changes_nothing(window):
    panel = window.panels[0]
    with pytest.raises(ValueError, match="finished"):
        panel.set_state("finished")
    assert panel.state == "todo" and panel.property("state") == "todo"
    assert panel.state_label.text() == "Not started"


def test_the_hint_is_the_first_row_of_the_body_and_wraps(window):
    panel = window.panels[3]
    assert panel.body.itemAt(0).widget() is panel.hint
    assert panel.hint.wordWrap()
    panel.set_hint("The circle needs 3 or more points. Click more points on the dish wall.")
    assert panel.hint.text() == "The circle needs 3 or more points. Click more points on the dish wall."


def test_a_click_on_a_header_opens_and_closes_that_panel_only(window, qtbot):
    shown(window, qtbot)
    second = window.panels[1]
    assert second.header.arrowType() == Qt.ArrowType.RightArrow
    qtbot.mouseClick(second.header, Qt.MouseButton.LeftButton)
    assert [panel.is_expanded() for panel in window.panels] == [True, True] + [False] * 7
    assert second.hint.isVisible()
    assert second.header.arrowType() == Qt.ArrowType.DownArrow
    qtbot.mouseClick(second.header, Qt.MouseButton.LeftButton)
    assert [panel.is_expanded() for panel in window.panels] == [True] + [False] * 8
    assert not second.hint.isVisible()
    assert second.header.isVisible() and second.badge.isVisible() and second.state_label.isVisible()
    assert second.header.arrowType() == Qt.ArrowType.RightArrow


def test_set_expanded_and_the_header_button_agree(window):
    panel = window.panels[5]
    for expanded in (True, False, False, True):
        panel.set_expanded(expanded)
        assert panel.is_expanded() is expanded
        assert panel.header.isChecked() is expanded
        assert panel.header.arrowType() == (Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)


def test_a_collapsed_panel_is_its_header_row_only(window, qtbot):
    shown(window, qtbot)
    panel = window.panels[1]
    assert not panel.is_expanded()
    frame = panel.height() - panel.contentsRect().height()  # the border the style sheet draws, if any
    assert panel.header_row.height() == 32
    assert panel.height() == 32 + frame


def test_the_title_is_one_point_larger_than_the_base_font_and_demibold(window, qapp):
    font = window.panels[0].header.font()
    assert font.pointSizeF() == qapp.font().pointSizeF() + 1
    assert font.weight() == QFont.Weight.DemiBold
    assert font.family() == qapp.font().family()  # the system's font, never a family of our own
