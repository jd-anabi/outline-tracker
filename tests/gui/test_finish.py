"""The last gaps of the window, and its two documents (task C8b; SPEC 8.1, 10.1, 16).

- The name. A name that code gives the controller (`set_student`, an opened session) shows in the
  field of panel 1, and what the field holds never replaces a newer name with an older or an
  empty one: before this, a save took the field's text for the name, whatever the name was.
- README.md has the section "Quickstart": ten numbered steps. Every control it marks (in bold or
  in backticks) is held against the texts of the window's own buttons and menu items, so the page
  cannot name a button the window does not have. Help > Quickstart opens that section.
- docs/DEVELOPER.md names every module of the package.

The README's own shape (ten steps, its commands) is checked in tests/test_readme.py, without Qt.
"""

import re

from PySide6.QtCore import QUrl

from last_controls_helpers import REPO, control_texts, marked_names, quickstart
from outline_tracker.gui import menus
from session_helpers import body, read_json, type_into

README = REPO / "README.md"
DEVELOPER = REPO / "docs" / "DEVELOPER.md"
PACKAGE = REPO / "outline_tracker"


# ---------------------------------------------------------------------------------------------
# The name in panel 1


def test_a_name_set_from_code_shows_in_the_field_and_survives_a_save(window, clip_in_odd_folder):
    window.open_path(clip_in_odd_folder.path)
    field, controller = body(window, 1).name_edit, window.controller
    assert field.text() == ""
    controller.set_student("Ada")
    assert field.text() == "Ada"
    written = controller.save_now()  # a save asks the panels for what is typed: the empty field's turn before
    assert controller.student == "Ada" and controller.session.student == "Ada"
    assert written is not None and read_json(written)["student"] == "Ada"
    assert written.parent.name == "dish_tracker_outline_Ada" and field.text() == "Ada"


def test_the_field_never_puts_an_older_name_in_place_of_a_newer_one(window, qtbot, clip_in_odd_folder):
    window.open_path(clip_in_odd_folder.path)
    field, controller = body(window, 1).name_edit, window.controller
    type_into(qtbot, field, "Ada")
    assert controller.student == "Ada"
    controller.set_student("Grace")  # newer than what the field holds
    assert field.text() == "Grace"
    field.editingFinished.emit()  # the field is left: it has nothing new to say
    controller.save_now()
    assert controller.student == "Grace" and field.text() == "Grace"
    type_into(qtbot, field, "Mary")  # what is typed after that is the newest again
    assert controller.student == "Mary"


def test_a_name_given_before_a_video_is_kept_by_a_save_and_shown_when_the_video_opens(window, clip_in_odd_folder):
    field, controller = body(window, 1).name_edit, window.controller
    controller.set_student("Ada")  # without a session the controller has nobody to tell
    assert controller.save_now() is None  # nothing to save yet; the empty field must not take the name away
    assert controller.student == "Ada" and field.text() == "Ada"
    window.open_path(clip_in_odd_folder.path)
    assert field.text() == "Ada" and controller.session.student == "Ada"
    assert controller.run_folder.name == "dish_tracker_outline_Ada"


def test_a_name_typed_and_emptied_by_the_user_still_counts(window, qtbot, clip_in_odd_folder):
    window.open_path(clip_in_odd_folder.path)
    field, controller = body(window, 1).name_edit, window.controller
    controller.set_student("Ada")
    type_into(qtbot, field, "")  # the user's own doing, and the newest
    assert controller.student == "" and controller.run_folder is None
    field.setText("Grace")  # typed and not entered: a save takes it
    controller.save_now()
    assert controller.student == "Grace"


def test_an_opened_sessions_name_shows_in_the_field(window, qtbot, clip_in_odd_folder):
    window.open_path(clip_in_odd_folder.path)
    field, controller = body(window, 1).name_edit, window.controller
    controller.set_student("Ada")
    session_file = controller.save_now()
    controller.set_student("Grace")
    assert field.text() == "Grace"
    window.open_path(session_file)
    assert controller.student == "Ada" and field.text() == "Ada"


# ---------------------------------------------------------------------------------------------
# README.md: the Quickstart names the window's own controls


def test_every_control_the_quickstart_marks_is_a_button_or_a_menu_item_of_the_window(window):
    marked = marked_names(quickstart(README.read_text(encoding="utf-8")))
    texts = control_texts(window)
    assert [name for name in marked if name not in texts] == []
    # the page is about the window: it names what the student has to press in panels 1 to 7
    assert {"Open video", "Stopwatch…", "Stick", "Tape", "Circle", "Origin to Center", "Add", "Track"} <= set(marked)


def test_the_check_of_marked_names_finds_a_button_the_window_does_not_have(window):
    page = "## Quickstart\n\n1. Click **Open video**, then `Trakc`.\n\n```shell\n`not this`\n```\n## Next\n**No**"
    section = quickstart(page)
    assert marked_names(section) == ["Open video", "Trakc"]
    assert [name for name in marked_names(section) if name not in control_texts(window)] == ["Trakc"]


def test_help_quickstart_opens_the_quickstart_section_of_the_readme(window, monkeypatch):
    opened = []

    class Browser:
        @staticmethod
        def openUrl(url):
            opened.append(url)
            return True

    monkeypatch.setattr(menus, "QDesktopServices", Browser)
    window.menus.quickstart_action.trigger()
    assert opened == [QUrl("https://github.com/jd-anabi/outline-tracker#quickstart")]
    assert "## Quickstart\n" in README.read_text(encoding="utf-8")  # the heading that address goes to


# ---------------------------------------------------------------------------------------------
# docs/DEVELOPER.md


def modules() -> list[str]:
    """Every module of the package as the developer's page names it: its path under
    outline_tracker/, with `/` (`gui/panels/track_panel.py`)."""
    return sorted(path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*.py"))


def test_the_developers_page_names_every_module_of_the_package():
    text = DEVELOPER.read_text(encoding="utf-8")
    named = set(re.findall(r"`([\w/]+\.py)`", text))
    assert {"cli.py", "gui/worker.py", "gui/worker_engine.py", "gui/panels/track_panel.py"} <= set(modules())
    assert [module for module in modules() if module not in named] == []


def test_the_developers_page_says_how_to_test_how_to_add_a_panel_and_who_writes_which_file():
    text = DEVELOPER.read_text(encoding="utf-8")
    for needed in ('uv run pytest -m "not slow"', "uv run pytest -m slow", "build(window)", "gui/panels/",
                   "session.json", "results.npz", "worker thread", "GUI thread", "(../CLAUDE.md)", "(ROADMAP.md)"):
        assert needed in text, needed
    # the rules are in CLAUDE.md and in section 2 of the roadmap: the page links to both and repeats none
    assert "Questions for J" not in text

