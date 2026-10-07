"""The default run folder and the guard against a folder of Tracker files (SPEC 8.1).

Where the expected values come from: the rule of SPEC 8.1, `<video folder>/<video stem>_outline_<student>/`,
worked out here by hand for each name; the characters Windows forbids in a file name (< > : " / \\ | ? *
and the control characters); and the list of this tool's files, `schema.FILES`. No units or coordinates
are involved: paths and names only.
"""

from pathlib import Path

import pytest

from outline_tracker import schema
from outline_tracker.run_folder import default_run_folder, folder_has_foreign_tables

VIDEO = Path("videos") / "day 1" / "groupB_2026-10-06_1325_main_tracker.mp4"
STEM = "groupB_2026-10-06_1325_main_tracker"


def folder_name(student):
    """The run folder's own name for `student`, after checking that it lies in the video's folder."""
    folder = default_run_folder(VIDEO, student)
    assert folder.parent == VIDEO.parent  # one folder name, directly in the video's folder
    assert folder.name.startswith(STEM + "_outline_")
    return folder.name[len(STEM + "_outline_"):]


# --------------------------------------------------------------------------- default_run_folder

def test_the_run_folder_is_next_to_the_video_and_keeps_the_stem_as_it_is():
    assert default_run_folder(VIDEO, "ana") == VIDEO.parent / "groupB_2026-10-06_1325_main_tracker_outline_ana"


def test_only_the_last_extension_is_dropped_from_the_video_name():
    assert default_run_folder(Path("clips") / "dish.v2.take3.MOV", "ana") == Path("clips") / "dish.v2.take3_outline_ana"


def test_a_video_given_by_name_alone_gives_a_folder_in_the_current_folder():
    assert default_run_folder("clip_tracker.mp4", "ana") == Path("clip_tracker_outline_ana")


def test_text_and_path_give_the_same_folder():
    assert default_run_folder(str(VIDEO), "ana") == default_run_folder(VIDEO, "ana")


def test_nothing_is_created(tmp_path):
    folder = default_run_folder(tmp_path / "clip_tracker.mp4", "ana")
    assert folder == tmp_path / "clip_tracker_outline_ana"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("student", ["ana", "Zoë", "José-María", "李雷", "Иван", "Ωmega", "j.d_anabi-2", "O'Neil",
                                     "ana(2)"])
def test_letters_of_any_alphabet_and_harmless_signs_are_kept(student):
    assert folder_name(student) == student


@pytest.mark.parametrize("student, safe", [
    ("Ana María", "Ana_María"),  # a space
    ("ana\tb", "ana_b"),  # a tab
    ("ana\nb", "ana_b"),  # a line break
    ("ana b", "ana_b"),  # a no-break space
    ("ana　b", "ana_b"),  # an ideographic space
    ("ana  b", "ana__b"),  # one _ per character
    ("a/b", "a_b"),
    ("a\\b", "a_b"),
    ("../../etc", ".._.._etc"),
    ("C:ana", "C_ana"),
    ('a<b>c"d|e?f*g', "a_b_c_d_e_f_g"),
    ("a\x00b\x1fc", "a_b_c"),  # control characters
])
def test_separators_forbidden_characters_and_whitespace_become_underscores(student, safe):
    assert folder_name(student) == safe


def test_whitespace_around_the_name_is_dropped():
    assert folder_name("  ana \n") == "ana"


@pytest.mark.parametrize("student, safe", [("J.", "J_"), ("ana...", "ana___"), ("..", "__")])
def test_a_name_never_ends_in_a_dot(student, safe):
    # Windows drops the dots at the end of a folder name, so "x_outline_J." would be another folder there.
    assert folder_name(student) == safe


@pytest.mark.parametrize("student", ["", "   ", "\t\n"])
def test_an_empty_name_is_refused(student):
    with pytest.raises(ValueError, match="name"):
        default_run_folder(VIDEO, student)


# --------------------------------------------------------------------------- folder_has_foreign_tables

def touch(folder: Path, *names: str) -> Path:
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    return folder


def test_a_missing_folder_and_an_empty_folder_have_no_foreign_tables(tmp_path):
    assert folder_has_foreign_tables(tmp_path / "not there") is False
    assert folder_has_foreign_tables(tmp_path) is False


def test_the_tools_own_files_are_not_foreign(tmp_path):
    own = [spec.name.replace("<model>", "edgetam").replace("<id>", "A") for spec in schema.FILES]
    assert {"positions.csv", "shapes.csv", "radial.csv", "probes.csv", "README.txt", "edgetam/A.csv"} <= set(own)
    assert folder_has_foreign_tables(touch(tmp_path, *own)) is False


@pytest.mark.parametrize("name", ["A.csv", "start.csv", "notes.txt", "B.TXT", "C.CSV", "positions-final.csv",
                                  "Positions.csv", "readme.txt", "positions.csv.txt"])
def test_any_other_csv_or_txt_file_is_foreign(tmp_path, name):
    # (the own files next to it have other names in any capitalization: some file systems ignore case)
    touch(tmp_path, "shapes.csv", "run.log", name)
    assert folder_has_foreign_tables(tmp_path) is True


def test_only_files_directly_in_the_folder_count(tmp_path):
    # last week's layout below a student's folder: the tables are in subfolders, which are not read
    touch(tmp_path, "edgetam/A.csv", "extra/start.csv", "notes/todo.txt")
    assert folder_has_foreign_tables(tmp_path) is False
    assert folder_has_foreign_tables(tmp_path / "edgetam") is True


def test_other_kinds_of_files_are_not_tables(tmp_path):
    touch(tmp_path, "clip_tracker.mp4", "clip.trk", "session.json", "frame.png", "csv", "txt", "table.csv.bak")
    assert folder_has_foreign_tables(tmp_path) is False


def test_a_folder_named_like_a_table_is_not_a_table(tmp_path):
    (tmp_path / "old.csv").mkdir()
    assert folder_has_foreign_tables(tmp_path) is False


def test_a_file_given_instead_of_a_folder_has_no_foreign_tables(tmp_path):
    touch(tmp_path, "A.csv")
    assert folder_has_foreign_tables(tmp_path / "A.csv") is False


def test_files_this_tool_leaves_when_a_target_is_locked_or_a_write_is_cut_short_are_its_own(tmp_path):
    # fileio.atomic_write: <stem>.new<suffix> for a locked target, <stem>-<8 hex digits>-tmp<suffix> while writing
    touch(tmp_path, "positions.new.csv", "probes.new.csv", "README.new.txt", "probes-1a2b3c4d-tmp.csv",
          "README-00ff00ff-tmp.txt")
    assert folder_has_foreign_tables(tmp_path) is False


@pytest.mark.parametrize("name", ["A.new.csv", "notes-1a2b3c4d-tmp.txt", "probes-1a2b3c4-tmp.csv",
                                  "probes-1A2B3C4D-tmp.csv", "probes.new.txt"])
def test_look_alikes_of_those_names_are_foreign(tmp_path, name):
    assert folder_has_foreign_tables(touch(tmp_path, name)) is True


def test_hidden_files_are_passed_over(tmp_path):
    # e.g. the "._name" files macOS writes next to each file on a USB stick or a network drive
    touch(tmp_path, "._positions.csv", "._A.csv", ".notes.txt")
    assert folder_has_foreign_tables(tmp_path) is False
