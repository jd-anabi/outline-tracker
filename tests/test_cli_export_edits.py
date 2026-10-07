"""`outline-tracker export` on a session.json that was edited by hand, and on files cut short: whatever is wrong
with them, the command prints one `ERROR: ...` line on stderr and returns 1, never a Python traceback
(SPEC 11: errors are one line; the command's own help tells the student to run it after correcting "a head
click" in session.json, so the hand-edited session is its main input).

The run folders are the ones of tests/test_cli_export.py (the synthetic dish clip, tracked with `ExactFake`; a
stick of 300 px = 30 mm). A value of the wrong kind is what a hand edit makes: a head click with one number
instead of [u, v], a number or a null where a path of the video belongs. Nothing here depends on how the
package words its own message: each case asks for exit code 1, exactly one line on stderr and no more output
files than the command was able to write before it stopped.
"""

import shutil

import cv2
import pytest
from cli_export_helpers import edit_session, error_line, export, listing
from export_helpers import coarse_run

from outline_tracker import cli_export
from outline_tracker import export as export_module


@pytest.fixture(scope="module")
def tracked(dish_clip, tmp_path_factory):
    """The dish clip's A, B and C tracked coarse inside the dish crop, A with a head click; nothing exported."""
    folder = tmp_path_factory.mktemp("cli_export_edits") / "run"
    coarse_run(dish_clip, folder, ["A", "B", "C"], heads={"A": (150.5, 100.5)})
    return folder


@pytest.fixture
def run_folder(tracked, tmp_path):
    """This test's own copy of the tracked run folder."""
    return shutil.copytree(tracked, tmp_path / "run")


def refused(run_folder, capsys, *args) -> str:
    """Run the export, expect exit code 1, exactly one ERROR line on stderr and no `wrote ...` line on stdout
    (a refused export lists nothing), and return the ERROR line."""
    assert export(run_folder / "session.json", *args) == 1
    shown = capsys.readouterr()
    assert not any(line.startswith("wrote ") for line in shown.out.splitlines()), shown.out
    return error_line(shown.err)


# ---------------------------------------------------------------------------------------------
# A value of the wrong kind in what a student edits: a head click, the origin of the axes


@pytest.mark.parametrize("head", [[412], 412, "x", [1, 2, 3], ["a", "b"], [], True],
                         ids=["one-number-in-a-list", "one-number", "text", "three-numbers", "two-texts",
                              "empty-list", "true"])
def test_a_head_click_that_is_not_a_point_is_one_error_line_and_writes_nothing(run_folder, capsys, head):
    edit_session(run_folder, lambda data: data["tracks"][0].update(head_px=head))
    before = listing(run_folder)
    refused(run_folder, capsys)
    assert listing(run_folder) == before


@pytest.mark.parametrize("origin", [[100.5], 5, "ab"], ids=["one-number-in-a-list", "one-number", "text"])
def test_an_origin_that_is_not_a_point_is_one_error_line_and_writes_nothing(run_folder, capsys, origin):
    edit_session(run_folder, lambda data: data["axes"].update(origin_px=origin))
    before = listing(run_folder)
    refused(run_folder, capsys)
    assert listing(run_folder) == before


def test_the_error_line_says_what_to_check_and_names_the_kind_of_error(run_folder, capsys):
    edit_session(run_folder, lambda data: data["tracks"][0].update(head_px=[412]))
    error = refused(run_folder, capsys)
    assert "TypeError" in error  # what a person who reports it needs
    assert "session.json" in error and "edited" in error  # what the student looks at next


# ---------------------------------------------------------------------------------------------
# The overlay looks the video up: its path in the session may be of the wrong kind too


@pytest.mark.parametrize(("key", "value"), [("abspath", 5), ("abspath", None), ("relpath", 5), ("relpath", ["a"])])
def test_a_video_path_of_the_wrong_kind_with_the_overlay_is_one_error_line_and_no_overlay(
        run_folder, capsys, key, value):
    edit_session(run_folder, lambda data: data["video"].update({key: value}))
    refused(run_folder, capsys, "--overlay")
    assert not (run_folder / "overlay.mp4").exists()


def test_a_video_path_of_the_wrong_kind_does_not_matter_without_the_overlay(run_folder, capsys):
    edit_session(run_folder, lambda data: data["video"].update(abspath=5, relpath=5))
    assert export(run_folder / "session.json") == 0  # the video is not read: nothing to refuse
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------------------------
# A results.npz that is cut short (a copy that did not finish): not an OSError, not a ValueError


# When np.load fails on a zip that is damaged, it leaves the file it opened for the garbage collector to close.
# That ResourceWarning is numpy's, not the command's, and `-W error` would turn it into a failure of the test.
@pytest.mark.filterwarnings("ignore:unclosed file:ResourceWarning")
@pytest.mark.parametrize("keep", [0, 100, "half"], ids=["empty", "first-100-bytes", "half"])
def test_a_results_file_cut_short_is_one_error_line_and_writes_nothing(run_folder, capsys, keep):
    data = (run_folder / "results.npz").read_bytes()
    size = len(data) // 2 if keep == "half" else keep
    (run_folder / "results.npz").write_bytes(data[:size])
    before = listing(run_folder)
    error = refused(run_folder, capsys)
    assert "results.npz" in error
    assert listing(run_folder) == before


# ---------------------------------------------------------------------------------------------
# One line means one line: whatever the reason says, the ERROR is a single line


@pytest.mark.parametrize(("text", "line"), [
    ("one line", "one line"),
    ("first\nsecond", "first second"),
    ("  padded \r\n\t and\n\n spread  out \n", "padded and spread out"),
    ("", ""),
])
def test_one_line_joins_the_lines_of_a_reason(text, line):
    assert cli_export.one_line(text) == line


@pytest.mark.parametrize("error", [
    ValueError("first reason\nsecond reason"),
    OSError("could not write\r\n  the file"),
    cv2.error("OpenCV(4.x) error: (-215:Assertion failed)\n> in function 'cvtColor'\n"),
    KeyError("a\nb"),
    AssertionError(),
], ids=["ValueError", "OSError", "cv2.error", "KeyError", "AssertionError-without-a-reason"])
def test_an_exception_that_says_its_reason_over_several_lines_is_still_one_error_line(
        run_folder, capsys, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(export_module, "export_all", fail)  # what the export calls: the reason is the test's
    line = refused(run_folder, capsys)
    assert "\n" not in line and "\r" not in line
    if isinstance(error, AssertionError):
        assert "(AssertionError)" in line  # no reason to add after the name
