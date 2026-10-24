"""Fixtures of the tests of tests/tracking/ that more than one test file needs.

`tracked` is a test's own copy of the tracked run of the dish clip (`tracked_dish_folder`, tests/conftest.py)
in a folder with a space and non-ASCII characters in its name; the tests of the corrections in
tests/tracking/test_corrections.py and tests/tracking/test_corrections_new_piece.py use it.
"""

import shutil

import pytest
from helpers import ODD_FOLDER

from outline_tracker.measure.results import ResultsStore
from outline_tracker.core.session import Session


@pytest.fixture
def tracked(tracked_dish_folder, tmp_path):
    """This test's own copy of the tracked run (`tracked_dish_folder`, tests/conftest.py), in a folder
    named with a space and non-ASCII characters: (run folder, session, store)."""
    folder = tmp_path / ODD_FOLDER
    shutil.copytree(tracked_dish_folder, folder)
    session = Session.load(folder / "session.json")
    assert session.complete is True
    return folder, session, ResultsStore.load(folder / "results.npz")
