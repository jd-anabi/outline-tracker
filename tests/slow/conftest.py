"""Fixtures for the tests with the real model (tests/slow/): the model, loaded once for each test module
that asks for it, and its processor without weights. pytest finds a fixture by its name; nothing
imports this file (tests/test_repo_rules.py says why). The fixtures for the tests of every folder are
in tests/conftest.py; plain helpers are in tests/slow/pipeline_helpers.py.

Every run of pytest loads this file, also a run of the fast tests, which collects tests/slow and
deselects it. So torch and transformers are imported inside the fixtures, never when the file is
loaded.
"""

import pytest


@pytest.fixture(scope="module")
def processor():
    """The processor that `load_edgetam` builds: no weights and no download."""
    from transformers import Sam2ImageProcessor, Sam2VideoProcessor, Sam2VideoVideoProcessor

    return Sam2VideoProcessor(image_processor=Sam2ImageProcessor(), video_processor=Sam2VideoVideoProcessor())


@pytest.fixture(scope="module")
def loaded():
    """(model, processor): the real EdgeTAM, loaded once for each test module that asks for it, as
    while each module had its own copy of this fixture."""
    from outline_tracker.segmenter import hf

    return hf.load_model("edgetam")
