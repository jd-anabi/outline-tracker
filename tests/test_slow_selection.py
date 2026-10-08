"""Which slow tests need no weights: the ones that CI runs (docs/ROADMAP.md, W1 step 8).

A slow test carries the mark `weights` when it loads the real model: it, or a fixture it uses, calls
`hf.load_model`, `from_tracker.load_segmenter`, `selftest` without a stand-in, `HFSegmenter` without
a given model or `from_pretrained`, or reads the model's cache folder. Such a test runs on a
computer that has the model, never in CI. The slow tests without that mark need torch and
transformers and no file of the model. Every CI job runs them after the fast tests, offline and
with a model folder that holds nothing (.github/workflows/tests.yml):

    uv run pytest -m "slow and not weights" tests/slow

That selection fails open: a slow test that loads the model and has no mark is selected.
`WITHOUT_WEIGHTS` is the list of the tests that were read and found to load no model, so that an
unread test is noticed here, on every computer, before it fails in CI.

No quantities here: no units and no coordinates.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The slow tests that load no model, each read with its fixtures for the calls named above (2026-10-08).
WITHOUT_WEIGHTS = [
    # arithmetic on made-up series of readings: numpy alone
    "tests/slow/test_memory.py::test_growth_is_not_moved_by_single_readings_and_sees_a_slow_leak",
    # the fixture `processor` (made from its classes, from no file) and a stand-in for the network
    "tests/slow/test_real_model.py::test_prompt_tensors_for_1_3_and_2_points_have_no_padding",
    "tests/slow/test_real_model.py::test_start_stores_only_the_points_inside_the_image",
    "tests/slow/test_real_model.py::test_preview_uses_a_session_of_its_own",
    "tests/slow/test_real_model.py::test_step_fallback_opens_a_new_session_on_cpu",
    # a child process that imports torch and counts its threads
    "tests/slow/test_real_model.py::test_reserve_ui_thread_with_the_real_torch",
    # the fixture `processor` and a stand-in that returns given logits
    "tests/slow/test_regression_reference.py::test_prompt_tensors_have_one_point_and_no_padding",
    "tests/slow/test_regression_reference.py::test_session_equals_the_one_the_reference_builds",
    "tests/slow/test_regression_reference.py::test_per_object_logits_give_the_reference_masks[random]",
    "tests/slow/test_regression_reference.py::test_per_object_logits_give_the_reference_masks[blobs]",
]


def _collected(marks: str, *where: str) -> list[str]:
    """The node ids that pytest collects for the marker expression `marks`, sorted. A child process, as
    the command line does it: `python -m pytest --collect-only -q -p no:cacheprovider -m MARKS`, followed
    by `where` (folders of the repository; none: the whole suite). Nothing is run."""
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-m", marks, *where],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    return sorted(line for line in done.stdout.splitlines() if line.startswith("tests/") and "::" in line)


def test_the_slow_tests_without_weights_are_exactly_these():
    """The command of CI's second step selects `WITHOUT_WEIGHTS` and nothing else.

    A new slow test that nobody has sorted fails here, and that is the purpose of this test. Read the
    new test and its fixtures. If it loads the model, mark it `weights` (in the file's `pytestmark`
    when every test of the file does). If it does not, add it to `WITHOUT_WEIGHTS` and see it pass
    with no model in reach:
    `HF_HUB_OFFLINE=1 SHRIMP_MODEL_CACHE=<an empty folder> uv run pytest -m "slow and not weights" tests/slow`
    (a computer that has run the model keeps Meta's file in the Hub's cache: there, point `HF_HOME`
    at an empty folder too). A test is never marked or listed only to make this one pass.
    """
    without = _collected("slow and not weights", "tests/slow")
    assert without == sorted(WITHOUT_WEIGHTS)

    with_weights = _collected("slow and weights", "tests/slow")
    # of the whole suite, not of tests/slow: a slow test in another folder would be in no step of CI
    everything = _collected("slow")
    assert with_weights == sorted(set(everything) - set(WITHOUT_WEIGHTS))  # the rest
    assert set(without) & set(with_weights) == set()
    assert sorted(without + with_weights) == everything
