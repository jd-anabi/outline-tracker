"""Where the GUI's panels find the edit API and how they call it (task A17 of the plan, "Produces").

The plan names one file for it, outline_tracker/tracking.py, eleven functions, and for five of them
the arguments. The functions are written in outline_tracker/tracking_edit.py (SPEC 12 splits a file
that grows past about 400 lines); outline_tracker.tracking gives the same objects under the same
names, so that both imports work.

Expected values are the plan's own text: its names, and its arguments in its order, which come
first in every call. An argument after them (the run folder, for a function that saves
results.npz) and what a function returns are not stated here: what the functions do is in
tests/test_corrections.py and tests/test_tracking_edit.py. No units, no coordinates.
"""

import inspect

import pytest

from outline_tracker import tracking, tracking_edit

# "Produces" of task A17, in its order.
PLAN_NAMES = ["retrack_from", "end_track", "new_piece", "next_track_id", "add_object", "remove_object", "add_prompt",
              "undo_prompt", "set_head", "pending_runs", "flags_table"]
PLAN_ARGUMENTS = {
    "retrack_from": ("session", "store", "track_ids", "frame_k", "prompts"),
    "end_track": ("session", "store", "track_id", "frame_k"),
    "new_piece": ("session", "parent_id", "frame_m", "prompts"),
    "next_track_id": ("session",),
    "flags_table": ("run_folder",),
}


@pytest.mark.parametrize("name", [*PLAN_NAMES, "Edit"])  # Edit: what an edit returns, for the caller's annotations
def test_the_edit_api_is_importable_from_tracking_the_file_the_plan_names(name):
    assert getattr(tracking, name) is getattr(tracking_edit, name)
    assert name in tracking.__all__
    assert name in tracking_edit.__all__


@pytest.mark.parametrize("name", PLAN_ARGUMENTS)
def test_the_plans_arguments_come_first_in_the_plans_order(name):
    asked = PLAN_ARGUMENTS[name]
    takes = inspect.signature(getattr(tracking, name)).parameters
    assert tuple(takes)[:len(asked)] == asked
    # ... and can be given by position, as the plan writes its calls
    by_position = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    assert all(takes[argument].kind in by_position for argument in asked)


def test_tracking_still_gives_the_job_api_next_to_the_edit_api():
    # what tasks A15 and A16 produce: jobs, planning, the frame-hash guard, the fine window
    job_api = ["AUTOSAVE_EVERY", "Callbacks", "FrameHashMismatch", "FrameHashUpdate", "Job", "RunPlan",
               "SessionChanges", "dish_box", "fine_window", "fine_window_px", "partial_tracks", "plan_runs", "run_job"]
    for name in job_api:
        assert name in tracking.__all__ and hasattr(tracking, name), name
    assert len(tracking.__all__) == len(set(tracking.__all__))
