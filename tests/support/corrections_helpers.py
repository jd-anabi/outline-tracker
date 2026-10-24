"""Checks that the tests of the corrections share: what stayed the same, and when a correction was made.

`assert_rows_identical` and `assert_tracks_identical` hold two readings of results.npz (`read_npz`) against each
other, `assert_recorded_now` holds the time of a correction. They serve tests/tracking/test_corrections.py and
tests/tracking/test_corrections_new_piece.py. No units or coordinates here.
"""

from datetime import datetime

import numpy as np

from outline_tracker.core.schema import RESULTS_KEYS, npz_key

def assert_rows_identical(after, before, track_id, rows):
    """The first `rows` rows of a track are the same in two readings of results.npz (`read_npz`):
    every array, and the bytes of the mask crops of those rows."""
    for key in RESULTS_KEYS:
        new, old = after[npz_key(track_id, key.name)], before[npz_key(track_id, key.name)]
        assert new.dtype == old.dtype, key.name
        if key.name == "mask_bits":
            shapes = before[npz_key(track_id, "mask_shape")][:rows].astype(np.int64)
            count = int(((shapes[:, 0] * shapes[:, 1] + 7) // 8).sum())
            new, old = new[:count], old[:count]
        else:
            new, old = new[:rows], old[:rows]
        np.testing.assert_array_equal(new, old, err_msg=f"{track_id}: {key.name}")


def assert_tracks_identical(after, before, track_ids):
    """Every array of the named tracks is the same in two readings of results.npz."""
    for track_id in track_ids:
        for key in RESULTS_KEYS:
            name = npz_key(track_id, key.name)
            assert after[name].dtype == before[name].dtype and after[name].shape == before[name].shape, name
            np.testing.assert_array_equal(after[name], before[name], err_msg=name)


def assert_recorded_now(correction):
    """A correction's time is ISO 8601 with the offset from UTC, and it is now (within a minute)."""
    when = datetime.fromisoformat(correction.time)
    assert when.utcoffset() is not None
    assert abs((datetime.now().astimezone() - when).total_seconds()) < 60
