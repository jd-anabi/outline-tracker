"""Records and checks shared by the tests of the results store (tests/test_results.py: the store,
saving and loading; tests/test_results_file.py: what may go wrong with the file).

The store is a container, so the truth is what was put in. `made_up` gives a record with a
different, known number in every field, `measured` and `lost` give what measure_mask really
returns, and `assert_holds` checks arrays against a list of records: every expected array is
written out here from the records (one row per record, in the dtype that `schema.RESULTS_KEYS`
names), never taken from the store. `read_npz` and `write_npz` read and make npz files with numpy
alone.

Records are in image pixels, Tracker's convention: (u, v) from the top-left corner of the frame,
pixel centers at +0.5; a mask crop is indexed [row, column] and its offset is (column, row) of its
top-left pixel in the full frame.
"""

import re

import analytic_shapes as shapes
import numpy as np

from outline_tracker import schema
from outline_tracker.measure import PixelRecord, measure_mask
from outline_tracker.results import ResultsStore, TrackArrays

FULL_HD = (0, 0, 1920, 1080)
DISK_ORIGIN = (300, 200)  # (column, row) of the raster the measured disks are drawn on


def crop(seed):
    """The mask crop of the made-up record `seed`: boolean [row, column], 1 to 12 px on each side,
    so that most crops do not fill a whole number of bytes."""
    rng = np.random.default_rng(1000 + seed)
    return rng.random((int(rng.integers(1, 13)), int(rng.integers(1, 13)))) < 0.5


def made_up(frame, seed, mode="coarse"):
    """A record for video frame `frame` with a different, known number in every field. It is not a
    measurement: a column stored under the wrong name, or a row out of place, shows at once."""
    rng = np.random.default_rng(seed)
    mask = crop(seed)
    return PixelRecord(
        frame=frame, visible=True, area_px=int(rng.integers(1, 50_000)),
        u=float(rng.uniform(0, 1920)), v=float(rng.uniform(0, 1080)),
        cov_full=rng.uniform(-50, 50, 3), cov_core=rng.uniform(-50, 50, 3),
        core_u=float(rng.uniform(0, 1920)), core_v=float(rng.uniform(0, 1080)),
        core_frac=float(rng.uniform(0, 1)), core_fallback=bool(rng.integers(0, 2)),
        core_r_px=int(rng.integers(1, 9)), outline_px=rng.uniform(0, 1920, (256, 2)).astype(np.float32),
        n_components=int(rng.integers(1, 5)), largest_fraction=float(rng.uniform(0.5, 1)),
        second_fraction=float(rng.uniform(0, 1)), cell_px=float(rng.uniform(0.3, 8)),
        edge=bool(rng.integers(0, 2)), mode=mode, score=float(np.float32(rng.uniform(-5, 20))),
        mask_offset=(int(rng.integers(0, 1900)), int(rng.integers(0, 1000))),
        mask_shape=(mask.shape[0], mask.shape[1]), mask_bits=np.packbits(mask.ravel()),
    )


def disk_distance(center, radius):
    """Signed distance (px) of a disk on the 60 x 80 px raster at DISK_ORIGIN; `center` is (u, v)
    in full-frame px."""
    u, v = shapes.pixel_centers(60, 80, origin=DISK_ORIGIN)
    return shapes.disk(u, v, center, radius)


def measured(frame, center, mode="coarse", radius=9.0):
    """What measure_mask really returns for a disk: a record with a real crop and outline."""
    result = shapes.to_result(disk_distance(center, radius), origin=DISK_ORIGIN, score=2.5)
    return measure_mask(result, frame, FULL_HD, mode)


def lost(frame, mode="coarse"):
    """What measure_mask really returns for an object that was not found on `frame`."""
    return measure_mask(shapes.pixel_result(np.zeros((0, 0), bool), score=-3.0), frame, FULL_HD, mode)


def two_tracks():
    """(store, records): track A, coarse, on frames 10, 12, ..., 22 and track B, fine, on frames
    14, 16, ..., 30, each with a lost frame in the middle. `records[id]` is in frame order."""
    records = {
        "A": [made_up(10, 1), measured(12, (340.3, 230.6)), lost(14), made_up(16, 2), made_up(18, 3),
              measured(20, (344.1, 228.2)), made_up(22, 4)],
        "B": [lost(22, "fine") if frame == 22 else made_up(frame, 100 + frame, "fine") for frame in range(14, 31, 2)],
    }
    store = ResultsStore()
    for track_id, rows in records.items():
        for record in rows:
            store.put(track_id, record)
    return store, records


def same(a, b):
    """Equal arrays: the same dtype, shape and values, NaN equal to NaN."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b))


def expected_column(records, key):
    """The array of results.npz for `key`, written out from the records: one row per record."""
    if key.name == "mask_bits":
        return np.concatenate([np.zeros(0, np.uint8)] + [record.mask_bits for record in records])
    name = "frame" if key.name == "frames" else key.name
    rows = [np.asarray(getattr(record, name), dtype=key.dtype) for record in records]
    assert all(row.shape == key.shape[1:] for row in rows), key.name
    return np.stack(rows)


def assert_holds(arrays, records):
    """`arrays` is exactly the given records, in this order, crops included."""
    assert isinstance(arrays, TrackArrays)
    assert len(arrays) == len(records)
    for key in schema.RESULTS_KEYS:
        column = getattr(arrays, key.name)
        assert column.dtype == np.dtype(key.dtype), key.name
        assert same(column, expected_column(records, key)), key.name
    for row, record in enumerate(records):
        rows, cols = record.mask_shape
        mask, offset = arrays.mask(row)
        assert mask.dtype == bool and mask.shape == (rows, cols)
        assert np.array_equal(mask, np.unpackbits(record.mask_bits, count=rows * cols).reshape(rows, cols))
        assert offset == record.mask_offset and all(type(x) is int for x in offset)


def read_npz(path):
    """Every array of an npz file, by key, read with numpy alone; the file is closed again."""
    with np.load(path) as z:
        return {name: z[name] for name in z.files}


def write_npz(path, data):
    """Write the arrays `data` (key -> array) as an npz file at exactly `path`, and return `path`."""
    with open(path, "wb") as f:
        np.savez(f, **data)
    return path


def names(folder):
    """The names of the files in `folder`, sorted."""
    return sorted(p.name for p in folder.iterdir())


def names_number(message, number):
    """True when the text holds this whole number on its own, not as part of "0.1.0" or "12"."""
    return re.search(rf"(?<![\d.]){number}(?!\d|\.\d)", message) is not None
