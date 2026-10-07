"""The store behind results.npz: the pixel-space records of every track (SPEC 8.12; X5, X15).

Tracking puts one `PixelRecord` per track and tracked frame into a `ResultsStore` and saves it to
the run folder; export loads it and derives every world-unit output from it and session.json, so a
new calibration needs no new tracking (SPEC 7). `TrackArrays` is one track as arrays, one row per
tracked frame in frame order: what derive, the flags, the runners and export read.

Units and coordinates: everything is in image pixels, Tracker's convention (SPEC 3.1): (u, v) from
the top-left corner of the frame, u to the right, v downward, the pixel in column c and row r with
its center at (c + 0.5, r + 0.5). Covariances are (mu_uu, mu_uv, mu_vv) in px^2. A mask crop is a
boolean array indexed [row, column] with the (column, row) of its top-left pixel in the full
frame. Frames are video frame numbers. Nothing here knows about mm or seconds.

The file: `version` (a whole number, `schema.RESULTS_VERSION`), and for every track the arrays of
`schema.RESULTS_KEYS` under the keys `<track id>__<name>`. The mask crops of a track lie one after
another in `mask_bits`, each packed by `measure.pack_mask`. It is written without compression and
without pickled objects, through `fileio.atomic_write`.

No Qt, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from outline_tracker import __version__
from outline_tracker.fileio import atomic_write
from outline_tracker.measure import MODES, PixelRecord, unpack_mask
from outline_tracker.schema import RESULTS_KEYS, RESULTS_VERSION, RESULTS_VERSION_KEY, npz_key, split_npz_key

_FRAMES, _MASK_BITS = "frames", "mask_bits"
# a record's field for each array: `frame` is one entry of `frames`
_FIELDS = {key.name: "frame" if key.name == _FRAMES else key.name for key in RESULTS_KEYS}


class ResultsVersionError(ValueError):
    """results.npz has another format version than this tool reads, or none. `found` is the file's
    version (None when it has no usable one), `supported` the version this tool reads."""

    def __init__(self, found, where: str = "The results file"):
        self.found, self.supported = found, RESULTS_VERSION
        if found is None:
            has = "no usable results format version"
            advice = "It is not a results file of outline-tracker, or it is damaged."
        else:
            has = f"results format version {found}"
            advice = ("It was written by a newer outline-tracker: update yours." if found > self.supported else
                      "It was written by an older outline-tracker: use that one to export, or track again.")
        super().__init__(f"{where} has {has}, but this outline-tracker ({__version__}) reads only version "
                         f"{self.supported}. {advice}")


@dataclass(frozen=True, eq=False)
class TrackArrays:
    """One track of the store as arrays: row i is the track's i-th tracked frame, frames ascending.

    The fields are the arrays of `schema.RESULTS_KEYS`, with its names, dtypes and shapes (n rows;
    `mask_bits` holds the bytes of all crops). Positions (u, v) and outline points are in px in
    full-frame image coordinates, pixel centers at +0.5; covariances are (mu_uu, mu_uv, mu_vv) in
    px^2; `frames` are video frame numbers. A lost frame keeps its row: `visible` False, counts 0,
    NaN in the measured numbers, an empty crop (see `measure.PixelRecord`). `len()` is n.
    """

    frames: np.ndarray
    visible: np.ndarray
    area_px: np.ndarray
    u: np.ndarray
    v: np.ndarray
    cov_full: np.ndarray
    cov_core: np.ndarray
    core_u: np.ndarray
    core_v: np.ndarray
    core_frac: np.ndarray
    core_fallback: np.ndarray
    core_r_px: np.ndarray
    outline_px: np.ndarray
    n_components: np.ndarray
    largest_fraction: np.ndarray
    second_fraction: np.ndarray
    cell_px: np.ndarray
    edge: np.ndarray
    mode: np.ndarray
    score: np.ndarray
    mask_offset: np.ndarray
    mask_shape: np.ndarray
    mask_bits: np.ndarray

    def __len__(self) -> int:
        return len(self.frames)

    def row(self, frame: int) -> int:
        """The row of video frame number `frame`. Raises KeyError if that frame was not tracked."""
        row = int(np.searchsorted(self.frames, frame))
        if row == len(self.frames) or self.frames[row] != frame:
            raise KeyError(f"frame {frame} is not among the tracked frames of this track")
        return row

    def mask(self, row: int) -> tuple[np.ndarray, tuple[int, int]]:
        """The mask of row `row` (an index into the arrays, not a frame number), cut to its
        bounding box: (crop, offset). `crop` is a new boolean array indexed [row, column];
        `offset` = (column, row) of its top-left pixel in the full frame, in px. A lost frame gives
        an empty crop of shape (0, 0). Raises IndexError for a row that does not exist."""
        end = int(self._crop_ends[row])
        rows, cols = (int(side) for side in self.mask_shape[row])
        offset = (int(self.mask_offset[row, 0]), int(self.mask_offset[row, 1]))
        return unpack_mask(self.mask_bits[end - _crop_bytes(rows, cols):end], (rows, cols)), offset

    @cached_property
    def _crop_ends(self) -> np.ndarray:
        """For each row, where its crop ends in `mask_bits` (a byte count)."""
        shapes = self.mask_shape.astype(np.int64)
        return np.cumsum(_crop_bytes(shapes[:, 0], shapes[:, 1]))


class ResultsStore:
    """The pixel-space records of every track, in memory; `save` and `load` are results.npz.

    A track is in the store as long as it has a record. All values are in image pixels and video
    frame numbers (see the module's text); the store does no geometry.
    """

    def __init__(self) -> None:
        self._tracks: dict[str, dict[int, PixelRecord]] = {}   # track id -> video frame number -> record

    @property
    def track_ids(self) -> list[str]:
        """The ids of the tracks that have records, in text order (A, A2, AA, B)."""
        return sorted(self._tracks)

    def put(self, track_id: str, record: PixelRecord) -> None:
        """Store `record` (from `measure.measure_mask`, in full-frame px) as frame `record.frame` of
        track `track_id`. A record already stored for that frame is replaced. Frames may come in
        any order. Raises ValueError for a track id that is not a non-empty text, and for a record
        that the file could not hold: a field of another shape than `schema.RESULTS_KEYS` gives
        it, a mode other than "coarse" or "fine", or crop bytes that do not match the crop's shape.
        """
        if not isinstance(track_id, str) or not track_id:
            raise ValueError(f"track id {track_id!r}: it must be a text such as 'A' or 'A2'.")
        _check(record)
        self._tracks.setdefault(track_id, {})[int(record.frame)] = record

    def arrays(self, track_id: str) -> TrackArrays:
        """The records of one track as arrays, one row per tracked frame, frames ascending; px in
        full-frame image coordinates (see `TrackArrays`). The arrays are new ones: changing them
        does not change the store. Raises KeyError for a track without records."""
        if track_id not in self._tracks:
            raise KeyError(f"no track {track_id!r} in the results (tracks: {', '.join(self.track_ids) or 'none'})")
        records = self._tracks[track_id]
        return _to_arrays([records[frame] for frame in sorted(records)])

    def replace_from(self, track_id: str, frame_k: int) -> None:
        """Remove the records of track `track_id` at video frames >= `frame_k`, to make room for a
        new run from there ("Re-track from here", SPEC 6.6). Other tracks are untouched. A track
        with no such frame, or not in the store at all, is left as it is."""
        self._keep(track_id, lambda frame: frame < frame_k)

    def truncate_after(self, track_id: str, frame_k: int) -> None:
        """Remove the records of track `track_id` at video frames > `frame_k`: the track then ends
        at `frame_k` ("End track here", SPEC 6.6). Other tracks are untouched. A track with no such
        frame, or not in the store at all, is left as it is."""
        self._keep(track_id, lambda frame: frame <= frame_k)

    def _keep(self, track_id: str, wanted) -> None:
        records = self._tracks.get(track_id, {})
        for frame in [frame for frame in records if not wanted(frame)]:
            del records[frame]
        if not records:
            self._tracks.pop(track_id, None)

    def save(self, path) -> Path:
        """Write the store to `path` (results.npz of the run folder) through `fileio.atomic_write`:
        `path` holds the old file or the complete new one, never a part. Returns the path that now
        holds the data: `path`, or `<stem>.new<suffix>` next to it when `path` stayed locked by
        another program (Windows), which the caller should report. Values are stored as they are,
        in px and frame numbers."""
        data = {RESULTS_VERSION_KEY: np.array(RESULTS_VERSION, np.int32)}
        for track_id in self.track_ids:
            arrays = self.arrays(track_id)
            data.update((npz_key(track_id, key.name), getattr(arrays, key.name)) for key in RESULTS_KEYS)

        def write(tmp: Path) -> None:
            with open(tmp, "wb") as f:   # an open file: given a path, numpy adds ".npz" to its name
                np.savez(f, **data)

        return atomic_write(path, write)

    @classmethod
    def load(cls, path) -> ResultsStore:
        """Read a results.npz written by `save`. The file is closed when this returns, so it can be
        replaced at once (on Windows an open file cannot). Raises FileNotFoundError,
        ResultsVersionError (another format version, or none; the message names both), or
        ValueError when a track's arrays are missing or do not fit together."""
        path = Path(path)
        with np.load(path) as z:
            if RESULTS_VERSION_KEY not in z.files:
                raise ResultsVersionError(None, str(path))
            version = z[RESULTS_VERSION_KEY]
            if version.shape != () or version.dtype.kind not in "iu":
                raise ResultsVersionError(None, str(path))
            if int(version) != RESULTS_VERSION:
                raise ResultsVersionError(int(version), str(path))
            track_ids = sorted({split[0] for split in map(split_npz_key, z.files) if split is not None})
            columns = {}
            for track_id in track_ids:
                missing = [key.name for key in RESULTS_KEYS if npz_key(track_id, key.name) not in z.files]
                if missing:
                    raise ValueError(f"{path} is damaged: it has no array {npz_key(track_id, missing[0])}.")
                columns[track_id] = {key.name: z[npz_key(track_id, key.name)] for key in RESULTS_KEYS}
        store = cls()
        for track_id, arrays in columns.items():
            try:
                records = _to_records(TrackArrays(**arrays), track_id)
            except ValueError as err:
                raise ValueError(f"{path} is damaged: {err}") from None
            if records:
                store._tracks[track_id] = records
        return store


def _crop_bytes(rows, cols):
    """Bytes that `measure.pack_mask` makes of a crop of `rows` x `cols` pixels (numbers or arrays)."""
    return (rows * cols + 7) // 8


def _check(record: PixelRecord) -> None:
    """Raise ValueError unless the record fits the arrays of results.npz."""
    for key in RESULTS_KEYS:
        if key.name in (_FRAMES, _MASK_BITS):
            continue
        shape = np.shape(getattr(record, key.name))
        if shape != key.shape[1:]:
            raise ValueError(f"{key.name} of the record for frame {record.frame} has shape {shape}; "
                             f"results.npz holds {key.shape[1:]} per frame.")
    if record.mode not in MODES:
        raise ValueError(f"mode = {record.mode!r} in the record for frame {record.frame}: it must be 'coarse' "
                         f"or 'fine'.")
    rows, cols = record.mask_shape
    if np.shape(record.mask_bits) != (_crop_bytes(rows, cols),):
        raise ValueError(f"mask_bits of the record for frame {record.frame} has shape {np.shape(record.mask_bits)}; "
                         f"a crop of {rows} x {cols} px is {_crop_bytes(rows, cols)} bytes.")


def _to_arrays(records: list[PixelRecord]) -> TrackArrays:
    """The given records, in this order, as the arrays of `schema.RESULTS_KEYS`."""
    arrays = {}
    for key in RESULTS_KEYS:
        values = [getattr(record, _FIELDS[key.name]) for record in records]
        if key.name == _MASK_BITS:
            arrays[key.name] = np.concatenate([np.zeros(0, np.uint8), *values]).astype(np.uint8, copy=False)
        else:
            arrays[key.name] = np.array(values, dtype=key.dtype).reshape((len(records), *key.shape[1:]))
    return TrackArrays(**arrays)


def _to_records(arrays: TrackArrays, track_id: str) -> dict[int, PixelRecord]:
    """The records of the given arrays, by video frame number. Raises ValueError, naming the
    array of the file at fault, when the arrays of track `track_id` do not fit together."""
    n = arrays.frames.size
    for key in RESULTS_KEYS:
        column, name = getattr(arrays, key.name), npz_key(track_id, key.name)
        if column.dtype != np.dtype(key.dtype):
            raise ValueError(f"{name} has dtype {column.dtype}, not {key.dtype}.")
        if key.name != _MASK_BITS and column.shape != (n, *key.shape[1:]):
            raise ValueError(f"{name} has shape {column.shape}, not {(n, *key.shape[1:])}.")
    if np.any(np.diff(arrays.frames) <= 0):
        raise ValueError(f"{npz_key(track_id, _FRAMES)} is not in ascending order.")
    ends = arrays._crop_ends
    total = int(ends[-1]) if n else 0
    if arrays.mask_bits.shape != (total,):
        raise ValueError(f"{npz_key(track_id, _MASK_BITS)} has {arrays.mask_bits.size} bytes, and the crops "
                         f"need {total}.")
    fields: dict[str, list] = {}
    for key in RESULTS_KEYS:
        column = getattr(arrays, key.name)
        if key.name == _MASK_BITS:
            fields[key.name] = np.split(column, ends[:-1]) if n else []
        elif key.name in ("mask_offset", "mask_shape"):
            fields[key.name] = [tuple(pair) for pair in column.tolist()]
        else:   # plain numbers for one value per frame, arrays for cov_* and outline_px
            fields[_FIELDS[key.name]] = column.tolist() if column.ndim == 1 else list(column)
    return {fields["frame"][i]: PixelRecord(**{name: values[i] for name, values in fields.items()}) for i in range(n)}
