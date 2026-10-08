"""A new conversion of Meta's EdgeTAM checkpoint gives the frozen weights (docs/ROADMAP.md, W1 step 5).

`outline_tracker/segmenter/edgetam_convert.py` renames the weights of Meta's `edgetam.pt` for
transformers and saves them as `model.safetensors`. The tool does that once per computer, on first
use; afterwards it reads the saved file, and so does every other slow test. Here the checkpoint is
converted again, into an empty model folder, by the function the tool calls on first use
(`hf.load_model`), and the new file is compared with tests/data/edgetam_weights.txt: its SHA-256
and its size in bytes.

What this holds: the two renaming tables (`KEYS_TO_MODIFY_MAPPING`, `PERCEIVER`), `_renumber`,
`convert_state_dict`, `edgetam_config` and `load_edgetam`. A weight that gets a name the model does
not have stops the conversion ("EdgeTAM conversion failed"); weights under each other's names give
another file.

The expected value is a frozen one (docs/ROADMAP.md, section 2, rule 2): the frozen file was
written by a run in which last week's code still agreed with the package, and its header says how.
On the machine that froze it, two conversions into two folders gave the same bytes, those of the
frozen file (2026-10-08).

The model folder: `SHRIMP_MODEL_CACHE` names a folder under pytest's temporary folder while the
test runs, so this computer's own model folder (~/.cache/shrimp-models) is not written; the test
asserts that its files are the same afterwards. Meta's checkpoint is read from the Hugging Face
cache; a computer that does not have it downloads it (56 MB).

The machine rule (`frozen_helpers.same_machine`), as for the weights test of
tests/slow/test_frozen_reference.py: the hash and the size are asserted on the machine that froze
them. On another machine the conversion has to succeed, and then the test is skipped; its reason
names both hashes.

The line printed with the prefix `VALIDATION` holds both hashes and sizes; show it with
`uv run pytest -m slow tests/slow/test_edgetam_conversion.py -q -rPs` (`s` adds the reason of a
skipped test). The test is slow: it loads torch and builds the model (about 5 s here).

No quantities here: no units and no coordinates.
"""

from __future__ import annotations

import pytest
from frozen_helpers import WEIGHTS, machine_here, machine_of, read_frozen, same_machine

pytestmark = pytest.mark.slow

MODEL = "edgetam"


def _files(folder) -> list[tuple[str, int, int]] | None:
    """What a folder holds: (name, size in bytes, time of the last change in ns) of each entry, by
    name; None when the folder does not exist."""
    if not folder.is_dir():
        return None
    return sorted((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in folder.iterdir())


def test_a_new_conversion_gives_the_frozen_weights(tmp_path, monkeypatch):
    from outline_tracker.segmenter import edgetam_convert, hf

    own = edgetam_convert.edgetam_cache()  # this computer's own model folder: looked at, never written
    own_before = _files(own)
    monkeypatch.setenv("SHRIMP_MODEL_CACHE", str(tmp_path / "models"))
    folder = tmp_path / "models" / "edgetam"
    assert edgetam_convert.edgetam_cache() == folder
    assert hf.weights_file(MODEL) is None  # nothing is converted there, so loading has to convert

    hf.load_model(MODEL)

    new = hf.weights_file(MODEL)
    assert new == folder / "model.safetensors"  # the conversion was saved, in the temporary folder
    assert _files(own) == own_before
    header, rows = read_frozen(WEIGHTS)
    frozen_weights = dict(row.split(": ", 1) for row in rows)
    sha256, size = hf.file_sha256(new), new.stat().st_size
    print(f"VALIDATION frozen numbers, a new conversion: {size} bytes, sha256 {sha256}; "
          f"frozen: {frozen_weights['bytes']} bytes, sha256 {frozen_weights['sha256']}")
    if not same_machine(header, machine_here()):
        made = machine_of(header)
        pytest.skip(f"The weights' hash is asserted on the machine that froze it ({made['chip']}, {made['platform']} "
                    f"{made['architecture']}, torch {made['torch']}). This is another one: the conversion succeeded "
                    f"here, and its file may differ. SHA-256 here: {sha256}; frozen: {frozen_weights['sha256']}.")
    assert sha256 == frozen_weights["sha256"]
    assert size == int(frozen_weights["bytes"])
