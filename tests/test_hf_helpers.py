"""The segmenter protocol and the torch-free parts of the Hugging Face backend (SPEC 6.2, 12).

Nothing here loads torch: the model is replaced by small stand-ins, and the methods kept from last
week's `TransformersSegmenter` are called on plain objects. Tests with the real processor or the
real model are in tests/slow/test_regression_reference.py.

Coordinates: masks are arrays indexed [row, column]; prompt points and offsets are in px of the
image given to the segmenter, Tracker's convention (pixel centers at +0.5, SPEC 3.1).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from outline_tracker.measure import mask_center

REPO = Path(__file__).resolve().parents[1]
HEAVY_MODULES = ["PySide6", "pyqtgraph", "timm", "torch", "torchvision", "transformers"]

# SHA-256 of the three bytes "abc" (the test vector of FIPS 180-2, appendix B.1).
SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# ---------------------------------------------------------------------------------------------
# Imports stay light


def test_importing_the_segmenter_modules_does_not_load_torch():
    # In a subprocess: pytest-qt has already imported PySide6 into this process.
    code = (
        "import json, sys\n"
        "import outline_tracker.segmenter\n"
        "import outline_tracker.segmenter.base\n"
        "import outline_tracker.segmenter.edgetam_convert\n"
        "import outline_tracker.segmenter.hf as hf\n"
        "assert callable(hf.HFSegmenter) and callable(hf.load_model)\n"
        "print(json.dumps(sorted(m for m in json.loads(sys.argv[1]) if m in sys.modules)))\n"
    )
    done = subprocess.run([sys.executable, "-c", code, json.dumps(HEAVY_MODULES)], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == []


# ---------------------------------------------------------------------------------------------
# base.py: the protocol and its two records, as SPEC 12 writes them


def _fields(cls) -> list[tuple[str, str]]:
    return [(f.name, f.type) for f in dataclasses.fields(cls)]


def test_prompt_and_result_have_the_fields_of_spec_12():
    from outline_tracker.segmenter.base import MaskResult, ObjectPrompt

    assert _fields(ObjectPrompt) == [
        ("obj_id", "str"),
        ("points_px", "list[tuple[float, float]]"),
        ("labels", "list[int]"),
        ("box_px", "tuple[float, float, float, float] | None"),
    ]
    assert ObjectPrompt("A", [(1.0, 2.0)], [1]).box_px is None
    assert _fields(MaskResult) == [
        ("obj_id", "str"),
        ("offset", "tuple[int, int]"),
        ("mask", "np.ndarray"),
        ("logits", "np.ndarray | None"),
        ("score", "float | None"),
    ]


def test_segmenter_protocol_has_start_step_preview_close():
    from outline_tracker.segmenter.base import Segmenter

    assert getattr(Segmenter, "_is_protocol", False)
    parameters = {name: list(inspect.signature(getattr(Segmenter, name)).parameters)
                  for name in ("start", "step", "preview", "close")}
    assert parameters == {
        "start": ["self", "image", "prompts"],
        "step": ["self", "image"],
        "preview": ["self", "image", "prompts"],  # X21
        "close": ["self"],
    }


def _frame_with_block(rows: slice, cols: slice, shape=(300, 400)):
    """A mask with one filled block, and logits that are positive exactly on the block."""
    mask = np.zeros(shape, bool)
    mask[rows, cols] = True
    rng = np.random.default_rng(1)
    logits = np.where(mask, 1.0, -1.0) * rng.uniform(0.5, 9.0, shape)
    return mask, logits.astype(np.float32)


def test_crop_is_the_bounding_box_plus_8_px():
    from outline_tracker.segmenter.base import MaskResult, crop_to_bbox

    mask, logits = _frame_with_block(slice(100, 110), slice(200, 220))  # rows 100-109, columns 200-219
    result = crop_to_bbox(mask, logits)
    assert isinstance(result, MaskResult)
    assert result.offset == (192, 92)  # (col0, row0) = (200 - 8, 100 - 8)
    assert all(type(v) is int for v in result.offset)
    assert result.mask.shape == (10 + 16, 20 + 16)
    assert result.mask.dtype == bool and result.logits.dtype == np.float32
    assert np.array_equal(result.mask, mask[92:118, 192:228])
    assert np.array_equal(result.logits, logits[92:118, 192:228])
    assert np.array_equal(result.logits > 0, result.mask)
    assert (result.obj_id, result.score) == ("", None)

    named = crop_to_bbox(mask, logits, pad=8, obj_id="B", score=3.5)
    assert (named.obj_id, named.score, named.offset) == ("B", 3.5, (192, 92))


def test_crop_pad_is_adjustable_and_clipped_at_the_image_border():
    from outline_tracker.segmenter.base import crop_to_bbox

    mask, logits = _frame_with_block(slice(0, 5), slice(395, 400))  # the top-right corner of 300 x 400
    result = crop_to_bbox(mask, logits)
    assert result.offset == (387, 0)
    assert result.mask.shape == (5 + 8, 5 + 8)
    assert np.array_equal(result.mask, mask[0:13, 387:400])

    tight = crop_to_bbox(mask, logits, pad=0)
    assert tight.offset == (395, 0) and tight.mask.shape == (5, 5) and tight.mask.all()


def test_crop_keeps_the_position_and_restores_the_full_mask():
    from outline_tracker.segmenter.base import crop_to_bbox

    rng = np.random.default_rng(2)
    mask = np.zeros((300, 400), bool)
    mask[120:160, 30:95] = rng.random((40, 65)) < 0.4  # a ragged object
    result = crop_to_bbox(mask, None)
    assert result.logits is None

    col0, row0 = result.offset
    restored = np.zeros_like(mask)
    restored[row0:row0 + result.mask.shape[0], col0:col0 + result.mask.shape[1]] = result.mask
    assert np.array_equal(restored, mask)

    # A position is mask_center of the crop plus its offset.
    u, v, area = mask_center(result.mask)
    u_full, v_full, area_full = mask_center(mask)
    assert (u + col0, v + row0) == pytest.approx((u_full, v_full), abs=1e-9)
    assert area == area_full == int(mask.sum())


def test_crop_does_not_keep_the_full_frame_alive():
    # SPEC 6.5: no full-frame float array survives the current frame, so a crop owns its data.
    from outline_tracker.segmenter.base import crop_to_bbox

    mask, logits = _frame_with_block(slice(100, 110), slice(200, 220))
    result = crop_to_bbox(mask, logits)
    assert result.mask.base is None and result.logits.base is None
    assert not np.shares_memory(result.logits, logits) and not np.shares_memory(result.mask, mask)


def test_an_empty_mask_gives_an_empty_result():
    from outline_tracker.segmenter.base import crop_to_bbox

    mask = np.zeros((300, 400), bool)
    result = crop_to_bbox(mask, np.full(mask.shape, -1024.0, np.float32), obj_id="A", score=-2.0)
    assert result.offset == (0, 0)
    assert result.mask.shape == (0, 0) and result.mask.dtype == bool
    assert result.logits.shape == (0, 0) and result.logits.dtype == np.float32
    assert (result.obj_id, result.score) == ("A", -2.0)
    u, v, area = mask_center(result.mask)
    assert np.isnan(u) and np.isnan(v) and area == 0
    assert crop_to_bbox(mask, None).logits is None


# ---------------------------------------------------------------------------------------------
# hf.py: prompts, one add-prompt call per object


def test_prompt_lists_have_the_nesting_image_object_point():
    from outline_tracker.segmenter import hf
    from outline_tracker.segmenter.base import ObjectPrompt

    # numpy scalars must not reach the processor: np.float32 and np.int64 raise TypeError there.
    prompt = ObjectPrompt("A", [(np.float32(700.5), np.float64(500.5))], [np.int64(1)])
    points, labels = hf.prompt_lists(prompt)
    assert points == [[[[700.5, 500.5]]]]  # [image][object][point][x, y]
    assert labels == [[[1]]]  # [image][object][point]
    x, y = points[0][0][0]
    assert type(x) is float and type(y) is float
    assert type(labels[0][0][0]) is int


class RecordingProcessor:
    """Stands in for Sam2VideoProcessor: records every add-prompt call and, like the real one
    (`inference_session.obj_with_new_inputs = obj_ids`), overwrites the session's list each time."""

    def __init__(self):
        self.calls = []

    def add_inputs_to_inference_session(self, inference_session, frame_idx, obj_ids, input_points=None,
                                        input_labels=None, original_size=None):
        self.calls.append({"frame_idx": frame_idx, "obj_ids": obj_ids, "input_points": input_points,
                           "input_labels": input_labels, "original_size": original_size})
        inference_session.obj_with_new_inputs = obj_ids


def test_prompts_are_added_with_one_call_per_object():
    from outline_tracker.segmenter import hf
    from outline_tracker.segmenter.base import ObjectPrompt

    clicks = [(700.5, 500.5), (1000.5, 600.5), (1400.5, 350.5)]
    prompts = [ObjectPrompt(name, [click], [1]) for name, click in zip("ABC", clicks)]
    processor, session, size = RecordingProcessor(), SimpleNamespace(obj_with_new_inputs=[]), object()

    ids = hf.add_prompts(processor, session, prompts, size)

    assert ids == [1, 2, 3]  # integers 1..N, as last week
    assert len(processor.calls) == 3
    for k, (call, (x, y)) in enumerate(zip(processor.calls, clicks)):
        assert call["frame_idx"] == 0
        assert call["obj_ids"] == [k + 1]
        assert call["input_points"] == [[[[x, y]]]]
        assert call["input_labels"] == [[[1]]]
        assert call["original_size"] is size
    # Each call overwrote the list of objects with new inputs: afterwards it names all of them,
    # in a list of its own (the model removes entries from it while it runs).
    assert session.obj_with_new_inputs == [1, 2, 3]
    assert session.obj_with_new_inputs is not ids
    assert all(session.obj_with_new_inputs is not call["obj_ids"] for call in processor.calls)


# ---------------------------------------------------------------------------------------------
# hf.py: kept from last week's TransformersSegmenter


def test_pruning_keeps_at_most_21_results_per_object():
    from outline_tracker.segmenter import hf

    assert hf.KEEP_FRAMES == 20
    objects = (0, 1)
    session = SimpleNamespace(
        processed_frames={},
        output_dict_per_obj={obj: {"cond_frame_outputs": {0: "click frame"}, "non_cond_frame_outputs": {}}
                             for obj in objects},
        frames_tracked_per_obj={obj: {} for obj in objects},
    )
    segmenter = SimpleNamespace(session=session, index=0)
    for index in range(60):  # what the model stores per frame, then the pruning
        segmenter.index = index
        session.processed_frames[index] = f"image {index}"
        for obj in objects:
            if index > 0:
                session.output_dict_per_obj[obj]["non_cond_frame_outputs"][index] = f"result {index}"
            session.frames_tracked_per_obj[obj][index] = {"reverse": False}
        hf.HFSegmenter._prune(segmenter)

        kept = list(range(max(0, index - 20), index + 1))  # the current frame and the 20 before it
        for obj in objects:
            store = session.output_dict_per_obj[obj]
            assert len(store["non_cond_frame_outputs"]) <= 21
            assert sorted(store["non_cond_frame_outputs"]) == [k for k in kept if k > 0]
            assert store["cond_frame_outputs"] == {0: "click frame"}  # the click frame is never forgotten
            assert sorted(session.frames_tracked_per_obj[obj]) == kept
        # Earlier images are emptied, not removed: the session's frame count stays right.
        assert list(session.processed_frames) == list(range(index + 1))
        assert [k for k, image in session.processed_frames.items() if image is not None] == [index]


def _fake_torch(cuda: bool, mps: bool | None):
    backends = SimpleNamespace() if mps is None else SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda), backends=backends)


@pytest.mark.parametrize("cuda, mps, expected", [
    (True, True, "cuda"),
    (True, False, "cuda"),
    (False, True, "mps"),
    (False, False, "cpu"),
    (False, None, "cpu"),  # a torch build without the mps backend
])
def test_device_order_is_cuda_then_mps_then_cpu(cuda, mps, expected):
    from outline_tracker.segmenter import hf

    segmenter = SimpleNamespace(torch=_fake_torch(cuda, mps))
    assert hf.HFSegmenter._pick_device(segmenter, "auto") == expected
    for named in ("cpu", "mps", "cuda"):
        assert hf.HFSegmenter._pick_device(segmenter, named) == named


def test_start_falls_back_from_the_apple_gpu_to_the_processor(capsys):
    from outline_tracker.segmenter import hf

    calls = []

    def forward(image, prompts=None):
        calls.append(("forward", segmenter.device, image, prompts))
        if segmenter.device == "mps":
            raise NotImplementedError("The operator 'aten::x' is not currently implemented for the MPS device")
        return ["masks"]

    model = SimpleNamespace(to=lambda device: calls.append(("to", device)))
    segmenter = SimpleNamespace(device="mps", model=model, _forward=forward)

    assert hf.HFSegmenter.start(segmenter, "image", ["prompt"]) == ["masks"]
    assert segmenter.device == "cpu"
    assert calls == [("forward", "mps", "image", ["prompt"]), ("to", "cpu"), ("forward", "cpu", "image", ["prompt"])]
    assert "The Apple GPU failed (NotImplementedError); using the processor instead." in capsys.readouterr().out


def test_start_does_not_hide_an_error_on_other_devices():
    from outline_tracker.segmenter import hf

    def forward(image, prompts=None):
        raise ValueError("bad prompt")

    segmenter = SimpleNamespace(device="cpu", model=None, _forward=forward)
    with pytest.raises(ValueError, match="bad prompt"):
        hf.HFSegmenter.start(segmenter, "image", ["prompt"])
    assert segmenter.device == "cpu"


def test_constructor_arguments():
    from outline_tracker.segmenter import hf

    parameters = inspect.signature(hf.HFSegmenter).parameters
    defaults = {name: p.default for name, p in parameters.items()}
    # model_key, device and the hidden edgetam_checkpoint in last week's order; model and processor
    # let one loaded model serve several segmenters.
    assert defaults == {"model_key": "edgetam", "device": "auto", "edgetam_checkpoint": None,
                        "model": None, "processor": None}
    assert set(hf.MODELS) == {"edgetam", "sam2", "sam2-small"}  # sam2-small stays a hidden option


# ---------------------------------------------------------------------------------------------
# hf.py: loading


def test_an_oserror_while_loading_says_the_first_run_needs_internet_once():
    from outline_tracker.segmenter import hf

    @hf.needs_internet_once
    def load(model_key, edgetam_checkpoint=None):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

    with pytest.raises(OSError) as caught:
        load("edgetam")
    message = str(caught.value)
    assert "the first run needs internet once" in message
    assert "edgetam" in message
    assert "We couldn't connect to 'https://huggingface.co'" in message  # the cause stays readable
    assert isinstance(caught.value.__cause__, OSError) and caught.value.__cause__ is not caught.value


def test_loading_passes_results_and_other_errors_through():
    from outline_tracker.segmenter import hf

    @hf.needs_internet_once
    def load(model_key, edgetam_checkpoint=None):
        if model_key == "nonsense":
            raise ValueError("Unknown model 'nonsense'")
        return model_key, edgetam_checkpoint

    assert load("edgetam", "weights.pt") == ("edgetam", "weights.pt")
    with pytest.raises(ValueError, match="Unknown model 'nonsense'"):
        load("nonsense")


def test_file_sha256_of_a_known_file(tmp_path):
    from outline_tracker.segmenter import hf

    path = tmp_path / "three bytes.bin"
    path.write_bytes(b"abc")
    assert hf.file_sha256(path) == SHA256_ABC


def test_edgetam_cache_folder_is_last_weeks(tmp_path, monkeypatch):
    from outline_tracker.segmenter import edgetam_convert

    monkeypatch.delenv("SHRIMP_MODEL_CACHE", raising=False)
    assert edgetam_convert.edgetam_cache() == Path.home() / ".cache" / "shrimp-models" / "edgetam"
    monkeypatch.setenv("SHRIMP_MODEL_CACHE", str(tmp_path / "my cache"))
    assert edgetam_convert.edgetam_cache() == tmp_path / "my cache" / "edgetam"


def test_weights_file_of_edgetam_is_the_converted_copy(tmp_path, monkeypatch):
    from outline_tracker.segmenter import hf

    monkeypatch.setenv("SHRIMP_MODEL_CACHE", str(tmp_path))
    assert hf.weights_file("edgetam") is None  # nothing converted yet
    saved = tmp_path / "edgetam" / "model.safetensors"
    saved.parent.mkdir()
    saved.write_bytes(b"abc")
    assert hf.weights_file("edgetam") == saved
    assert hf.file_sha256(hf.weights_file("edgetam")) == SHA256_ABC


def test_weights_file_of_sam2_is_in_the_hugging_face_cache(tmp_path, monkeypatch):
    from huggingface_hub import constants

    from outline_tracker.segmenter import hf

    # The Hub's cache layout: <cache>/models--<owner>--<name>/refs/main names the snapshot folder.
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    assert hf.weights_file("sam2") is None  # not downloaded
    repo = tmp_path / "models--facebook--sam2.1-hiera-tiny"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("0123abcd")
    saved = repo / "snapshots" / "0123abcd" / "model.safetensors"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"abc")
    assert hf.weights_file("sam2") == saved
    assert hf.weights_file("sam2-small") is None  # another repository
    assert hf.weights_file("nonsense") is None
