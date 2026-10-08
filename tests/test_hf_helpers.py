"""The segmenter protocol and the torch-free parts of the Hugging Face backend (SPEC 6.2, 12).

Nothing here loads torch: the model is replaced by small stand-ins, and the methods kept from last
week's `TransformersSegmenter` are called on plain objects. Tests with the real processor or the
real model are in tests/slow/test_regression_reference.py and tests/slow/test_real_model.py.

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


def test_each_model_key_names_its_repository_on_the_hugging_face_hub():
    from outline_tracker.segmenter import hf

    # Typed from NOTICE, which names the models that the tool downloads: "EdgeTAM (facebook/EdgeTAM) and
    # SAM 2.1 (facebook/sam2.1-hiera-tiny, facebook/sam2.1-hiera-small)". Of the two SAM 2.1 models
    # `sam2` is the tiny one (SPEC.md, decision D1: "SAM 2.1 tiny stays available as an option") and
    # `sam2-small` the small one.
    assert hf.MODELS == {
        "edgetam": "facebook/EdgeTAM",
        "sam2": "facebook/sam2.1-hiera-tiny",
        "sam2-small": "facebook/sam2.1-hiera-small",
    }


# ---------------------------------------------------------------------------------------------
# Several points per object, negative points (hf.py); points outside the image (base.py)


def _objects_with_1_3_and_2_points():
    """Three objects in a 1920 x 1080 image; labels: 1 positive, 0 negative."""
    from outline_tracker.segmenter.base import ObjectPrompt

    return [
        ObjectPrompt("A", [(700.5, 500.5)], [1]),
        ObjectPrompt("B", [(1000.5, 600.5), (1030.5, 640.5), (1010.0, 590.25)], [1, 0, 1]),
        ObjectPrompt("C", [(1400.5, 350.5), (1380.5, 360.5)], [1, 0]),
    ]


def test_prompt_lists_for_objects_with_1_3_and_2_points():
    from outline_tracker.segmenter import hf

    lists = [hf.prompt_lists(prompt) for prompt in _objects_with_1_3_and_2_points()]
    assert lists == [
        ([[[[700.5, 500.5]]]], [[[1]]]),
        ([[[[1000.5, 600.5], [1030.5, 640.5], [1010.0, 590.25]]]], [[[1, 0, 1]]]),
        ([[[[1400.5, 350.5], [1380.5, 360.5]]]], [[[1, 0]]]),
    ]
    for points, labels in lists:  # [image][object]: one image and one object in every pair of lists
        assert len(points) == len(labels) == 1 and len(points[0]) == len(labels[0]) == 1
        assert all(type(value) is float for point in points[0][0] for value in point)
        assert all(type(label) is int for label in labels[0][0])


def test_objects_with_different_point_counts_are_never_put_into_one_call():
    from outline_tracker.segmenter import hf

    processor, session = RecordingProcessor(), SimpleNamespace(obj_with_new_inputs=[])
    ids = hf.add_prompts(processor, session, _objects_with_1_3_and_2_points(), (1080, 1920))

    assert ids == [1, 2, 3]
    assert [call["obj_ids"] for call in processor.calls] == [[1], [2], [3]]
    # One object per call, so the processor has nothing to pad (it pads unequal counts with -10).
    assert [len(call["input_points"][0]) for call in processor.calls] == [1, 1, 1]
    assert [len(call["input_points"][0][0]) for call in processor.calls] == [1, 3, 2]
    assert [call["input_labels"] for call in processor.calls] == [[[[1]]], [[[1, 0, 1]]], [[[1, 0]]]]
    assert session.obj_with_new_inputs == [1, 2, 3]


def test_points_outside_the_image_are_dropped():
    from outline_tracker.segmenter import base
    from outline_tracker.segmenter.base import ObjectPrompt

    # A 1920 x 1080 image covers 0 <= u < 1920 and 0 <= v < 1080: pixel c is the square [c, c + 1) (SPEC 3.1).
    points = [(0.0, 0.0), (-0.01, 500.0), (1920.0, 500.0), (1919.99, 1079.99), (700.0, -0.01), (700.0, 1080.0),
              (700.5, 500.5), (-10.0, -10.0), (float("nan"), 500.0), (700.0, float("inf"))]
    labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    prompt = ObjectPrompt("A", list(points), list(labels), box_px=(1.0, 2.0, 3.0, 4.0))

    kept = base.points_in_image(prompt, height=1080, width=1920)

    assert kept.points_px == [(0.0, 0.0), (1919.99, 1079.99), (700.5, 500.5)]
    assert kept.labels == [1, 0, 1]  # every label stays with its point
    assert (kept.obj_id, kept.box_px) == ("A", (1.0, 2.0, 3.0, 4.0))
    assert (prompt.points_px, prompt.labels) == (points, labels)  # the caller's prompt is not changed

    # "The image" is the one given to the model: in a 96 px crop the same points are judged anew.
    small = ObjectPrompt("A", [(95.5, 0.5), (96.0, 0.5), (48.0, 95.99), (48.0, 96.0)], [1, 0, 0, 1])
    assert base.points_in_image(small, height=96, width=96).points_px == [(95.5, 0.5), (48.0, 95.99)]


def test_an_object_left_without_a_positive_point_is_refused():
    from outline_tracker.segmenter import base
    from outline_tracker.segmenter.base import ObjectPrompt

    with pytest.raises(base.PromptError) as caught:  # the positive point is outside, the negative one inside
        base.points_in_image(ObjectPrompt("B2", [(2000.5, 500.5), (700.5, 500.5)], [1, 0]), height=1080, width=1920)
    message = str(caught.value)
    assert "'B2'" in message and "no positive point inside the image" in message
    assert "1920 x 1080 px" in message
    assert isinstance(caught.value, ValueError)
    with pytest.raises(base.PromptError, match="no positive point inside the image"):
        base.points_in_image(ObjectPrompt("A", [], []), height=1080, width=1920)
    with pytest.raises(base.PromptError, match="no positive point inside the image"):
        base.points_in_image(ObjectPrompt("A", [(700.5, 500.5)], [0]), height=1080, width=1920)


def test_points_and_labels_that_do_not_fit_are_refused():
    from outline_tracker.segmenter import base
    from outline_tracker.segmenter.base import ObjectPrompt

    with pytest.raises(base.PromptError, match="'A' has 2 points and 1 labels"):
        base.points_in_image(ObjectPrompt("A", [(700.5, 500.5), (710.5, 500.5)], [1]), height=1080, width=1920)
    # To the model, 2 and 3 are the corners of a box and negative labels are padding.
    for label in (2, 3, -1, -10):
        with pytest.raises(base.PromptError, match="labels must be 1 .positive. or 0 .negative."):
            base.points_in_image(ObjectPrompt("A", [(700.5, 500.5), (710.5, 500.5)], [1, label]), height=1080,
                               width=1920)


# ---------------------------------------------------------------------------------------------
# hf.py: bookkeeping around the model call (fall back in step, preview, the lock), with a stand-in
# for `HFSegmenter._infer`, the one method that touches torch

FRAME_SHAPE = (120, 160, 3)  # rows, columns, channels of the stand-in frames


def _frame(t: int) -> np.ndarray:
    """Stand-in frame number t: an image whose every value is t."""
    return np.full(FRAME_SHAPE, t, np.uint8)


def _block(name: str, cols: tuple[int, int], rows: tuple[int, int]):
    """An object that fills columns cols[0]..cols[1] and rows rows[0]..rows[1], both inclusive.
    Its center is ((cols[0] + cols[1] + 1) / 2, (rows[0] + rows[1] + 1) / 2) px (SPEC 3.1)."""
    from outline_tracker.segmenter.base import crop_to_bbox

    mask = np.zeros(FRAME_SHAPE[:2], bool)
    mask[rows[0]:rows[1] + 1, cols[0]:cols[1] + 1] = True
    return crop_to_bbox(mask, np.where(mask, 3.0, -3.0).astype(np.float32), obj_id=name, score=4.0)


def _not_found(name: str):
    """What the model returns for an object it calls absent: all logits -1024, so an empty mask."""
    from outline_tracker.segmenter.base import crop_to_bbox

    mask = np.zeros(FRAME_SHAPE[:2], bool)
    return crop_to_bbox(mask, np.full(mask.shape, -1024.0, np.float32), obj_id=name, score=-5.0)


def _scene(name: str, t: int):
    """The stand-in scene on frame t. Centers in px:
    A moves right, (25 + 10 t, 35); B is found on frame 0 only, at (105, 52); C is never found;
    D moves down, (65, 85 + 5 t)."""
    if name == "A":
        return _block("A", (20 + 10 * t, 29 + 10 * t), (30, 39))
    if name == "B" and t == 0:
        return _block("B", (100, 109), (50, 53))
    if name == "D":
        return _block("D", (60, 69), (80 + 5 * t, 89 + 5 * t))
    return _not_found(name)


def _scene_prompts(names: str = "ABCD"):
    from outline_tracker.segmenter.base import ObjectPrompt

    clicks = {"A": (25.0, 35.0), "B": (105.0, 52.0), "C": (140.0, 100.0), "D": (65.0, 85.0)}
    return [ObjectPrompt(name, [clicks[name]], [1]) for name in names]


def _center(result):
    """(u, v) of a result in px of the frame, or None if the object was not found."""
    u, v, area = mask_center(result.mask)
    return None if area == 0 else (u + result.offset[0], v + result.offset[1])


class InferStub:
    """Stands in for `HFSegmenter._infer`: returns the scene's results for the objects of the
    session, like the model one result per object in the order of the prompts, and raises `error`
    on frame `fail_on_frame` while the segmenter's device is `fail_device`. Records every call."""

    def __init__(self, segmenter, fail_on_frame=None, fail_device="mps", error=None):
        self.segmenter, self.fail_on_frame, self.fail_device = segmenter, fail_on_frame, fail_device
        self.error = error or NotImplementedError(
            "The operator 'aten::x' is not currently implemented for the MPS device.")
        self.calls, self.objects = [], []

    def __call__(self, rgb, prompts=None, keep=True):
        from outline_tracker.segmenter import hf

        t = int(rgb[0, 0, 0])
        self.calls.append(SimpleNamespace(device=self.segmenter.device, frame=t, prompts=prompts, keep=keep,
                                          locked=hf.MODEL_LOCK.locked()))
        if t == self.fail_on_frame and self.segmenter.device == self.fail_device:
            raise self.error
        objects = self.objects if prompts is None else [prompt.obj_id for prompt in prompts]
        if keep:
            self.objects = objects
        return [_scene(name, t) for name in objects]


def _stub_segmenter(device: str, **failure):
    """An `HFSegmenter` whose constructor did not run (no torch): a device, a model that records
    where it is moved, a log that collects lines, and the stand-in for `_infer`."""
    from outline_tracker.segmenter import hf

    segmenter = object.__new__(hf.HFSegmenter)
    segmenter.device = device
    segmenter.moves, segmenter.lines = [], []
    segmenter.model = SimpleNamespace(to=lambda where: segmenter.moves.append((where, hf.MODEL_LOCK.locked())))
    segmenter.log = segmenter.lines.append
    segmenter._infer = InferStub(segmenter, **failure)
    return segmenter


def test_step_falls_back_from_the_apple_gpu_and_reseeds_every_object(capsys):
    segmenter = _stub_segmenter("mps", fail_on_frame=2)  # the model call fails on the third frame

    first = segmenter.start(_frame(0), _scene_prompts())
    second = segmenter.step(_frame(1))
    assert segmenter.device == "mps" and segmenter.moves == [] and segmenter.lines == []
    third = segmenter.step(_frame(2))
    fourth = segmenter.step(_frame(3))

    # It moved to cpu once, opened a new session on the frame that failed, and went on from there.
    assert segmenter.device == "cpu"
    assert segmenter.moves == [("cpu", True)]  # with the model lock held
    calls = segmenter._infer.calls
    assert [(call.device, call.frame, call.prompts is not None) for call in calls] == [
        ("mps", 0, True), ("mps", 1, False), ("mps", 2, False), ("cpu", 2, True), ("cpu", 3, False)]
    assert all(call.keep and call.locked for call in calls)

    # The new session has one positive click per object, at the center where it was last found.
    seeds = calls[3].prompts
    assert [(seed.obj_id, seed.labels, len(seed.points_px)) for seed in seeds] == [
        ("A", [1], 1), ("B", [1], 1), ("D", [1], 1)]  # C was never found: no click
    assert seeds[0].points_px[0] == pytest.approx((35.0, 35.0))  # A on frame 1
    assert seeds[1].points_px[0] == pytest.approx((105.0, 52.0))  # B was not found on frame 1: frame 0
    assert seeds[2].points_px[0] == pytest.approx((65.0, 90.0))  # D on frame 1
    assert all(type(value) is float for seed in seeds for value in seed.points_px[0])

    # Every frame still gives one result per object, in the order of the prompts.
    for results in (first, second, third, fourth):
        assert [result.obj_id for result in results] == ["A", "B", "C", "D"]
    assert [_center(result) for result in first] == [(25.0, 35.0), (105.0, 52.0), None, (65.0, 85.0)]
    assert [_center(result) for result in second] == [(35.0, 35.0), None, None, (65.0, 90.0)]
    assert [_center(result) for result in third] == [(45.0, 35.0), None, None, (65.0, 95.0)]
    assert [_center(result) for result in fourth] == [(55.0, 35.0), None, None, (65.0, 100.0)]
    lost = third[2]  # C is not in the new session: an empty result without a score
    assert lost.mask.shape == (0, 0) and lost.mask.dtype == bool
    assert lost.logits.shape == (0, 0) and lost.logits.dtype == np.float32
    assert lost.offset == (0, 0) and lost.score is None

    # It is logged, once, through the segmenter's `log`; nothing is printed besides.
    (line,) = segmenter.lines
    assert "The Apple GPU failed (NotImplementedError" in line and "aten::x" in line
    assert "processor" in line and "A, B, D" in line and "C" in line.split("A, B, D")[1]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("error", [
    NotImplementedError("The operator 'aten::x' is not currently implemented for the MPS device."),
    RuntimeError("Expected all tensors to be on the same device"),
    TypeError("Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64."),
], ids=lambda error: type(error).__name__)
def test_step_falls_back_on_a_runtime_error_or_a_type_error(error):
    segmenter = _stub_segmenter("mps", fail_on_frame=1, error=error)
    segmenter.start(_frame(0), _scene_prompts("AD"))
    results = segmenter.step(_frame(1))
    assert segmenter.device == "cpu" and segmenter.moves == [("cpu", True)]
    assert [_center(result) for result in results] == [(35.0, 35.0), (65.0, 90.0)]
    assert type(error).__name__ in segmenter.lines[0]


def test_step_does_not_hide_an_error_on_other_devices_or_of_another_kind():
    segmenter = _stub_segmenter("cpu", fail_on_frame=1, fail_device="cpu", error=RuntimeError("out of memory"))
    segmenter.start(_frame(0), _scene_prompts())
    with pytest.raises(RuntimeError, match="out of memory"):
        segmenter.step(_frame(1))
    assert segmenter.device == "cpu" and segmenter.moves == [] and segmenter.lines == []

    segmenter = _stub_segmenter("mps", fail_on_frame=1, error=KeyError("frame"))  # not what a failing GPU raises
    segmenter.start(_frame(0), _scene_prompts())
    with pytest.raises(KeyError):
        segmenter.step(_frame(1))
    assert segmenter.device == "mps" and segmenter.moves == [] and segmenter.lines == []


def test_step_after_a_fall_back_with_no_object_ever_found_calls_no_model():
    segmenter = _stub_segmenter("mps", fail_on_frame=1)
    segmenter.start(_frame(0), _scene_prompts("C"))
    for t in (1, 2):
        (result,) = segmenter.step(_frame(t))
        assert result.obj_id == "C" and result.mask.shape == (0, 0)
    assert segmenter.device == "cpu"
    # No click can be placed, so no new session is opened and the model is not called again.
    assert [(call.device, call.frame) for call in segmenter._infer.calls] == [("mps", 0), ("mps", 1)]
    assert len(segmenter.lines) == 1


def test_a_prompt_error_is_raised_before_the_model_is_called(capsys):
    from outline_tracker.segmenter import base
    from outline_tracker.segmenter.base import ObjectPrompt

    outside = ObjectPrompt("B", [(160.0, 52.0), (105.0, 52.0)], [1, 0])  # the frames are 160 px wide
    for call in ("start", "preview"):
        segmenter = _stub_segmenter("mps")
        with pytest.raises(base.PromptError, match="'B' has no positive point inside the image"):
            getattr(segmenter, call)(_frame(0), [*_scene_prompts("A"), outside])
        assert segmenter._infer.calls == []
        # A click in the wrong place is not a failure of the Apple GPU.
        assert segmenter.device == "mps" and segmenter.moves == [] and segmenter.lines == []
    assert capsys.readouterr().out == ""


def test_start_and_preview_give_the_model_only_the_points_inside_the_image():
    from outline_tracker.segmenter.base import ObjectPrompt

    prompts = [ObjectPrompt("A", [(25.0, 35.0), (25.0, 120.0), (-3.0, 35.0)], [1, 1, 0]),  # the frames have 120 rows
               ObjectPrompt("D", [(65.0, 85.0), (70.0, 119.5)], [1, 0])]
    for call in ("start", "preview"):
        segmenter = _stub_segmenter("cpu")
        results = getattr(segmenter, call)(_frame(0), prompts)
        (made,) = segmenter._infer.calls
        assert [(p.obj_id, p.points_px, p.labels) for p in made.prompts] == [
            ("A", [(25.0, 35.0)], [1]), ("D", [(65.0, 85.0), (70.0, 119.5)], [1, 0])]
        assert made.locked and made.keep is (call == "start")
        assert [_center(result) for result in results] == [(25.0, 35.0), (65.0, 85.0)]


def test_preview_keeps_no_tracking_state():
    from outline_tracker.segmenter.base import ObjectPrompt

    segmenter = _stub_segmenter("mps", fail_on_frame=2)
    segmenter.start(_frame(0), _scene_prompts("AD"))

    # A preview of another object, on another frame, in the middle of the run.
    (shown,) = segmenter.preview(_frame(0), [ObjectPrompt("B", [(105.0, 52.0)], [1])])
    assert shown.obj_id == "B" and _center(shown) == (105.0, 52.0)
    assert segmenter._infer.calls[-1].keep is False  # a session of its own, thrown away

    second = segmenter.step(_frame(1))
    third = segmenter.step(_frame(2))  # the fall back re-seeds the objects of the run, not the previewed one
    assert [seed.obj_id for seed in segmenter._infer.calls[-1].prompts] == ["A", "D"]
    assert [result.obj_id for result in second] == [result.obj_id for result in third] == ["A", "D"]
    assert [_center(result) for result in third] == [(45.0, 35.0), (65.0, 95.0)]


def test_preview_falls_back_from_the_apple_gpu_between_runs_only():
    segmenter = _stub_segmenter("mps", fail_on_frame=2)
    (shown,) = segmenter.preview(_frame(2), _scene_prompts("A"))  # no run is active
    assert _center(shown) == (45.0, 35.0)
    assert segmenter.device == "cpu" and segmenter.moves == [("cpu", True)]
    assert [(call.device, call.keep) for call in segmenter._infer.calls] == [("mps", False), ("cpu", False)]
    assert len(segmenter.lines) == 1 and "The Apple GPU failed (NotImplementedError" in segmenter.lines[0]

    # During a run the session lives on the Apple GPU: the preview fails and the run is left alone.
    segmenter = _stub_segmenter("mps", fail_on_frame=2)
    segmenter.session = object()  # what `_infer` leaves behind when a run starts
    segmenter.start(_frame(0), _scene_prompts("A"))
    with pytest.raises(NotImplementedError):
        segmenter.preview(_frame(2), _scene_prompts("A"))
    assert segmenter.device == "mps" and segmenter.moves == [] and segmenter.lines == []
    assert [_center(result) for result in segmenter.step(_frame(1))] == [(35.0, 35.0)]


def test_start_moves_the_model_with_the_lock_held():
    from outline_tracker.segmenter import hf

    held = []

    def forward(image, prompts=None):
        if segmenter.device == "mps":
            raise NotImplementedError("not implemented for the MPS device")
        return ["masks"]

    model = SimpleNamespace(to=lambda device: held.append((device, hf.MODEL_LOCK.locked())))
    segmenter = SimpleNamespace(device="mps", model=model, _forward=forward)
    assert hf.HFSegmenter.start(segmenter, "image", ["prompt"]) == ["masks"]
    assert held == [("cpu", True)]
    assert not hf.MODEL_LOCK.locked()


# ---------------------------------------------------------------------------------------------
# hf.py: one thread for the user interface (SPEC 6.4)


@pytest.mark.parametrize("threads, expected", [(8, 7), (2, 1), (1, 1)])
def test_reserve_ui_thread_takes_one_thread_from_torch_once(monkeypatch, threads, expected):
    from outline_tracker.segmenter import hf

    state = {"threads": threads, "set": []}

    def set_num_threads(count):
        state["set"].append(count)
        state["threads"] = count

    # A stand-in for the torch module: no fast test loads the real one.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_num_threads=lambda: state["threads"],
                                                               set_num_threads=set_num_threads))
    monkeypatch.setattr(hf, "_ui_thread_reserved", False)

    assert hf.reserve_ui_thread() == expected  # max(1, threads - 1)
    assert hf.reserve_ui_thread() == expected  # once per process: a second call changes nothing
    assert state["set"] == [expected]
