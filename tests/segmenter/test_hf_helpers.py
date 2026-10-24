"""`segmenter/hf.py` without torch: what is kept from last week, loading a model, and the thread (SPEC 6.2, 6.4).

Nothing here loads torch: the model is replaced by small stand-ins, and the methods kept from last
week's `TransformersSegmenter` are called on plain objects. Tests with the real processor or the
real model are in tests/slow/test_regression_reference.py and tests/slow/test_real_model.py.

The rest of the tests of the segmenter without a model are beside this file:
- tests/segmenter/test_hf_helpers_base.py: the protocol and its records, `crop_to_bbox`, and the
  points inside the image (`segmenter/base.py`);
- tests/segmenter/test_hf_helpers_prompts.py: the prompts that go to the processor (`segmenter/hf_prompts.py`);
- tests/segmenter/test_hf_helpers_fallback.py: the fall back from the Apple GPU in `step()` and `preview()`,
  and the model lock.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from layout_helpers import REPO

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
        "import outline_tracker.segmenter.load as load\n"
        "assert callable(hf.HFSegmenter) and callable(hf.load_model) and callable(load.load_segmenter)\n"
        "print(json.dumps(sorted(m for m in json.loads(sys.argv[1]) if m in sys.modules)))\n"
    )
    done = subprocess.run([sys.executable, "-c", code, json.dumps(HEAVY_MODULES)], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == []


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
    # `sam2` is the tiny one (docs/design/SPEC-v0.1.md, decision D1: "SAM 2.1 tiny stays available as an option") and
    # `sam2-small` the small one.
    assert hf.MODELS == {
        "edgetam": "facebook/EdgeTAM",
        "sam2": "facebook/sam2.1-hiera-tiny",
        "sam2-small": "facebook/sam2.1-hiera-small",
    }


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
