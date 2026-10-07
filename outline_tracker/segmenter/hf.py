"""SAM 2.1 and EdgeTAM through Hugging Face transformers, one frame at a time (SPEC 6.2).

Ported from `TransformersSegmenter` and `load_model` of the course's shrimp.segment, written for
transformers 5.18.0. Kept from last week (tests/test_port_fidelity.py compares the unchanged
parts with the reference copy):
- the streaming session, which counts its own frames 0, 1, 2, ... (not the video's frame numbers:
  the model looks for its memory at the previous indices);
- the pruning of what the model no longer uses, with `KEEP_FRAMES = 20`;
- the device order cuda > mps > cpu, and the fall back from mps to cpu in `start()`;
- the models `sam2`, `sam2-small` and `edgetam`, and the EdgeTAM cache folder.

Changed:
- prompts go to the session with one call per object. One call for all objects, as last week,
  makes the processor pad objects that have fewer points (with -10), and the padding changes how a
  one-click object is decoded. With one click per object both ways leave the same session;
- results are `MaskResult`s (segmenter/base.py): logits from `post_process_masks(binarize=False)`,
  one object at a time, mask = logits > 0 (equal to last week's `binarize=True`), both cropped to
  the bounding box +/- 8 px; `score` is the model's object-presence logit (> 0: present);
- an `OSError` while loading says that the first run needs internet once.

Added (SPEC 5, 6.2, 6.4):
- an object can have several points, positive (label 1) and negative (label 0). Points outside
  the image given to the model are dropped; an object left without a positive point is refused
  with a `PromptError` before the model is called (`points_in_image` of segmenter/base.py);
- `step()` falls back from mps to cpu too: a new session on the frame that failed, with one
  positive click per object where it was last found;
- `preview()`: the masks for one image, in a session that is thrown away;
- `MODEL_LOCK` is held during every call of a model, and `reserve_ui_thread()` leaves one
  processor thread to a user interface.

Coordinates: an image is an RGB uint8 array indexed [row, column]. Prompt points are px in the
pixel frame of that image, Tracker's convention (pixel centers at +0.5, SPEC 3.1), and go to the
model unchanged, as last week. A result's `offset` = (col0, row0) is in px of the same image.
Logits have no unit.

torch and transformers are imported only when a model is loaded or a segmenter is made.
"""

from __future__ import annotations

import functools
import hashlib
import threading
from pathlib import Path

import numpy as np

from outline_tracker.measure import mask_center
from outline_tracker.segmenter.base import (MaskResult, ObjectPrompt, PromptError, crop_to_bbox,
                                            points_in_image)
from outline_tracker.segmenter.edgetam_convert import edgetam_cache

MODELS = {
    "sam2": "facebook/sam2.1-hiera-tiny",
    "sam2-small": "facebook/sam2.1-hiera-small",
    "edgetam": "facebook/EdgeTAM",
}
KEEP_FRAMES = 20        # per-frame model results kept in memory (the model looks back at most 16 frames)
# What a failing Apple GPU raises: a missing operation is a NotImplementedError (a RuntimeError), a
# tensor on the wrong device a RuntimeError, a float64 tensor a TypeError.
MPS_ERRORS = (RuntimeError, TypeError)
# Held during every call of a model and while a model is moved to another device: a preview and a
# run never use a model at the same time, whichever segmenters and threads they come from.
MODEL_LOCK = threading.Lock()


# --------------------------------------------------------------------------- loading

def needs_internet_once(load):
    """Wrap a model loader so that an `OSError` while loading says what to do about it.

    Loading reads from the Hugging Face Hub the first time: the weights, and for EdgeTAM also one
    small settings file of its backbone, even when the converted weights are already on this
    computer. Without a connection that fails with an `OSError`; it is raised again as an `OSError`
    whose text says "the first run needs internet once" and ends with the original error. Other
    errors pass unchanged. No quantities, so no units.
    """

    @functools.wraps(load)
    def wrapper(model_key: str, edgetam_checkpoint=None):
        try:
            return load(model_key, edgetam_checkpoint)
        except OSError as err:
            raise OSError(
                f"Could not load the model {model_key!r}: the first run needs internet once. The model "
                "(for EdgeTAM also one small settings file) is downloaded from the Hugging Face Hub and "
                "then kept on this computer. Connect to the internet and run this again. "
                f"({type(err).__name__}: {err})") from err

    return wrapper


@needs_internet_once
def load_model(model_key: str, edgetam_checkpoint=None):
    """The model and its image processor. SAM 2.1 comes from Meta's Hugging Face page; EdgeTAM is
    Meta's original file (facebook/EdgeTAM), converted once and kept in ~/.cache/shrimp-models."""
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    if model_key.startswith("sam2"):
        name = MODELS[model_key]
        return Sam2VideoModel.from_pretrained(name), Sam2VideoProcessor.from_pretrained(name)
    if model_key == "edgetam":
        from outline_tracker.segmenter.edgetam_convert import load_edgetam

        return load_edgetam(edgetam_checkpoint)
    raise ValueError(f"Unknown model {model_key!r}: choose sam2, sam2-small or edgetam.")


def weights_file(model_key: str) -> Path | None:
    """The `model.safetensors` file that `load_model(model_key)` reads on this computer.

    For EdgeTAM it is the converted copy in the EdgeTAM cache folder; for SAM 2.1 the file in the
    Hugging Face cache. None if the file is not there: the model was never loaded here, the
    converted EdgeTAM could not be saved, or the key is unknown. No quantities, so no units.
    """
    if model_key == "edgetam":
        path = edgetam_cache() / "model.safetensors"
        return path if path.exists() else None
    if model_key in MODELS:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(MODELS[model_key], "model.safetensors")
        return Path(found) if isinstance(found, str) else None
    return None


def file_sha256(path) -> str:
    """SHA-256 of a file's bytes, as 64 hexadecimal digits. No quantities, so no units."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# --------------------------------------------------------------------------- prompts

def prompt_lists(prompt: ObjectPrompt) -> tuple[list, list]:
    """One object's points and labels in the nesting the processor expects.

    Returns (input_points, input_labels) = ([[[[x, y], ...]]], [[[label, ...]]]): the levels are
    [image][object][point], with one image and one object. x and y are px in the pixel frame of
    the image given to the model (SPEC 3.1), unchanged; a label is 1 (positive) or 0 (negative).
    The values are Python floats and ints: the processor refuses numpy's float32 and int64 scalars.
    """
    points = [[float(x), float(y)] for x, y in prompt.points_px]
    labels = [int(label) for label in prompt.labels]
    return [[points]], [[labels]]


def add_prompts(processor, session, prompts: list[ObjectPrompt], original_size) -> list[int]:
    """Put the prompts of a run's first frame into a new session, with one call per object.

    Returns the integers 1..N by which the session knows the objects, in the order of `prompts`
    (as last week). Points are px in the pixel frame of the image; `original_size` is the image's
    (height, width) in px, as the processor returned it. Objects are never put into one call: the
    processor would pad those with fewer points. Each call replaces the session's list of objects
    with new inputs, so the list is set to all objects at the end, in a list of its own: the model
    removes entries from it while it runs.
    """
    ids = list(range(1, len(prompts) + 1))
    for obj, prompt in zip(ids, prompts):
        points, labels = prompt_lists(prompt)
        processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=[obj], input_points=points, input_labels=labels,
            original_size=original_size)
    session.obj_with_new_inputs = list(ids)
    return ids


# --------------------------------------------------------------------------- threads

_ui_thread_reserved = False


def reserve_ui_thread() -> int:
    """Leave one processor thread to the user interface (SPEC 6.4).

    torch gets one thread fewer than it has, but at least one. Only the first call in a process
    changes anything. Only the GUI worker calls this, just before it loads the model: the command
    line keeps torch's own number, which the comparison with last week's script needs (SPEC 13.3).
    Returns the number of threads torch uses afterwards. No quantities, so no units.
    """
    global _ui_thread_reserved
    import torch

    if not _ui_thread_reserved:
        torch.set_num_threads(max(1, torch.get_num_threads() - 1))
        _ui_thread_reserved = True
    return torch.get_num_threads()


# --------------------------------------------------------------------------- the segmenter

def _lost(obj_id: str) -> MaskResult:
    """The result for an object that is not in the session: an empty mask and no score."""
    return MaskResult(obj_id, (0, 0), np.zeros((0, 0), bool), np.zeros((0, 0), np.float32), None)


def _first_line(err: BaseException, limit: int = 200) -> str:
    """The first line of an error's text, cut to `limit` characters."""
    lines = str(err).splitlines()
    return lines[0][:limit] if lines else ""


class HFSegmenter:
    """SAM 2 / EdgeTAM through Hugging Face transformers, one frame at a time ("streaming").

    `model_key` is `edgetam`, `sam2` or `sam2-small`; `device` is `auto` (cuda, else mps, else
    cpu), `cpu`, `mps` or `cuda`; `edgetam_checkpoint` is a local edgetam.pt (testing). Give
    `model` and `processor`, as returned by `load_model(model_key)`, to share one loaded model
    between segmenters. Attributes: `device` (the one in use; it changes to `cpu` after a fall
    back, and stays there), `model_id` (the model's name on the Hugging Face Hub),
    `weights_sha256` (SHA-256 of the `model.safetensors` file that `load_model(model_key)` reads,
    or None if that file is not on this computer), `log` (a function that takes one line of text,
    `print` unless replaced: `step()` and `preview()` report a fall back through it).

    A run is `start`, any number of `step`s, `close`. `preview` can be called at any time and
    leaves a run as it is. Images and coordinates: see the module text.
    """

    session = None  # the session of the run; None between runs
    log = staticmethod(print)

    def __init__(self, model_key: str = "edgetam", device: str = "auto", edgetam_checkpoint=None, model=None,
                 processor=None):
        import torch

        self.torch = torch
        self.model_key = model_key
        if model is None:
            model, processor = load_model(model_key, edgetam_checkpoint)
        self.model, self.processor = model.eval(), processor
        self.device = self._pick_device(device)
        with MODEL_LOCK:
            self.model.to(self.device)
        self.model_id = MODELS.get(model_key, model_key)
        weights = weights_file(model_key)
        self.weights_sha256 = None if weights is None else file_sha256(weights)

    def _pick_device(self, device):
        torch = self.torch
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _results(self, out, inputs, ids, names) -> list[MaskResult]:
        sizes = [[int(v) for v in size] for size in inputs.original_sizes]
        low = out.pred_masks.float().cpu()  # (objects, 1, 256, 256): logits on the model's grid
        scores = out.object_score_logits.float().cpu().reshape(-1)
        row = {obj: k for k, obj in enumerate(out.object_ids)}
        results = []
        for obj, name in zip(ids, names):
            k = row[obj]
            # one object at a time: only one full-frame float array exists at any moment
            logits = self.processor.post_process_masks([low[k:k + 1]], original_sizes=sizes,
                                                       binarize=False)[0][0, 0].numpy()
            results.append(crop_to_bbox(logits > 0, logits, pad=8, obj_id=name, score=float(scores[k])))
        return results

    def _infer(self, rgb, prompts=None, keep=True) -> list[MaskResult]:
        """One image through the model: the only method that calls it (with `MODEL_LOCK` held).

        With `prompts`, a new session is opened and the image is its frame 0; with `keep` it
        becomes the session of the run, without `keep` it is thrown away after this image (a
        preview: the run's session, frame count and pruning are not touched). Without `prompts`
        the image is the next frame of the run. Returns one result per object of the session, in
        the order of its prompts.
        """
        torch = self.torch
        inputs = self.processor(images=rgb, device=self.device, return_tensors="pt")
        if prompts is not None:
            session = self.processor.init_video_session(inference_device=self.device, dtype=torch.float32)
            index, names = 0, [prompt.obj_id for prompt in prompts]
            ids = add_prompts(self.processor, session, prompts, inputs.original_sizes[0])
            if keep:
                self.session, self.index, self.ids, self.session_names = session, index, ids, names
        else:
            self.index += 1
            session, index, ids, names = self.session, self.index, self.ids, self.session_names
        with torch.inference_mode():
            # the frame number is given explicitly: the session would otherwise count its stored frames
            out = self.model(inference_session=session, frame_idx=index,
                             frame=inputs.pixel_values[0].to(self.device))
        if keep:
            self._prune()
        return self._results(out, inputs, ids, names)

    def _forward(self, rgb, prompts=None, slots=None):
        """One frame of the run through the model, and the notes a fall back needs.

        With `prompts` alone a run begins: points outside the image are dropped (`PromptError`
        before the model is called). With `prompts` and `slots`, a new session continues the run
        after a fall back: `slots[k]` is the place, in the order of the run's prompts, of the
        object of `prompts[k]`. Returns one result per object of the run, in the order of its
        prompts; an object that is not in the session is lost. Remembers where each object was
        last found: (u, v) in px in the pixel frame of the image (SPEC 3.1).
        """
        if prompts is not None and slots is None:
            prompts = [points_in_image(prompt, rgb.shape[0], rgb.shape[1]) for prompt in prompts]
            self.names = [prompt.obj_id for prompt in prompts]
            self.last_center = [None] * len(prompts)
            slots = list(range(len(prompts)))
        if slots is not None:
            self.slots = slots
        results = [_lost(name) for name in self.names]
        if self.slots:  # else no object is left to follow: the model is not called
            with MODEL_LOCK:
                found = self._infer(rgb, prompts)
            for slot, result in zip(self.slots, found):
                results[slot] = result
                u, v, area = mask_center(result.mask)
                if area:
                    self.last_center[slot] = (u + result.offset[0], v + result.offset[1])
        return results

    def _to_cpu(self) -> None:
        """Give up the Apple GPU: this segmenter uses the processor (cpu) from now on."""
        self.device = "cpu"
        with MODEL_LOCK:
            self.model.to("cpu")

    def start(self, image: np.ndarray, prompts: list[ObjectPrompt]) -> list[MaskResult]:
        """First frame of a run: the prompts start the tracking in a new session.

        `image` is an RGB uint8 array [row, column]; prompt points are px in its pixel frame
        (SPEC 3.1), with labels 1 (positive) and 0 (negative). Points outside the image are
        dropped; an object without a positive point inside it raises `PromptError` before the
        model is called. Returns one result per prompt, in the same order, with offsets in px of
        `image`.
        """
        try:
            return self._forward(image, prompts)
        except PromptError:
            raise
        except Exception as err:  # an Apple GPU (mps) can lack an operation: fall back to the processor
            if self.device == "mps":
                print(f"  The Apple GPU failed ({type(err).__name__}); using the processor instead.")
                self.device = "cpu"
                with MODEL_LOCK:
                    self.model.to("cpu")
                return self._forward(image, prompts)
            raise

    def step(self, image: np.ndarray) -> list[MaskResult]:
        """The next frame of the run: an RGB uint8 array [row, column] of the same size as the
        first. Returns one result per object, in the order of the prompts, offsets in px of `image`.

        If the Apple GPU fails here (a `RuntimeError` or a `TypeError` while the device is mps),
        the run continues on cpu: a new session starts on this frame, with one positive click per
        object at the center where it was last found (px in the pixel frame of the image). The
        model's memory of the earlier frames is lost. An object that was never found gets no click
        and stays lost for the rest of the run. One line goes to `log`.
        """
        try:
            return self._forward(image)
        except MPS_ERRORS as err:
            if self.device != "mps":
                raise
            slots = [slot for slot, center in enumerate(self.last_center) if center is not None]
            seeds = [ObjectPrompt(self.names[slot], [self.last_center[slot]], [1]) for slot in slots]
            self._to_cpu()
            never = [name for slot, name in enumerate(self.names) if slot not in slots]
            self.log(f"  The Apple GPU failed ({type(err).__name__}: {_first_line(err)}); using the processor "
                     "instead. Tracking starts again on this frame, from the last position of "
                     f"{', '.join(seed.obj_id for seed in seeds) or 'no object'}"
                     + (f"; never found, so still lost: {', '.join(never)}." if never else "."))
            return self._forward(image, seeds, slots)

    def preview(self, image: np.ndarray, prompts: list[ObjectPrompt]) -> list[MaskResult]:
        """The masks for one image, as the first frame of a run would give them; nothing is kept.

        `image` is an RGB uint8 array [row, column]; prompt points are px in its pixel frame
        (SPEC 3.1) and are checked as in `start`. The image goes through a session of its own on
        the same loaded model, thrown away afterwards, so a run in progress is not changed.
        Returns one result per prompt, in the same order, with offsets in px of `image`. Between
        runs an Apple GPU that fails is given up as in `step` (one line to `log`); during a run
        the error is raised instead, because the run's session lives on that device.
        """
        prompts = [points_in_image(prompt, image.shape[0], image.shape[1]) for prompt in prompts]
        if not prompts:
            return []
        try:
            with MODEL_LOCK:
                return self._infer(image, prompts, keep=False)
        except MPS_ERRORS as err:
            if self.device != "mps" or self.session is not None:
                raise
            self._to_cpu()
            self.log(f"  The Apple GPU failed ({type(err).__name__}: {_first_line(err)}); using the processor "
                     "instead.")
            with MODEL_LOCK:
                return self._infer(image, prompts, keep=False)

    def close(self) -> None:
        """End the run: forget the session (the loaded model stays). No quantities, so no units."""
        self.session = None

    def _prune(self):
        """Forget what the model no longer uses: the image of every earlier frame (12.6 MB each at the
        model's 1024 x 1024 input) and the per-frame results older than KEEP_FRAMES frames. Without this
        the memory grows by about 15 MB per frame. Entries are emptied, not removed, so the session's
        frame count stays right."""
        s = self.session
        current = self.index
        old = current - KEEP_FRAMES
        if s.processed_frames:
            for k in s.processed_frames:
                if k < current:
                    s.processed_frames[k] = None
        for obj in s.output_dict_per_obj:
            store = s.output_dict_per_obj[obj]["non_cond_frame_outputs"]
            for k in [k for k in store if k < old]:
                del store[k]
        for obj in s.frames_tracked_per_obj:
            tracked = s.frames_tracked_per_obj[obj]
            for k in [k for k in tracked if k < old]:
                del tracked[k]
