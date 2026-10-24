"""The real model's factory: `load_segmenter(model, device)` makes the Hugging Face segmenter (SPEC 6.2).

The from-tracker workflow calls it when a run needs the model, and the window gives it to its worker
(`gui/app.py`). It is the one function that makes the real model, so a test's stand-in is a different function, and
the worker tells the two apart by identity. torch and transformers are loaded only when it is called: `segmenter.hf`
is imported inside it, so importing this module stays light.

No quantities here: nothing in this module has units or a coordinate frame.
"""


def load_segmenter(model: str, device: str):
    """The real model behind the segmenter protocol: `segmenter.hf.HFSegmenter(model, device)`.

    model: "edgetam", "sam2" or "sam2-small". device: "auto" (the fastest that works), "cpu", "mps"
    or "cuda". The first call downloads the model. Images and prompts are in px of the image given
    (segmenter/base.py). This is the one place of the module that loads torch.
    """
    from outline_tracker.segmenter.hf import HFSegmenter  # here, never at the top: it loads torch

    return HFSegmenter(model, device)
