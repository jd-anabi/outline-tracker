"""Segmenters: what turns a frame and a few clicks into one mask per object (SPEC 6.2, 12).

`base` holds the protocol and its two records, `hf` the Hugging Face backend (EdgeTAM, SAM 2.1),
`edgetam_convert` the one-time conversion of Meta's EdgeTAM checkpoint. torch and transformers are
imported only in here, and only when a model is actually loaded: importing this package or any of
its modules stays light.
"""
