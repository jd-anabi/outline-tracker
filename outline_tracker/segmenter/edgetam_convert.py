"""EdgeTAM: Meta's original checkpoint converted for Hugging Face transformers.

Ported from the course's shrimp._edgetam; the constants and functions it had are unchanged. No test
holds the two renaming tables themselves: a slow test compares the hash of the converted file with
the frozen one (tests/data/edgetam_weights.txt). Meta publishes EdgeTAM as a PyTorch file
(facebook/EdgeTAM on Hugging Face, also in github.com/facebookresearch/EdgeTAM). transformers can
run EdgeTAM but needs the weights renamed; the renaming below is taken from transformers' own
conversion script (src/transformers/models/edgetam_video/convert_edgetam_video_to_hf.py, Apache
License 2.0, Copyright 2025 The HuggingFace Inc. team). The converted model is saved once in
~/.cache/shrimp-models/edgetam (or $SHRIMP_MODEL_CACHE/edgetam) and reused: the same folder as last
week, so a laptop that ran last week's script does not download or convert again.

No quantities here: nothing in this module has units or a coordinate frame. torch and transformers
are imported inside the functions that need them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "dwconv": "depthwise_conv",
    "pwconv": "pointwise_conv",
    "fuser": "memory_fuser",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    "obj_ptr_tpos_proj": "temporal_positional_encoding_projection_layer",
    "no_obj_embed_spatial": "occlusion_spatial_embedding_parameter",
    "sam_prompt_encoder": "prompt_encoder",
    "sam_mask_decoder": "mask_decoder",
    "maskmem_tpos_enc": "memory_temporal_positional_encoding",
    "gamma": "scale",
    "image_encoder.neck": "vision_encoder.neck",
    "image_encoder": "vision_encoder.backbone",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "pix_feat_proj": "feature_projection",
    "patch_embed.proj": "patch_embed.projection",
    "no_mem_embed": "no_memory_embedding",
    "no_mem_pos_enc": "no_memory_positional_encoding",
    "obj_ptr": "object_pointer",
    ".norm": ".layer_norm",
    "trunk.": "",
    "out_proj": "o_proj",
    "body.": "timm_model.",
    "ff.0": "mlp.layer_norm",
    "ff.1": "mlp.up_proj",
    "ff.3": "mlp.down_proj",
}

PERCEIVER = {
    r"spatial_perceiver.latents": r"spatial_perceiver.latents_1d",
    r"spatial_perceiver.latents_1d_2d": r"spatial_perceiver.latents_2d",
    r"spatial_perceiver.layers.(\d+).attn.layer_norm_x": r"spatial_perceiver.layers.\1.layer_norm_input",
    r"spatial_perceiver.layers.(\d+).attn.layer_norm_latents": r"spatial_perceiver.layers.\1.layer_norm_latents",
    r"spatial_perceiver.layers.(\d+).self_attn.layer_norm": r"spatial_perceiver.layers.\1.layer_norm_self",
    r"spatial_perceiver.layers.(\d+).attn.to_q": r"spatial_perceiver.layers.\1.cross_attention.q_proj",
    r"spatial_perceiver.layers.(\d+).attn.to_kv": r"spatial_perceiver.layers.\1.cross_attention.kv_proj_combined",
    r"spatial_perceiver.layers.(\d+).attn.to_out": r"spatial_perceiver.layers.\1.cross_attention.o_proj",
    r"spatial_perceiver.layers.(\d+).self_attn.to_q": r"spatial_perceiver.layers.\1.self_attention.q_proj",
    r"spatial_perceiver.layers.(\d+).self_attn.to_kv": r"spatial_perceiver.layers.\1.self_attention.kv_proj_combined",
    r"spatial_perceiver.layers.(\d+).self_attn.to_out": r"spatial_perceiver.layers.\1.self_attention.o_proj",
    r"spatial_perceiver.layers.(\d+).attn": r"spatial_perceiver.layers.\1.cross_attention",
    r"spatial_perceiver.layers.(\d+).self_attn": r"spatial_perceiver.layers.\1.self_attention",
}


def edgetam_cache() -> Path:
    """The folder that holds the converted EdgeTAM (`model.safetensors` and its settings files):
    the one `load_edgetam` uses, ~/.cache/shrimp-models/edgetam or $SHRIMP_MODEL_CACHE/edgetam."""
    return Path(os.environ.get("SHRIMP_MODEL_CACHE", Path.home() / ".cache" / "shrimp-models")) / "edgetam"


def _renumber(key, pattern, group, mapping):
    m = re.match(pattern, key)
    if m:
        nb = int(m.group(group))
        for old, new in mapping.get(nb, []):
            key = key.replace(old, new)
    return key


def convert_state_dict(state_dict):
    import torch

    out = {}
    for key, value in state_dict.items():
        for old, new in KEYS_TO_MODIFY_MAPPING.items():
            if old in key:
                key = key.replace(old, new)
        for pattern, repl in PERCEIVER.items():
            if re.match(pattern, key):
                key = re.sub(pattern, repl, key)
        key = _renumber(key, r"vision_encoder.backbone.blocks.(\d+).mlp.layers.(\d+).*", 2,
                        {0: [("layers.0", "proj_in")], 1: [("layers.1", "proj_out")]})
        if re.match(r"memory_attention.*", key):
            key = key.replace("linear1", "mlp.up_proj").replace("linear2", "mlp.down_proj")
        key = _renumber(key, r"mask_decoder.transformer.layers.(\d+).mlp.layers.(\d+).*", 2,
                        {0: [("mlp.layers.0", "mlp.proj_in")], 1: [("mlp.layers.1", "mlp.proj_out")]})
        three = {0: [("layers.0", "proj_in")], 1: [("layers.1", "layers.0")], 2: [("layers.2", "proj_out")]}
        key = _renumber(key, r"mask_decoder.pred_obj_score_head.layers.(\d+).*", 1, three)
        key = _renumber(key, r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*", 2, three)
        if re.match(r"vision_encoder.neck.convs.(\d+).conv", key):
            key = key.replace(".conv.", ".")
        if re.match(r"memory_encoder.o_proj.*", key):
            key = key.replace(".o_proj.", ".projection.")
        key = _renumber(key, r"object_pointer_proj.layers.(\d+).*", 1, three)
        m = re.match(r"memory_encoder.mask_downsampler.encoder.(\d+).*", key)
        if m:
            nb = int(m.group(1))
            if nb == 12:
                key = key.replace(f"encoder.{nb}", "final_conv")
            elif nb % 3 == 0:
                key = key.replace(f"encoder.{nb}", f"layers.{nb // 3}.conv")
            elif nb % 3 == 1:
                key = key.replace(f"encoder.{nb}", f"layers.{nb // 3}.layer_norm")
        if "kv_proj_combined" in key:
            k_weight, v_weight = torch.chunk(value, 2, dim=0)
            out[key.replace("kv_proj_combined", "k_proj")] = k_weight
            out[key.replace("kv_proj_combined", "v_proj")] = v_weight
            continue
        out[key] = value
    out["shared_image_embedding.positional_embedding"] = out["prompt_encoder.shared_embedding.positional_embedding"]
    out["prompt_encoder.point_embed.weight"] = torch.cat(
        [out.pop(f"prompt_encoder.point_embed.{i}.weight") for i in range(4)], dim=0)
    return out


def edgetam_config():
    from transformers import (EdgeTamVideoConfig, EdgeTamVideoMaskDecoderConfig, EdgeTamVideoPromptEncoderConfig,
                              EdgeTamVisionConfig, TimmWrapperConfig)

    backbone = TimmWrapperConfig(architecture="repvit_m1.dist_in1k",
                                 model_args={"in_chans": 3, "features_only": True, "out_indices": (0, 1, 2, 3)})
    return EdgeTamVideoConfig(vision_config=EdgeTamVisionConfig(backbone_config=backbone),
                              prompt_encoder_config=EdgeTamVideoPromptEncoderConfig(),
                              mask_decoder_config=EdgeTamVideoMaskDecoderConfig(),
                              enable_temporal_pos_encoding_for_object_pointers=False,
                              enable_occlusion_spatial_embedding=False)


def load_edgetam(checkpoint=None):
    """(model, processor). Converts Meta's checkpoint the first time; afterwards loads the saved copy."""
    import torch
    from transformers import EdgeTamVideoModel, Sam2ImageProcessor, Sam2VideoProcessor, Sam2VideoVideoProcessor

    cache = Path(os.environ.get("SHRIMP_MODEL_CACHE", Path.home() / ".cache" / "shrimp-models")) / "edgetam"
    if (cache / "model.safetensors").exists() and checkpoint is None:
        return EdgeTamVideoModel.from_pretrained(cache), Sam2VideoProcessor.from_pretrained(cache)
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download("facebook/EdgeTAM", "edgetam.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    model = EdgeTamVideoModel(edgetam_config())
    missing, unexpected = model.load_state_dict(convert_state_dict(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"EdgeTAM conversion failed: missing {missing[:5]}, unexpected {unexpected[:5]}")
    processor = Sam2VideoProcessor(image_processor=Sam2ImageProcessor(), video_processor=Sam2VideoVideoProcessor())
    try:
        cache.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(cache)
        processor.save_pretrained(cache)
    except Exception as err:  # saving is only a speed-up for the next run
        print(f"  (could not save the converted EdgeTAM in {cache}: {err})")
    return model, processor
