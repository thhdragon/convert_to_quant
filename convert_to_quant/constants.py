"""
Constants and configuration values for convert_to_quant.

Contains model-specific key name filters, dtype settings, and INT4 ConvRot constants.
"""

import torch

# --- Model-specific exclusion lists (layers to skip quantization) ---
AVOID_KEY_NAMES = [
    "norm",
    "bias",
    "embed_tokens",
    "lm_head",
    "shared",
    "patch_embedding",
    "audio_model.patch_embedding",
    "ref_conv",
    "control_adapter",
    "motion_encoder.enc.net_app",
    "face_encoder.conv",
    "pose_patch_embedding",
    "motion_encoder.enc.fc",
    "img_emb.proj",
    "k_norm",
    "q_norm",
    "motion_encoder.dec",
    "head.modulation",
    "casual_audio_encoder",
    "cond_encoder",
    "frame_packer",
    "norm_k",
    "norm_q",
    "tekken_model",
    "multi_modal_projector",
    "patch_conv",
    "ln_pre",
    "input_layernorm",
    "attention_norm",
    "post_attention_layernorm",
    "mm_input_projection_weight",
]
T5XXL_REMOVE_KEY_NAMES = ["decoder", "lm_head"]
VISUAL_AVOID_KEY_NAMES = [
    "mlp.down_proj",
    "mlp.up_proj",
    "mlp.gate_proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
    "patch_embed",
    "pos_embed",
    "merger",
    "visual",
]
QWEN_AVOID_KEY_NAMES = ["norm_added_k", "norm_added_q", "norm_k", "norm_q", "txt_norm"]
HUNYUAN_AVOID_KEY_NAMES = [
    "layernorm", "img_attn_k_norm", "img_attn_q_norm", "txt_attn_k_norm", "txt_attn_q_norm", "norm1", "norm2",
    "vision_in.proj.0", "vision_in.proj.4", "img_in.proj", "cond_type_embedding"
]
ZIMAGE_AVOID_KEY_NAMES = [
    "cap_embedder.0", "cap_pad_token", "attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2", "k_norm", "q_norm",
    "x_pad_token"
]
GEMMA4_AVOID_KEY_NAMES = [
    "audio",
    "audio_projector",
    "embed_tokens",
    "per_layer_input_gate",
    "per_layer_projection",
    "per_layer_model_projection",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "vision",
    "multi_modal_projector",
]

# --- Layer key names for specific models (layers to include as high-precision) ---
FLUX_STYLE_LAYER_KEYNAMES = ["stream_modulation", "guidance_in", "time_in", "final_layer", "img_in", "txt_in"]
FLUX1_LAYER_KEYNAMES = FLUX_STYLE_LAYER_KEYNAMES
FLUX2_LAYER_KEYNAMES = FLUX_STYLE_LAYER_KEYNAMES
FLUX_KLEIN_LAYER_KEYNAMES = ["stream_modulation", "time_in", "final_layer", "img_in", "txt_in"]
DISTILL_LAYER_KEYNAMES_LARGE = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]
DISTILL_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer"]
NERF_LAYER_KEYNAMES_LARGE = ["distilled_guidance_layer", "nerf_blocks", "nerf_image_embedder", "txt_in", "_attn.proj"]
NERF_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer", "nerf_blocks", "nerf_image_embedder"]
RADIANCE_LAYER_KEYNAMES = ["img_in_patch", "nerf_final_layer_conv", "__x0__"]
WAN_LAYER_KEYNAMES = [
    "text_embedding", "time_embedding", "audio_model.text_embedding", "casual_audio_encoder", "frame_packer",
    "trainable_cond_mask", "cond_encoder", "audio_model.time_embedding", "time_projection", "video_model.time_projection",
    "head.head", "face_encoder.out_proj", "face_adapter", "audio_injector"
]
QWEN_LAYER_KEYNAMES = ["time_text_embed", "img_in", "norm_out", "proj_out", "txt_in"]
ERNIE_IMAGE_LAYER_KEYNAMES = [
    "time_embedding", "adaLN_modulation", "final_linear", "final_norm", "x_embedder",
    "layers.0.self_attention", "layers.0.mlp.gate_proj", "layers.0.mlp.up_proj", "text_proj"
]
ZIMAGE_LAYER_KEYNAMES = [
    "x_embedder", "clip_text_pooled_proj", "final_layer", "cap_embedder.1", "adaLN_modulation", "t_embedder", "time_text_embed"
]
ZIMAGE_REFINER_LAYER_KEYNAMES = ["context_refiner", "noise_refiner"]
ANIMA_LAYER_KEYNAMES = [
    "net.blocks.0.", "net.blocks.1.adaln_modulation", "final_layer", "llm_adapter", "t_embedder", "x_embedder"
]
LENS_LAYER_KEYNAMES = ["time_text_embed", "img_in", "norm_out", "proj_out", "img_mod.1", "txt_mod.1", "txt_in"]
QWEN35_AVOID_KEY_NAMES = [
    ".layers.0.", ".layers.23.", ".layers.31.", ".layers.63.", "lm_head", "embed_tokens",
    "in_proj_a", "in_proj_b", "visual.", "merger", "mtp.fc"
]
LTXV2_LAYER_KEYNAMES = [
    "scale_shift_table", "text_embedding_projection", "audio_vae", "audio_embeddings_connector",
    "adaln_single", "audio_adaln_single", "audio_patchify_proj", "audio_proj_out",
    "audio_prompt_adaln_single", "av_ca_a2v_gate_adaln_single", "av_ca_audio_scale_shift_adaln_single",
    "av_ca_v2a_gate_adaln_single", "av_ca_video_scale_shift_adaln_single", "patchify_proj",
    "proj_out", "prompt_adaln_single", "transformer_blocks.0.", "transformer_blocks.1.",
    "transformer_blocks.46.", "transformer_blocks.47.", "video_embeddings_connector",
    "vae.decoder", "vae.encoder", "vocoder", "to_gate_logits"
]
KREA2_LAYER_KEYNAMES = ["firs", "las", "tml", "txtfusion", "last.modulatio", "tpro"]
BOOGU_LAYER_KEYNAMES = [
    "image_index_embedding", "ref_image_patch_embedder", "time_caption_embed", "x_embedder", "norm1.linear", "norm_out"
]
IDEOGRAM4_LAYER_KEYNAMES = ["embed_image_indicator", "t_embedding", "llm_cond_proj", "adaln_proj", "final_layer", "input_proj"]

# --- Model Filter Registry ---
MODEL_FILTERS = {
    "gemma4": {"help": "Gemma 4 text/multimodal model", "category": "text", "exclude": GEMMA4_AVOID_KEY_NAMES},
    "qwen35": {"help": "Qwen3.5 text/multimodal model", "category": "text", "exclude": QWEN35_AVOID_KEY_NAMES},
    "t5xxl": {"help": "T5-XXL text encoder", "category": "text", "exclude": AVOID_KEY_NAMES, "remove": T5XXL_REMOVE_KEY_NAMES},
    "mistral": {"help": "Mistral text encoder", "category": "text", "exclude": AVOID_KEY_NAMES},
    "visual": {"help": "Visual encoder", "category": "text", "exclude": VISUAL_AVOID_KEY_NAMES},
    "generic_text": {"help": "Generic text encoder", "category": "text"},
    "flux1": {"help": "Flux.1 model", "category": "diffusion", "highprec": FLUX1_LAYER_KEYNAMES},
    "anima": {"help": "Anima model", "category": "diffusion", "highprec": ANIMA_LAYER_KEYNAMES},
    "lens": {"help": "LENS model", "category": "diffusion", "highprec": LENS_LAYER_KEYNAMES},
    "flux2": {"help": "Flux.2 model", "category": "diffusion", "highprec": FLUX2_LAYER_KEYNAMES},
    "flux_klein": {"help": "FLUX.2 Klein model", "category": "diffusion", "highprec": FLUX_KLEIN_LAYER_KEYNAMES},
    "distillation_large": {"help": "Distilled (large) model", "category": "diffusion", "highprec": DISTILL_LAYER_KEYNAMES_LARGE},
    "distillation_small": {"help": "Distilled (small) model", "category": "diffusion", "highprec": DISTILL_LAYER_KEYNAMES_SMALL},
    "nerf_large": {"help": "NeRF (large) model", "category": "diffusion", "highprec": NERF_LAYER_KEYNAMES_LARGE},
    "nerf_small": {"help": "NeRF (small) model", "category": "diffusion", "highprec": NERF_LAYER_KEYNAMES_SMALL},
    "radiance": {"help": "Radiance model", "category": "diffusion", "highprec": RADIANCE_LAYER_KEYNAMES},
    "krea2": {"help": "Krea2 model", "category": "diffusion", "highprec": KREA2_LAYER_KEYNAMES},
    "ideogram4": {"help": "Ideogram4 model", "category": "diffusion", "highprec": IDEOGRAM4_LAYER_KEYNAMES},
    "wan": {"help": "WAN video model", "category": "video", "exclude": AVOID_KEY_NAMES, "highprec": WAN_LAYER_KEYNAMES},
    "hunyuan": {"help": "Hunyuan Video model", "category": "video", "exclude": HUNYUAN_AVOID_KEY_NAMES},
    "qwen": {"help": "Qwen Image model", "category": "image", "exclude": QWEN_AVOID_KEY_NAMES, "highprec": QWEN_LAYER_KEYNAMES},
    "ernie_image": {"help": "ERNIE Image model", "category": "image", "highprec": ERNIE_IMAGE_LAYER_KEYNAMES},
    "zimage": {"help": "Z-Image model", "category": "image", "exclude": ZIMAGE_AVOID_KEY_NAMES, "highprec": ZIMAGE_LAYER_KEYNAMES},
    "zimage_refiner": {"help": "Z-Image Refiner model", "category": "image", "exclude": ZIMAGE_AVOID_KEY_NAMES, "highprec": ZIMAGE_REFINER_LAYER_KEYNAMES},
    "boogu": {"help": "Boogu model", "category": "image", "highprec": BOOGU_LAYER_KEYNAMES},
    "ltxv2": {"help": "LTXv2 model", "category": "video", "highprec": LTXV2_LAYER_KEYNAMES},
    "ltx2": {"help": "LTX v2 model", "category": "video", "highprec": LTXV2_LAYER_KEYNAMES},
    "ltx2_3": {"help": "LTX v2.3 model", "category": "video", "highprec": LTXV2_LAYER_KEYNAMES},
}


def build_exclusion_patterns(active_filters: dict) -> tuple:
    skip = []
    remove = []
    for name, cfg in MODEL_FILTERS.items():
        if active_filters.get(name, False):
            skip.extend(cfg.get("exclude", []))
            skip.extend(cfg.get("highprec", []))
            remove.extend(cfg.get("remove", []))
    return skip, skip, remove


# --- Dtype settings ---
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

# INT4 / ConvRot W4A4 constants
INT4_MIN = -7
INT4_MAX = 7
INT4_GROUP_SIZE = 64
TARGET_INT4_DTYPE = torch.int8

# Valid quantization formats
VALID_QUANT_FORMATS = {
    "convrot_w4a4",
    "int4",
}

NORMALIZE_SCALES_ENABLED = True
