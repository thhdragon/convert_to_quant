"""
W4A8 INT8 conversion functions for convert_to_quant.

Converts safetensors models to W4A8 INT8 quantized format (AsymW4A8Int8Layout).
Uses ConvRot Hadamard rotation, per-group FP8 relative scales, per-channel FP32 scales,
and optional Lloyd-Max codebooks.
"""

import os
from typing import Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from ..constants import NORMALIZE_SCALES_ENABLED, W4A8_CONVROT_GROUPSIZE, W4A8_GROUP_SIZE
from ..utils.comfy_quant import create_comfy_quant_tensor, fix_comfy_quant_params_structure
from ..utils.logging import error, info, minimal, verbose, warning
from ..utils.tensor_utils import normalize_tensorwise_scales
from .fp8_conversion import convert_to_fp8_scaled


def convert_to_w4a8_int8(
    input_file: str,
    output_file: str,
    comfy_quant: bool = True,
    calib_samples: int = 8192,
    seed: int = -1,
    filter_flags: Optional[Dict[str, bool]] = None,
    exclude_layers: Optional[str] = None,
    simple: bool = False,
    group_size: int = 16,
    convrot_group_size: int = 256,
    scale_dtype: str = "float32",  # gfx1150 falls back to widened bf16
    symmetric: bool = True,
    codebook: bool = True,
    low_memory: bool = False,
    **kwargs,
):
    """
    Convert a safetensors model to W4A8 INT8 format.

    Args:
        input_file: Path to input safetensors file
        output_file: Path to output safetensors file
        comfy_quant: Enable comfy_quant metadata tensors (default True)
        calib_samples: Number of random calibration samples for bias correction (default 3072)
        seed: Random seed for calibration data; -1 for a random seed
        filter_flags: Filter flags dict (model exclusions)
        exclude_layers: Regex pattern for layer exclusions
        simple: Skip iterative optimization
        group_size: Quantization group size (default 16)
        convrot_group_size: Hadamard rotation group size (default 256)
        scale_dtype: Scale dtype ("float8_e4m3fn" or "float32")
        symmetric: Use symmetric quantization
        codebook: Use Lloyd-Max codebook
        low_memory: Memory-efficient loading mode
    """
    import torch

    if seed == -1:
        seed = int(torch.randint(0, 2**32 - 1, ()).item())

    # Resolve scale_dtype string to actual torch.dtype
    _dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    resolved_scale_dtype = _dtype_map.get(scale_dtype, torch.float32)

    return convert_to_fp8_scaled(
        input_file=input_file,
        output_file=output_file,
        comfy_quant=comfy_quant,
        calib_samples=calib_samples,
        seed=seed,
        primary_format="w4a8_int8",
        filter_flags=filter_flags,
        exclude_layers=exclude_layers,
        no_learned_rounding=simple,
        block_size=group_size,
        convrot_group_size=convrot_group_size,
        scale_dtype=resolved_scale_dtype,
        symmetric=symmetric,
        codebook=codebook,
        low_memory=low_memory,
        **kwargs,
    )


def convert_w4a8_int8_to_comfy_quant(
    input_file: str,
    output_file: str,
    group_size: int = 16,
    convrot_group_size: int = 256,
    save_quant_metadata: bool = True,
):
    """
    Format an existing W4A8 INT8 model with .comfy_quant metadata.
    """
    info(f"Formatting W4A8 INT8 model to comfy_quant format: {input_file}")
    info("-" * 60)
    info(f"Group size: {group_size}, ConvRot group size: {convrot_group_size}")
    info("-" * 60)

    tensors: Dict[str, torch.Tensor] = {}
    original_metadata: Dict[str, str] = {}
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            original_metadata = f.metadata() or {}
            minimal(f"Loading {len(f.keys())} tensors from source file...")
            for key in tqdm(f.keys(), desc="Loading tensors"):
                tensors[key] = f.get_tensor(key)
    except Exception as e:
        error(f"FATAL: Error loading '{input_file}': {e}")
        return

    quant_metadata_layers = {} if save_quant_metadata else None
    output_tensors: Dict[str, torch.Tensor] = {}

    layer_info: Dict[str, Dict[str, torch.Tensor]] = {}
    other_tensors: Dict[str, torch.Tensor] = {}

    for key, tensor in tensors.items():
        if key.endswith(".weight"):
            base = key[: -len(".weight")]
            layer_info.setdefault(base, {})["weight"] = tensor
        elif key.endswith(".weight_scale") or key.endswith("._s_rel"):
            base = key.rsplit(".", 1)[0]
            layer_info.setdefault(base, {})["s_rel"] = tensor
        elif key.endswith("._s_channel"):
            base = key[: -len("._s_channel")]
            layer_info.setdefault(base, {})["s_channel"] = tensor
        elif key.endswith("._codebook"):
            base = key[: -len("._codebook")]
            layer_info.setdefault(base, {})["codebook"] = tensor
        elif key.endswith("._correction"):
            base = key[: -len("._correction")]
            layer_info.setdefault(base, {})["correction"] = tensor
        else:
            other_tensors[key] = tensor

    for base_name, layer_data in tqdm(layer_info.items(), desc="Processing layers"):
        weight = layer_data.get("weight")
        if weight is None:
            for k, v in layer_data.items():
                output_tensors[f"{base_name}.{k}"] = v
            continue

        for k, v in layer_data.items():
            output_tensors[f"{base_name}.{k}"] = v

        comfy_quant_tensor = create_comfy_quant_tensor(
            "asym_w4a8_int8",
            block_size=group_size,
            convrot=True,
            convrot_groupsize=convrot_group_size,
        )
        output_tensors[f"{base_name}.comfy_quant"] = comfy_quant_tensor

        if save_quant_metadata and quant_metadata_layers is not None:
            quant_metadata_layers[base_name] = {
                "format": "asym_w4a8_int8",
                "group_size": group_size,
                "convrot": True,
                "convrot_groupsize": convrot_group_size,
            }

    for key, tensor in other_tensors.items():
        if key.endswith(".comfy_quant"):
            fixed_tensor, _ = fix_comfy_quant_params_structure(tensor)
            output_tensors[key] = fixed_tensor
        else:
            output_tensors[key] = tensor

    info(f"\nSaving to {output_file}...")
    try:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        output_metadata = dict(original_metadata)
        if save_quant_metadata and quant_metadata_layers:
            import json

            full_metadata = {"format_version": "1.0", "layers": quant_metadata_layers}
            output_metadata["_quantization_metadata"] = json.dumps(full_metadata)

        save_kwargs = {"metadata": output_metadata} if output_metadata else {}
        output_tensors, _ = normalize_tensorwise_scales(output_tensors, NORMALIZE_SCALES_ENABLED)
        save_file(output_tensors, output_file, **save_kwargs)
        info("Conversion complete!")
    except Exception as e:
        error(f"FATAL: Error saving file '{output_file}': {e}")
        return
