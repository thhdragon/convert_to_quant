"""
INT4 conversion utilities for convert_to_quant.

Converts INT4 quantized models to comfy_quant format.
"""

import os
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from ..constants import NORMALIZE_SCALES_ENABLED
from ..utils.comfy_quant import create_comfy_quant_tensor, fix_comfy_quant_params_structure
from ..utils.logging import error, info, minimal, verbose, warning
from ..utils.tensor_utils import normalize_tensorwise_scales, tensor_to_dict


def convert_int4_to_comfy_quant(input_file: str, output_file: str, block_size: int = 64, convrot_group_size: int = 256, save_quant_metadata: bool = True):
    """
    Convert INT4 quantized models to comfy_quant format.

    Args:
        input_file: Path to input INT4 safetensors file
        output_file: Path to output comfy_quant safetensors file
        block_size: Quantization group size (default 64)
        convrot_group_size: Hadamard rotation group size (default 256)
        save_quant_metadata: If True, save _quantization_metadata header
    """
    info(f"Converting INT4 model to comfy_quant format: {input_file}")
    info("-" * 60)
    info(f"Block size: {block_size}, ConvRot group size: {convrot_group_size}")
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
        elif key.endswith(".weight_scale") or key.endswith(".scale_weight"):
            base = key[: -len(".weight_scale")] if key.endswith(".weight_scale") else key[: -len(".scale_weight")]
            layer_info.setdefault(base, {})["weight_scale"] = tensor
        else:
            other_tensors[key] = tensor

    for base_name, layer_data in tqdm(layer_info.items(), desc="Processing layers"):
        weight = layer_data.get("weight")
        weight_scale = layer_data.get("weight_scale")

        if weight is None:
            if weight_scale is not None:
                output_tensors[f"{base_name}.weight_scale"] = weight_scale
            continue

        output_tensors[f"{base_name}.weight"] = weight
        if weight_scale is not None:
            output_tensors[f"{base_name}.weight_scale"] = weight_scale

        comfy_quant_tensor = create_comfy_quant_tensor("convrot_w4a4", block_size=block_size, convrot=True, convrot_groupsize=convrot_group_size)
        output_tensors[f"{base_name}.comfy_quant"] = comfy_quant_tensor

        if save_quant_metadata and quant_metadata_layers is not None:
            quant_metadata_layers[base_name] = {"format": "convrot_w4a4", "group_size": block_size, "convrot": True, "convrot_groupsize": convrot_group_size}

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
        output_tensors, normalized_count = normalize_tensorwise_scales(output_tensors, NORMALIZE_SCALES_ENABLED)
        save_file(output_tensors, output_file, **save_kwargs)
        info("Conversion complete!")
    except Exception as e:
        error(f"FATAL: Error saving file '{output_file}': {e}")
        return
