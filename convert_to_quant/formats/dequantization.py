"""
Model dequantization utilities for convert_to_quant.

Dequantizes quantized safetensors models (FP8, INT8, INT4/ConvRot, NVFP4, MXFP8, ComfyQuant)
back to unquantized high-precision tensors (default: bfloat16).
"""

import json
import os
from typing import Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from ..comfy.quant_ops import (
    BlockWiseFP8Layout,
    BlockWiseINT8Layout,
    RowWiseFP8Layout,
    TensorCoreConvRotW4A4Layout,
    TensorCoreFP8Layout,
    TensorWiseINT8Layout,
)
from ..converters.mxfp8_converter import dequantize_mxfp8
from ..converters.nvfp4_converter import dequantize_nvfp4
from ..utils.comfy_quant import tensor_to_dict
from ..utils.convrot import dequantize_convrot_w4a4_weight
from ..utils.logging import error, info, minimal, verbose, warning


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def dequantize_model(
    input_file: str,
    output_file: str,
    dtype: str = "bf16",
    low_memory: bool = False,
):
    """
    Dequantize a quantized safetensors model back to high precision.

    Args:
        input_file: Path to input quantized safetensors model
        output_file: Path to output dequantized safetensors file
        dtype: Target precision string ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32")
        low_memory: Stream loading/saving if set
    """
    target_dtype = DTYPE_MAP.get(dtype.lower())
    if target_dtype is None:
        error(f"ERROR: Unsupported target dtype '{dtype}'. Supported: {list(DTYPE_MAP.keys())}")
        return

    info(f"Dequantizing model to {dtype} ({target_dtype})")
    info(f"Input:  {input_file}")
    info(f"Output: {output_file}")
    info("-" * 60)

    # Load input tensors and preserve metadata (excluding quantization header entries)
    tensors: Dict[str, torch.Tensor] = {}
    original_metadata: Dict[str, str] = {}
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            raw_meta = f.metadata() or {}
            for k, v in raw_meta.items():
                if k != "_quantization_metadata":
                    original_metadata[k] = v

            minimal(f"Loading {len(f.keys())} tensors from source file...")
            for key in tqdm(f.keys(), desc="Loading tensors"):
                tensors[key] = f.get_tensor(key)
    except Exception as e:
        error(f"FATAL: Error loading '{input_file}': {e}")
        return

    # Identify layer groups and standalone tensors
    layer_info: Dict[str, Dict[str, torch.Tensor]] = {}
    other_tensors: Dict[str, torch.Tensor] = {}

    QUANT_SUFFIXES = (
        ".weight",
        ".comfy_quant",
        ".weight_scale",
        ".scale_weight",
        ".input_scale",
        ".scale_input",
        ".per_tensor_scale",
        "._s_rel",
        "._s_channel",
        "._correction",
        "._codebook",
    )

    for key, tensor in tensors.items():
        if key == "scaled_fp8":
            continue  # Skip marker tensor

        matched_suffix = None
        for suffix in QUANT_SUFFIXES:
            if key.endswith(suffix):
                matched_suffix = suffix
                break

        if matched_suffix:
            base = key[: -len(matched_suffix)]
            field = matched_suffix[1:]  # e.g., "weight", "comfy_quant", etc.
            if base not in layer_info:
                layer_info[base] = {}
            layer_info[base][field] = tensor
        else:
            other_tensors[key] = tensor

    output_tensors: Dict[str, torch.Tensor] = {}
    dequantized_count = 0
    passthrough_count = 0

    for base_name, data in tqdm(layer_info.items(), desc="Dequantizing layers"):
        weight = data.get("weight")
        comfy_quant = data.get("comfy_quant")
        weight_scale = (
            data.get("weight_scale")
            if data.get("weight_scale") is not None
            else (data.get("scale_weight") if data.get("scale_weight") is not None else data.get("_s_rel"))
        )
        per_tensor_scale = data.get("per_tensor_scale")

        if weight is None:
            # No weight tensor - copy through any other tensors in this layer
            for sub_k, sub_v in data.items():
                if sub_k not in (
                    "comfy_quant",
                    "weight_scale",
                    "scale_weight",
                    "input_scale",
                    "scale_input",
                    "per_tensor_scale",
                    "_s_rel",
                    "_s_channel",
                    "_correction",
                    "_codebook",
                ):
                    if sub_v.is_floating_point():
                        output_tensors[f"{base_name}.{sub_k}"] = sub_v.to(target_dtype)
                    else:
                        output_tensors[f"{base_name}.{sub_k}"] = sub_v
            continue

        dequantized_weight: Optional[torch.Tensor] = None

        # Check if comfy_quant is present
        if comfy_quant is not None:
            try:
                config = tensor_to_dict(comfy_quant)
                fmt = config.get("format", "")
                group_size = config.get("group_size") or config.get("block_size")
                convrot = config.get("convrot", False)
                convrot_groupsize = config.get("convrot_groupsize", 256)

                if fmt in ("float8_e4m3fn", "float8_e4m3fn_tensorwise", "fp8_e4m3fn"):
                    if weight_scale is not None:
                        dequantized_weight = TensorCoreFP8Layout.dequantize(
                            weight, weight_scale, orig_dtype=target_dtype
                        )
                elif fmt == "float8_e4m3fn_rowwise":
                    if weight_scale is not None:
                        dequantized_weight = RowWiseFP8Layout.dequantize(
                            weight, weight_scale, orig_dtype=target_dtype
                        )
                elif fmt in ("float8_e4m3fn_blockwise", "float8_e4m3fn_block"):
                    if weight_scale is not None:
                        bs = group_size or 128
                        dequantized_weight = BlockWiseFP8Layout.dequantize(
                            weight, weight_scale, block_size=bs, orig_dtype=target_dtype
                        )
                elif fmt in ("int8_tensorwise", "int8_rowwise", "int8"):
                    if weight_scale is not None:
                        dequantized_weight = TensorWiseINT8Layout.dequantize(
                            weight, weight_scale, is_weight=True, orig_dtype=target_dtype
                        )
                elif fmt in ("int8_blockwise", "int8_block"):
                    if weight_scale is not None:
                        bs = group_size or 128
                        dequantized_weight = BlockWiseINT8Layout.dequantize(
                            weight, weight_scale, block_size=bs, is_weight=True, orig_dtype=target_dtype
                        )
                elif fmt in ("convrot_w4a4", "int4_convrot"):
                    if weight_scale is not None:
                        bs = group_size or 64
                        c_gs = convrot_groupsize or 256
                        dequantized_weight = dequantize_convrot_w4a4_weight(
                            weight,
                            weight_scale,
                            convrot_groupsize=c_gs,
                            quant_group_size=bs,
                            output_dtype=target_dtype,
                        )
                elif fmt in ("w4a8_int8", "asym_w4a8_int8", "w4a8", "AsymW4A8Int8Layout"):
                    s_rel = weight_scale
                    s_channel = data.get("_s_channel")
                    codebook = data.get("_codebook")
                    correction = data.get("_correction")
                    if s_rel is not None and s_channel is not None:
                        bs = group_size or 16
                        c_gs = convrot_groupsize or 256
                        from ..converters.w4a8_int8_converter import dequantize_w4a8_int8_pytorch

                        dequantized_weight = dequantize_w4a8_int8_pytorch(
                            weight,
                            s_rel,
                            s_channel,
                            codebook=codebook,
                            correction=correction,
                            group_size=bs,
                            convrot_groupsize=c_gs,
                            output_dtype=target_dtype,
                        )
                elif fmt == "nvfp4":
                    if weight_scale is not None:
                        pts = per_tensor_scale if per_tensor_scale is not None else torch.tensor(1.0)
                        dequantized_weight = dequantize_nvfp4(
                            weight,
                            block_scales=weight_scale,
                            per_tensor_scale=pts,
                            output_dtype=target_dtype,
                        )
                elif fmt == "mxfp8":
                    if weight_scale is not None:
                        dequantized_weight = dequantize_mxfp8(
                            weight, block_scales=weight_scale, output_dtype=target_dtype
                        )
            except Exception as e:
                warning(
                    f"Failed to parse comfy_quant for {base_name}: {e}. Falling back to tensor inspection."
                )

        # Fallback if no comfy_quant or dequantization via comfy_quant failed
        if dequantized_weight is None:
            if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                if weight_scale is not None:
                    if weight_scale.numel() == 1:
                        dequantized_weight = TensorCoreFP8Layout.dequantize(
                            weight, weight_scale, orig_dtype=target_dtype
                        )
                    elif weight_scale.ndim == 1 and weight_scale.shape[0] == weight.shape[0]:
                        dequantized_weight = RowWiseFP8Layout.dequantize(
                            weight, weight_scale, orig_dtype=target_dtype
                        )
                    elif weight_scale.ndim >= 2:
                        M = weight.shape[0]
                        bs = max(1, M // weight_scale.shape[0])
                        dequantized_weight = BlockWiseFP8Layout.dequantize(
                            weight, weight_scale, block_size=bs, orig_dtype=target_dtype
                        )
                    else:
                        dequantized_weight = weight.to(target_dtype) * weight_scale.to(target_dtype)
                else:
                    dequantized_weight = weight.to(target_dtype)
            elif weight.dtype == torch.int8:
                if weight_scale is not None:
                    if weight_scale.numel() == 1 or (
                        weight_scale.ndim == 1 and weight_scale.shape[0] == weight.shape[0]
                    ):
                        dequantized_weight = TensorWiseINT8Layout.dequantize(
                            weight, weight_scale, is_weight=True, orig_dtype=target_dtype
                        )
                    elif weight_scale.ndim >= 2:
                        M = weight.shape[0]
                        bs = max(1, M // weight_scale.shape[0])
                        dequantized_weight = BlockWiseINT8Layout.dequantize(
                            weight, weight_scale, block_size=bs, is_weight=True, orig_dtype=target_dtype
                        )
                    else:
                        dequantized_weight = weight.to(target_dtype) * weight_scale.to(target_dtype)
                else:
                    dequantized_weight = weight.to(target_dtype)
            else:
                if weight.is_floating_point():
                    dequantized_weight = weight.to(target_dtype)
                else:
                    dequantized_weight = weight

        if dequantized_weight is not None:
            output_tensors[f"{base_name}.weight"] = dequantized_weight.to(target_dtype)
            if (
                weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8)
                or weight_scale is not None
                or comfy_quant is not None
            ):
                dequantized_count += 1
            else:
                passthrough_count += 1

    # Process other tensors (bias, norm weights, embeddings, etc.)
    for key, tensor in tqdm(other_tensors.items(), desc="Processing non-weight tensors"):
        if tensor.is_floating_point():
            output_tensors[key] = tensor.to(target_dtype)
        else:
            output_tensors[key] = tensor

    info("Dequantization summary:")
    info(f"  Dequantized layers : {dequantized_count}")
    info(f"  Passthrough layers : {passthrough_count + len(other_tensors)}")
    info(f"  Total output tensors: {len(output_tensors)}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    minimal(f"Saving dequantized model to: {output_file}")
    save_file(output_tensors, output_file, metadata=original_metadata)
    info("Done!")
