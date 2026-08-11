"""
Format migration utilities for convert_to_quant.

Converts between legacy quantization formats and comfy_quant format.
"""

import gc
import json
import os
import re
from typing import Any, Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from ..constants import COMPUTE_DTYPE, NORMALIZE_SCALES_ENABLED, SCALE_DTYPE, TARGET_FP8_DTYPE
from ..utils.comfy_quant import create_comfy_quant_tensor, fix_comfy_quant_params_structure
from ..utils.logging import debug, error, info, log_debug, minimal, verbose, warning
from ..utils.tensor_utils import dict_to_tensor, normalize_tensorwise_scales, tensor_to_dict


def convert_fp8_scaled_to_comfy_quant(input_file: str, output_file: str, hp_filter: Optional[str] = None, full_precision_mm: bool = False, include_input_scale: bool = False, save_quant_metadata: bool = True):
    """
    Convert legacy fp8_scaled format to comfy_quant format.

    This is a format conversion only - NO quantization is performed.
    FP8 layers are detected by weight dtype (float8_e4m3fn), not by scale presence.
    High-precision layers may have dummy .scale_weight which are removed.

    Args:
        input_file: Path to input fp8_scaled safetensors file
        output_file: Path to output comfy_quant safetensors file
        hp_filter: Optional regex pattern to validate high-precision layers
        full_precision_mm: If True, set full_precision_matrix_mult in .comfy_quant
        include_input_scale: If True, add input_scale tensor (1.0 fp32) when missing
    """
    info("Converting fp8_scaled to comfy_quant format")
    info(f"Input: {input_file}")
    info(f"Output: {output_file}")
    info("-" * 60)

    # Load input tensors and preserve original metadata
    tensors: Dict[str, torch.Tensor] = {}
    original_metadata: Dict[str, str] = {}
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            # Preserve original file metadata
            original_metadata = f.metadata() or {}
            if original_metadata:
                verbose(f"Preserving {len(original_metadata)} original metadata entries")

            minimal(f"Loading {len(f.keys())} tensors from source file...")
            for key in tqdm(f.keys(), desc="Loading tensors"):
                tensors[key] = f.get_tensor(key)
    except Exception as e:
        error(f"FATAL: Error loading '{input_file}': {e}")
        return

    # Initialize metadata collection if enabled
    quant_metadata_layers = {} if save_quant_metadata else None

    # Verify this is an fp8_scaled model
    if "scaled_fp8" not in tensors:
        error("ERROR: This does not appear to be an fp8_scaled model (missing 'scaled_fp8' marker)")
        error("       Use this mode only for legacy fp8_scaled format models.")
        return

    info("Verified: Input is fp8_scaled format")

    # Compile hp_filter regex if provided
    hp_pattern = None
    if hp_filter:
        try:
            hp_pattern = re.compile(hp_filter)
            info(f"High-precision filter: {hp_filter}")
        except re.error as e:
            error(f"ERROR: Invalid regex pattern '{hp_filter}': {e}")
            return

    # Group tensors by layer base name
    # Find all .weight tensors and their associated scales
    layer_info: Dict[str, Dict[str, torch.Tensor]] = {}
    other_tensors: Dict[str, torch.Tensor] = {}

    for key, tensor in tensors.items():
        if key == "scaled_fp8":
            continue  # Skip marker, will be removed

        # Parse layer and suffix
        if key.endswith(".weight"):
            base = key[: -len(".weight")]
            if base not in layer_info:
                layer_info[base] = {}
            layer_info[base]["weight"] = tensor
        elif key.endswith(".scale_weight"):
            base = key[: -len(".scale_weight")]
            if base not in layer_info:
                layer_info[base] = {}
            layer_info[base]["scale_weight"] = tensor
        elif key.endswith(".scale_input"):
            base = key[: -len(".scale_input")]
            if base not in layer_info:
                layer_info[base] = {}
            layer_info[base]["scale_input"] = tensor
        else:
            other_tensors[key] = tensor

    # Process layers
    output_tensors: Dict[str, torch.Tensor] = {}
    fp8_layers = []
    hp_layers = []

    for base_name, layer_data in tqdm(layer_info.items(), desc="Processing layers"):
        weight = layer_data.get("weight")
        scale_weight = layer_data.get("scale_weight")
        scale_input = layer_data.get("scale_input")

        if weight is None:
            # No weight tensor - just copy any scales through (unusual case)
            if scale_weight is not None:
                warning(f"  WARNING: {base_name} has scale_weight but no weight tensor")
                output_tensors[f"{base_name}.scale_weight"] = scale_weight
            if scale_input is not None:
                output_tensors[f"{base_name}.scale_input"] = scale_input
            continue

        # Detect if this is an FP8 layer by weight dtype
        is_fp8 = weight.dtype == TARGET_FP8_DTYPE

        if is_fp8:
            # FP8 layer: rename scales and add .comfy_quant
            fp8_layers.append(base_name)
            output_tensors[f"{base_name}.weight"] = weight

            if scale_weight is not None:
                output_tensors[f"{base_name}.weight_scale"] = scale_weight
            else:
                warning(f"  WARNING: FP8 layer {base_name} missing scale_weight")

            # Handle scale_input -> input_scale
            if scale_input is not None:
                output_tensors[f"{base_name}.input_scale"] = scale_input
            elif include_input_scale:
                # No scale_input but flag is set - add default input_scale (scalar)
                output_tensors[f"{base_name}.input_scale"] = torch.tensor(1.0, dtype=torch.float32)

            # Detect format and block_size from scale_weight tensor shape
            # Scale shape conventions from quant_ops.py layouts:
            # - TensorCoreFP8Layout: () or (1,) - scalar, single global scale
            # - RowWiseFP8Layout: (M,) - 1D, one scale per output row
            # - BlockWiseFP8Layout: (M//bs, N//bs) - 2D grid, one scale per tile
            # - Block3DFP8Layout: (M, N//bs, 1) - 3D, per-row-block scaling
            M, N = weight.shape[0], weight.shape[1] if weight.ndim >= 2 else 1

            if scale_weight is None:
                # No scale tensor - assume tensor-wise (this shouldn't happen for valid FP8 models)
                format_type = "float8_e4m3fn"
                block_size = None
                verbose(f"    → Format: {format_type} (missing scale, assumed tensor-wise)")
            elif scale_weight.numel() == 1:
                # Scalar or single-element tensor → tensor-wise scaling
                format_type = "float8_e4m3fn"
                block_size = None
                verbose(f"    → Format: {format_type} (scale numel=1)")
            elif scale_weight.ndim == 1:
                # 1D scale tensor - check if it matches row count
                if scale_weight.shape[0] == M:
                    # One scale per row → row-wise
                    format_type = "float8_e4m3fn_rowwise"
                    block_size = None
                    verbose(f"    → Format: {format_type} (scale shape={scale_weight.shape}, M={M})")
                else:
                    # 1D but doesn't match M - could be flattened block scale
                    # Try to infer block_size: scale_count = (M//bs) * (N//bs) = M*N / bs^2
                    # So bs = sqrt(M*N / scale_count)
                    scale_count = scale_weight.shape[0]
                    total_elements = M * N
                    if scale_count > 0 and total_elements % scale_count == 0:
                        bs_squared = total_elements // scale_count
                        bs = int(bs_squared**0.5)
                        if bs * bs == bs_squared and M % bs == 0 and N % bs == 0:
                            format_type = "float8_e4m3fn_blockwise"
                            block_size = bs
                            verbose(f"    → Format: {format_type} (scale 1D flattened, inferred bs={bs})")
                        else:
                            format_type = "float8_e4m3fn"
                            block_size = None
                            verbose(f"    → Format: {format_type} (scale 1D unknown pattern, fallback)")
                    else:
                        format_type = "float8_e4m3fn"
                        block_size = None
                        verbose(f"    → Format: {format_type} (scale 1D, cannot infer block)")
            elif scale_weight.ndim == 2:
                # 2D scale - most likely block-wise: (M//bs, N//bs)
                scale_M, scale_N = scale_weight.shape
                if M % scale_M == 0 and N % scale_N == 0:
                    bs_M = M // scale_M
                    bs_N = N // scale_N
                    if bs_M == bs_N:
                        # Square blocks
                        format_type = "float8_e4m3fn_blockwise"
                        block_size = bs_M
                        verbose(f"    → Format: {format_type} (scale 2D, bs={block_size})")
                    else:
                        # Non-square blocks - use smaller dimension as block_size
                        format_type = "float8_e4m3fn_blockwise"
                        block_size = min(bs_M, bs_N)
                        verbose(f"    → Format: {format_type} (scale 2D non-square, bs={block_size})")
                else:
                    # Doesn't divide evenly - fallback
                    format_type = "float8_e4m3fn"
                    block_size = None
                    verbose(f"    → Format: {format_type} (scale 2D but dims don't divide)")
            elif scale_weight.ndim == 3:
                # 3D scale - likely Block3DFP8Layout: (M, N//bs, 1)
                scale_M, scale_blocks, scale_last = scale_weight.shape
                if scale_M == M and scale_last == 1 and N % scale_blocks == 0:
                    format_type = "float8_e4m3fn_block3d"
                    block_size = N // scale_blocks
                    verbose(f"    → Format: {format_type} (scale 3D, bs={block_size})")
                else:
                    format_type = "float8_e4m3fn"
                    block_size = None
                    verbose(f"    → Format: {format_type} (scale 3D unknown pattern)")
            else:
                # Unknown ndim
                format_type = "float8_e4m3fn"
                block_size = None
                verbose(f"    → Format: {format_type} (scale ndim={scale_weight.ndim} unknown)")

            # Create .comfy_quant metadata
            comfy_quant_tensor = create_comfy_quant_tensor(format_type, block_size=block_size, full_precision_matrix_mult=full_precision_mm if full_precision_mm else None)
            output_tensors[f"{base_name}.comfy_quant"] = comfy_quant_tensor

            # Collect metadata if enabled
            if save_quant_metadata:
                meta_entry = {"format": format_type}
                block_based_formats = {"int8_blockwise", "float8_e4m3fn_blockwise"}
                if block_size is not None and format_type in block_based_formats:
                    meta_entry["group_size"] = block_size
                if full_precision_mm:
                    meta_entry["full_precision_matrix_mult"] = True

                quant_metadata_layers[base_name] = meta_entry

        else:
            # High-precision layer: keep weight, remove dummy scales
            hp_layers.append(base_name)
            output_tensors[f"{base_name}.weight"] = weight

            if scale_weight is not None:
                verbose(f"  Removing dummy scale_weight from high-precision layer: {base_name}")
            if scale_input is not None:
                verbose(f"  Removing dummy scale_input from high-precision layer: {base_name}")

    # Add other tensors (bias, norms, etc.) - also fix any incorrect comfy_quant structures
    fixed_comfy_quant_count = 0
    for key, tensor in other_tensors.items():
        if key.endswith(".comfy_quant"):
            fixed_tensor, was_fixed = fix_comfy_quant_params_structure(tensor)
            if was_fixed:
                fixed_comfy_quant_count += 1
            output_tensors[key] = fixed_tensor
        else:
            output_tensors[key] = tensor

    # Validate hp_filter if provided
    if hp_pattern:
        info("\nValidating high-precision filter...")
        violations = []
        for base_name in fp8_layers:
            if hp_pattern.search(base_name):
                violations.append(base_name)

        if violations:
            error("ERROR: The following layers matched hp-filter but are FP8 (not high-precision):")
            for v in violations:
                error(f"  - {v}")
            error("\nThese layers have float8_e4m3fn weights. If they should be high-precision,")
            error("the input model needs to be regenerated with correct layer exclusions.")
            return

        # Report matched hp layers
        matched_hp = [b for b in hp_layers if hp_pattern.search(b)]
        if matched_hp:
            info(f"  Validated {len(matched_hp)} high-precision layers match filter")

    # Summary
    info("\n" + "-" * 60)
    info("Conversion Summary:")
    info(f"  FP8 layers:            {len(fp8_layers)}")
    info(f"  High-precision layers: {len(hp_layers)}")
    info(f"  Other tensors:         {len(other_tensors)}")
    info(f"  Total output tensors:  {len(output_tensors)}")
    if fixed_comfy_quant_count > 0:
        info(f"  Fixed comfy_quant:     {fixed_comfy_quant_count} (nested params → flat)")
    info("-" * 60)

    # Save output
    info(f"\nSaving to {output_file}...")
    try:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

        # Prepare metadata args - merge original metadata with new quantization metadata
        output_metadata = dict(original_metadata)  # Start with original metadata

        if save_quant_metadata and quant_metadata_layers:
            full_metadata = {"format_version": "1.0", "layers": quant_metadata_layers}
            output_metadata["_quantization_metadata"] = json.dumps(full_metadata)
            verbose(f"  Adding quantization metadata for {len(quant_metadata_layers)} layers")

        save_kwargs = {"metadata": output_metadata} if output_metadata else {}

        # Normalize any 1-element scale tensors to scalars
        output_tensors, normalized_count = normalize_tensorwise_scales(output_tensors, NORMALIZE_SCALES_ENABLED)
        if normalized_count > 0:
            verbose(f"  Normalized {normalized_count} scale tensors to scalars")
        save_file(output_tensors, output_file, **save_kwargs)

        info("Conversion complete!")
    except Exception as e:
        error(f"FATAL: Error saving file '{output_file}': {e}")
        return


def scan_and_replace_comfy_quant_metadata(
    input_file: str,
    output_file: str,
    default_block_size: Optional[int] = None,
    full_precision_mm: bool = False,
    include_input_scale: bool = False,
    strip_non_comfy_metadata: bool = True,
    int4: bool = False,
    convrot: bool = False,
    convrot_group_size: int = 256,
):
    """
    Scan a quantized model, auto-detect layer quantization formats,
    generate/replace .comfy_quant tensors, and replace header metadata with
    standardized ComfyQuant _quantization_metadata.

    This function inspects all layers and:
    1. Standardizes legacy scale tensor names (.scale_weight -> .weight_scale, .scale_input -> .input_scale).
    2. Auto-detects layer quantization format (FP8, INT8, NVFP4, MXFP8, ConvRot W4A4, etc.)
       and quantization parameters (block/group size, matrix mult flags).
    3. Creates or updates .comfy_quant tensors for all quantized layers.
    4. Strips obsolete/non-comfy quantization metadata keys from the safetensors header
       (e.g., quantization_config, quant_method, scaled_fp8, scaled_int8, etc.).
    5. Populates header metadata with _quantization_metadata.

    Args:
        input_file: Path to input safetensors model file
        output_file: Path to output safetensors model file
        default_block_size: Default block/group size for blockwise formats (defaults: FP8/INT8=128, INT4=64)
        full_precision_mm: Set full_precision_matrix_mult=True in generated layer configs
        include_input_scale: Add default input_scale tensor (1.0 fp32) for quantized layers missing it
        strip_non_comfy_metadata: Strip non-comfy quantization metadata header keys
        int4: Force INT4 / ConvRot W4A4 quantization format detection
        convrot: Force ConvRot Hadamard rotation flag
        convrot_group_size: ConvRot group size (default 256)
    """
    info("Scanning model and replacing metadata with ComfyQuant format")
    info(f"Input:  {input_file}")
    info(f"Output: {output_file}")
    info("-" * 60)

    # Load input tensors and original header metadata
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

    # Check if original header metadata hints at ConvRot / INT4
    header_is_convrot = any(
        s in str(k).lower() or s in str(v).lower()
        for k, v in original_metadata.items()
        for s in ("convrot", "w4a4", "int4_convrot", "convrot_w4a4")
    )

    # Non-comfy quantization metadata keys to purge from file header
    NON_COMFY_QUANT_META_KEYS = {
        "quantization_config",
        "quant_method",
        "quantization",
        "scaled_fp8",
        "scaled_int8",
        "bitsandbytes",
        "format",
        "_quantization_metadata",
    }

    # Prepare cleaned file header metadata
    output_metadata: Dict[str, str] = {}
    removed_header_keys = []
    for k, v in original_metadata.items():
        if strip_non_comfy_metadata and k in NON_COMFY_QUANT_META_KEYS:
            removed_header_keys.append(k)
        else:
            output_metadata[k] = v

    if removed_header_keys:
        info(f"Purging non-comfy quantization header metadata keys: {removed_header_keys}")

    # Group tensors by layer base name
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
    )

    for key, tensor in tensors.items():
        if key in ("scaled_fp8", "scaled_int8"):
            continue  # Strip obsolete marker tensors

        matched_suffix = None
        for suffix in QUANT_SUFFIXES:
            if key.endswith(suffix):
                matched_suffix = suffix
                break

        if matched_suffix:
            base = key[:-len(matched_suffix)]
            field = matched_suffix[1:]
            if base not in layer_info:
                layer_info[base] = {}
            layer_info[base][field] = tensor
        else:
            other_tensors[key] = tensor

    output_tensors: Dict[str, torch.Tensor] = {}
    quant_metadata_layers: Dict[str, Any] = {}
    quantized_layer_count = 0
    detected_formats: Dict[str, int] = {}

    for base_name, layer_data in tqdm(layer_info.items(), desc="Scanning layers"):
        weight = layer_data.get("weight")
        existing_comfy_quant = layer_data.get("comfy_quant")
        weight_scale = (
            layer_data.get("weight_scale")
            if layer_data.get("weight_scale") is not None
            else layer_data.get("scale_weight")
        )
        input_scale = (
            layer_data.get("input_scale")
            if layer_data.get("input_scale") is not None
            else layer_data.get("scale_input")
        )
        per_tensor_scale = layer_data.get("per_tensor_scale")

        if weight is None:
            # Pass through non-weight layer components
            for sub_k, sub_v in layer_data.items():
                if sub_k not in ("scale_weight", "scale_input", "comfy_quant"):
                    output_tensors[f"{base_name}.{sub_k}"] = sub_v
            continue

        # Standardize weight output
        output_tensors[f"{base_name}.weight"] = weight

        # Check existing comfy_quant tensor config first
        existing_config: Optional[Dict[str, Any]] = None
        if existing_comfy_quant is not None:
            try:
                fixed_cq, _ = fix_comfy_quant_params_structure(existing_comfy_quant)
                existing_config = tensor_to_dict(fixed_cq)
            except Exception:
                existing_config = None

        # Determine if this layer is quantized
        is_fp8 = weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        is_integer = weight.dtype in (torch.int8, torch.uint8)
        has_scales = weight_scale is not None or per_tensor_scale is not None
        is_quantized = is_fp8 or is_integer or has_scales or (existing_config is not None)

        if is_quantized:
            quantized_layer_count += 1

            # Standardize scale tensors
            if weight_scale is not None:
                output_tensors[f"{base_name}.weight_scale"] = weight_scale
            if input_scale is not None:
                output_tensors[f"{base_name}.input_scale"] = input_scale
            elif include_input_scale:
                output_tensors[f"{base_name}.input_scale"] = torch.tensor(1.0, dtype=torch.float32)
            if per_tensor_scale is not None:
                output_tensors[f"{base_name}.per_tensor_scale"] = per_tensor_scale

            # Infer or decode format parameters
            format_type = "float8_e4m3fn" if is_fp8 else "int8_blockwise"
            block_size: Optional[int] = None
            fpmm = full_precision_mm if full_precision_mm else None
            convrot_flag: Optional[bool] = convrot if convrot else None
            convrot_gs: Optional[int] = convrot_group_size if convrot else None
            per_row: Optional[bool] = None

            # Detect ConvRot / INT4 / W4A8 indicators
            is_convrot_layer = int4 or convrot or header_is_convrot
            is_w4a8_layer = False
            if existing_config:
                existing_fmt = str(existing_config.get("format", "")).lower()
                if existing_fmt in ("convrot_w4a4", "int4_convrot", "int4", "int4_blockwise", "w4a4") or existing_config.get("convrot") is True:
                    is_convrot_layer = True
                elif existing_fmt in ("w4a8_int8", "asym_w4a8_int8", "w4a8", "asymw4a8int8layout"):
                    is_w4a8_layer = True
            elif data.get("_s_rel") is not None and data.get("_s_channel") is not None:
                is_w4a8_layer = True

            if existing_config:
                format_type = existing_config.get("format", format_type)
                block_size = existing_config.get("group_size") or existing_config.get("block_size")
                if "full_precision_matrix_mult" in existing_config:
                    fpmm = existing_config["full_precision_matrix_mult"]
                if existing_config.get("convrot") is True:
                    convrot_flag = True
                    convrot_gs = existing_config.get("convrot_groupsize", convrot_group_size)
                per_row = existing_config.get("per_row")

            M, N = weight.shape[0], weight.shape[1] if weight.ndim >= 2 else 1

            if is_w4a8_layer:
                format_type = "w4a8_int8"
                convrot_flag = True
                convrot_gs = convrot_gs or convrot_group_size
                if block_size is None:
                    block_size = default_block_size if default_block_size is not None else 16
            elif is_convrot_layer and is_integer:
                format_type = "convrot_w4a4" if format_type not in ("convrot_w4a4", "int4_convrot") else format_type
                convrot_flag = True
                convrot_gs = convrot_gs or convrot_group_size
                if block_size is None:
                    block_size = default_block_size if default_block_size is not None else 64
            elif is_fp8 and (not existing_config or "format" not in existing_config):
                if weight_scale is None or weight_scale.numel() == 1:
                    format_type = "float8_e4m3fn"
                    block_size = None
                elif weight_scale.ndim == 1:
                    if weight_scale.shape[0] == M:
                        format_type = "float8_e4m3fn_rowwise"
                        block_size = None
                    else:
                        scale_count = weight_scale.shape[0]
                        total_elements = M * N
                        if scale_count > 0 and total_elements % scale_count == 0:
                            bs_sq = total_elements // scale_count
                            bs = int(bs_sq**0.5)
                            if bs * bs == bs_sq:
                                format_type = "float8_e4m3fn_blockwise"
                                block_size = bs
                elif weight_scale.ndim == 2:
                    scale_M, scale_N = weight_scale.shape
                    if M % scale_M == 0 and N % scale_N == 0:
                        bs_M, bs_N = M // scale_M, N // scale_N
                        format_type = "float8_e4m3fn_blockwise"
                        block_size = bs_M if bs_M == bs_N else min(bs_M, bs_N)

            elif is_integer and (not existing_config or "format" not in existing_config):
                fallback_bs = default_block_size if default_block_size is not None else 128
                if weight_scale is None or weight_scale.numel() == 1:
                    format_type = "int8_tensorwise"
                    block_size = None
                elif weight_scale.ndim == 1:
                    if weight_scale.shape[0] in (M, N):
                        format_type = "int8_tensorwise"
                        block_size = None
                    else:
                        format_type = "int8_blockwise"
                        block_size = fallback_bs
                elif weight_scale.ndim == 2:
                    scale_M, scale_N = weight_scale.shape
                    if scale_M == N and scale_N == 1:
                        format_type = "int8_tensorwise"
                        block_size = None
                    elif M % scale_M == 0 and N % scale_N == 0:
                        bs_M, bs_N = M // scale_M, N // scale_N
                        format_type = "int8_blockwise"
                        block_size = bs_M if bs_M == bs_N else min(bs_M, bs_N)
                    else:
                        format_type = "int8_blockwise"
                        block_size = fallback_bs

            # Ensure block-based formats have group_size set
            if format_type in ("int8_blockwise", "float8_e4m3fn_blockwise", "convrot_w4a4", "int4_convrot", "w4a8_int8") and block_size is None:
                block_size = default_block_size if default_block_size is not None else (16 if format_type == "w4a8_int8" else (64 if format_type in ("convrot_w4a4", "int4_convrot") else 128))

            # Create updated .comfy_quant tensor
            comfy_quant_tensor = create_comfy_quant_tensor(
                format_type=format_type,
                block_size=block_size,
                full_precision_matrix_mult=fpmm,
                convrot=convrot_flag,
                convrot_groupsize=convrot_gs,
                per_row=per_row,
                scale_dtype=existing_config.get("scale_dtype") if existing_config else None,
                symmetric=existing_config.get("symmetric") if existing_config else None,
                codebook=existing_config.get("codebook") if existing_config else None,
            )
            output_tensors[f"{base_name}.comfy_quant"] = comfy_quant_tensor

            # Build metadata entry
            meta_entry = tensor_to_dict(comfy_quant_tensor)
            quant_metadata_layers[base_name] = meta_entry
            detected_formats[format_type] = detected_formats.get(format_type, 0) + 1


    # Preserve other non-layer tensors
    for key, tensor in other_tensors.items():
        if key.endswith(".comfy_quant"):
            fixed_tensor, _ = fix_comfy_quant_params_structure(tensor)
            output_tensors[key] = fixed_tensor
        else:
            output_tensors[key] = tensor

    # Populate header _quantization_metadata JSON
    if quant_metadata_layers:
        full_metadata = {"format_version": "1.0", "layers": quant_metadata_layers}
        output_metadata["_quantization_metadata"] = json.dumps(full_metadata)
        info(f"Populated _quantization_metadata header with {len(quant_metadata_layers)} layer entries")

    # Summary
    info("-" * 60)
    info("Scan & Replace Summary:")
    info(f"  Quantized layers processed: {quantized_layer_count}")
    info(f"  Total output tensors:       {len(output_tensors)}")
    if detected_formats:
        info("  Layer formats detected:")
        for fmt, count in sorted(detected_formats.items(), key=lambda x: -x[1]):
            info(f"    {fmt}: {count} layers")
    info("-" * 60)

    # Save output file
    info(f"\nSaving to {output_file}...")
    try:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        output_tensors, normalized_count = normalize_tensorwise_scales(output_tensors, NORMALIZE_SCALES_ENABLED)
        if normalized_count > 0:
            verbose(f"  Normalized {normalized_count} scale tensors to scalars")

        save_kwargs = {"metadata": output_metadata} if output_metadata else {}
        save_file(output_tensors, output_file, **save_kwargs)
        info("Metadata scan and replacement complete!")
    except Exception as e:
        error(f"FATAL: Error saving file '{output_file}': {e}")
        return

