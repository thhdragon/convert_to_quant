"""
FP8 conversion functions for convert_to_quant.

Main quantization function that processes safetensors files and applies
FP8/INT8 quantization with learned rounding optimization.
"""

import gc
import json
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from safetensors.torch import save_file

from ..config.layer_config import get_layer_settings
from ..constants import COMPUTE_DTYPE, FP8_MAX, FP8_MIN, INT4_MAX, INT4_MIN, INT8_SYMMETRIC_MAX, MODEL_FILTERS, NORMALIZE_SCALES_ENABLED, SCALE_DTYPE, T5XXL_REMOVE_KEY_NAMES, TARGET_FP8_DTYPE, TARGET_INT8_DTYPE
from ..converters.learned_int4 import LearnedINT4Converter
from ..converters.learned_mxfp8 import LearnedMXFP8Converter
from ..converters.learned_nvfp4 import LearnedNVFP4Converter
from ..converters.learned_rounding import LearnedRoundingConverter
from ..utils.comfy_quant import create_comfy_quant_tensor, should_skip_layer_for_performance
from ..utils.logging import error, info, log_debug, minimal, verbose, warning
from ..utils.memory_efficient_loader import MemoryEfficientSafeOpen
from ..utils.tensor_utils import normalize_tensorwise_scales


@log_debug
def convert_to_fp8_scaled(
    input_file: str,
    output_file: str,
    comfy_quant: bool,
    filter_flags: Dict[str, bool],
    calib_samples: int,
    seed: int,
    calib_cpu: bool = False,
    int8: bool = False,
    primary_format: Optional[str] = None,  # Override: "nvfp4", "mxfp8", or None (use int8 flag)
    fallback: Optional[str] = None,
    custom_layers: Optional[str] = None,
    exclude_layers: Optional[str] = None,
    custom_type: Optional[str] = None,
    custom_block_size: Optional[int] = None,
    custom_scaling_mode: Optional[str] = None,
    custom_simple: bool = False,
    custom_heur: bool = False,
    custom_full_precision_mm: bool = False,
    custom_convrot: bool = False,
    custom_convrot_group_size: int = 256,
    convrot: bool = False,
    convrot_group_size: int = 256,
    dynamic_convrot: bool = False,
    fallback_block_size: Optional[int] = None,
    fallback_simple: bool = False,
    full_precision_matrix_mult: bool = False,
    skip_inefficient_layers: bool = False,
    include_input_scale: bool = False,
    no_learned_rounding: bool = False,
    save_quant_metadata: bool = True,
    layer_config: Optional[Dict[str, Any]] = None,
    layer_config_fullmatch: bool = False,
    low_memory: bool = False,
    device: Optional[str] = None,
    devices: Optional[Union[str, List[str]]] = None,
    # LoRA extraction options
    extract_lora: bool = False,
    lora_rank: int = 16,
    lora_target: Optional[str] = None,
    lora_depth: int = -1,
    lora_ar_threshold: float = 0.0,
    lora_save_path: Optional[str] = None,
    # Added for CLI compatibility
    lora_output: Optional[str] = None,
    input_scales: Optional[Dict[str, Any]] = None,
    actcal_lora: Optional[str] = None,
    **converter_kwargs,
):
    # Ensure filter_flags is a dict
    filter_flags = filter_flags or {}

    from ..utils.parallel_utils import parse_devices, run_parallel_layer_processing

    target_devices = parse_devices(device=device, devices=devices)
    device = target_devices[0]

    # Determine target format (priority: primary_format > int8 > fp8)
    if primary_format:
        target_format = primary_format
        format_name = primary_format.upper()
    elif int8:
        target_format = "int8"
        format_name = "INT8"
    else:
        target_format = "fp8"
        format_name = "FP8"

    info(f"Processing: {input_file}\nOutput will be saved to: {output_file}")
    info("-" * 60)
    if target_format in ("int4", "convrot_w4a4"):
        info("Target format: INT4 ConvRot W4A4 (4-bit signed quantization)")
        info(f"INT4 Range: [{INT4_MIN}, {INT4_MAX}]")
    elif target_format == "int8" or int8:
        if convrot:
            info("Target format: INT8 ConvRot (row-wise quantization with Hadamard rotation)")
        else:
            info("Target format: INT8 (block-wise quantization)")
        info(f"INT8 Range: [{-INT8_SYMMETRIC_MAX}, {INT8_SYMMETRIC_MAX}]")
    elif target_format == "nvfp4":
        info("Target format: NVFP4 (NVIDIA FP4 E2M1)")
    elif target_format == "mxfp8":
        info("Target format: MXFP8 (Microscaling FP8)")
    else:
        info(f"Target FP8 format: {TARGET_FP8_DTYPE}\nFP8 Range: [{FP8_MIN}, {FP8_MAX}]")
    info("-" * 60)

    # Enforce CUDA for kernel-dependent formats if they are the PRIMARY target
    if target_format in ("mxfp8", "nvfp4") and device == "cpu":
        warning(f"Format {target_format} requires CUDA kernels. Forcing device='cuda'.")
        device = "cuda"

    # Calibration cache configuration
    calib_cache_dir = None
    if low_memory and not calib_cpu:
        out_dir = os.path.dirname(os.path.abspath(output_file)) or "."
        os.makedirs(out_dir, exist_ok=True)
        calib_cache_dir = tempfile.mkdtemp(prefix="ctq_calib_", dir=out_dir)
        info(f"Using disk-based calibration cache: {calib_cache_dir}")
        seed_device = "cpu"
    else:
        seed_device = "cpu"  # Always use CPU for generation to avoid VRAM leak

    seed_generator = torch.Generator(device=seed_device)
    seed_generator.manual_seed(seed)

    if comfy_quant:
        info("Comfy quantization mode enabled: Using comfy_quant layer names and settings.")
        comfy_quant = True
    else:
        comfy_quant = True

    # Use unified loader (handles both standard and low-memory modes)
    try:
        loader = MemoryEfficientSafeOpen(input_file, low_memory=low_memory)
    except Exception as e:
        error(f"FATAL: Error loading '{input_file}': {e}")
        if calib_cache_dir and os.path.exists(calib_cache_dir):
            shutil.rmtree(calib_cache_dir)
        return

    all_keys = loader.keys()

    # Read original file metadata to preserve during conversion
    original_metadata = loader.metadata()

    # Initialize metadata collection if enabled
    quant_metadata_layers = {} if save_quant_metadata else None

    # Add target_format and no_learned_rounding to converter kwargs
    converter_kwargs["target_format"] = target_format
    converter_kwargs["no_learned_rounding"] = no_learned_rounding
    converter_kwargs["device"] = device

    # Add ConvRot options to converter kwargs
    converter_kwargs["convrot"] = convrot
    converter_kwargs["convrot_group_size"] = convrot_group_size
    converter_kwargs["dynamic_convrot"] = dynamic_convrot

    # Add LoRA options to converter kwargs
    converter_kwargs["extract_lora"] = extract_lora
    converter_kwargs["lora_rank"] = lora_rank
    converter_kwargs["lora_target"] = lora_target
    converter_kwargs["lora_depth"] = lora_depth
    converter_kwargs["lora_ar_threshold"] = lora_ar_threshold

    # Get format-aware block_size default (converters handle their own fixed sizes)
    # This is only used for metadata/display; converters use their __init__ defaults
    format_block_sizes = {"nvfp4": 16, "mxfp8": 32, "int8": 128, "int4": 64, "convrot_w4a4": 64, "fp8": 64}
    block_size = converter_kwargs.get("block_size") or format_block_sizes.get(target_format, 64)

    # Helper function to create converter for a specific format type
    def create_converter_for_format(fmt: str, overrides: dict = None, is_primary: bool = True, target_device: str = None):
        """Create appropriate converter instance for the given format.

        Args:
            fmt: Format string (fp8, int8, mxfp8, nvfp4)
            overrides: Parameter overrides for this specific converter
            is_primary: If True, inherit no_learned_rounding from global --simple.
                        If False (custom/fallback), only use override value.
            target_device: Optional GPU/CPU device override for this converter instance.
        """
        kwargs = converter_kwargs.copy()
        kwargs["target_format"] = fmt
        if target_device:
            kwargs["device"] = target_device

        # Custom/fallback should NOT inherit global no_learned_rounding
        # They use their own --custom-simple / --fallback-simple flags
        if not is_primary:
            kwargs["no_learned_rounding"] = False  # Default to learned rounding

        if overrides:
            kwargs.update(overrides)

        if fmt == "mxfp8":
            # MXFP8 has fixed block_size=32, remove incompatible kwargs
            mxfp8_kwargs = {k: v for k, v in kwargs.items() if k not in ("target_format", "scaling_mode", "block_size")}
            return LearnedMXFP8Converter(**mxfp8_kwargs)
        elif fmt == "nvfp4":
            # NVFP4 has fixed block_size=16, remove incompatible kwargs
            nvfp4_kwargs = {k: v for k, v in kwargs.items() if k not in ("target_format", "scaling_mode", "block_size")}
            return LearnedNVFP4Converter(**nvfp4_kwargs)
        elif fmt in ("int4", "convrot_w4a4"):
            int4_kwargs = kwargs.copy()
            int4_kwargs["target_format"] = "int4"
            int4_kwargs["convrot"] = True
            int4_kwargs["scaling_mode"] = "row"
            return LearnedINT4Converter(**int4_kwargs)
        else:
            return LearnedRoundingConverter(**kwargs)

    # Helper function to get format metadata
    def get_format_info(fmt: str) -> dict:
        """Returns dtype and format name for a quantization format."""
        format_map = {
            "int8": {"dtype": TARGET_INT8_DTYPE, "name": "INT8"},
            "int4": {"dtype": TARGET_INT8_DTYPE, "name": "INT4 ConvRot W4A4"},
            "convrot_w4a4": {"dtype": TARGET_INT8_DTYPE, "name": "INT4 ConvRot W4A4"},
            "fp8": {"dtype": TARGET_FP8_DTYPE, "name": "FP8"},
            "mxfp8": {"dtype": torch.uint8, "name": "MXFP8"},
            "nvfp4": {"dtype": torch.uint8, "name": "NVFP4"},
        }
        return format_map.get(fmt, format_map["fp8"])

    # Create converters for each format type used
    converters = {"primary": create_converter_for_format(target_format)}

    # Create fallback converter with optional overrides
    if fallback:
        fallback_overrides = {}
        if fallback_block_size is not None:
            fallback_overrides["block_size"] = fallback_block_size
        if fallback_simple:
            fallback_overrides["no_learned_rounding"] = True
        converters["fallback"] = create_converter_for_format(fallback, fallback_overrides if fallback_overrides else None, is_primary=False)
        override_note = f" (block_size={fallback_block_size})" if fallback_block_size else ""
        override_note += " (simple)" if fallback_simple else ""
        info(f"Fallback quantization enabled: {fallback.upper()}{override_note} for excluded layers")

    # Create custom converter with optional overrides
    if custom_layers and custom_type:
        custom_overrides = {}
        if custom_block_size is not None:
            custom_overrides["block_size"] = custom_block_size
        if custom_scaling_mode is not None:
            custom_overrides["scaling_mode"] = custom_scaling_mode
        if custom_simple:
            custom_overrides["no_learned_rounding"] = True
        if custom_convrot:
            custom_overrides["convrot"] = True
            custom_overrides["convrot_group_size"] = custom_convrot_group_size
        converters["custom"] = create_converter_for_format(custom_type, custom_overrides if custom_overrides else None, is_primary=False)
        override_note = f" (block_size={custom_block_size})" if custom_block_size else ""
        override_note += f" (scaling_mode={custom_scaling_mode})" if custom_scaling_mode else ""
        override_note += " (simple)" if custom_simple else ""
        info(f"Custom layer quantization enabled: {custom_type.upper()}{override_note} for pattern '{custom_layers}'")

    # Compile custom_layers regex pattern
    custom_pattern = None
    if custom_layers:
        try:
            custom_pattern = re.compile(custom_layers)
        except re.error as e:
            error(f"ERROR: Invalid regex pattern '{custom_layers}': {e}")
            if calib_cache_dir and os.path.exists(calib_cache_dir):
                shutil.rmtree(calib_cache_dir)
            return

    # Compile exclude_layers regex pattern
    exclude_pattern = None
    if exclude_layers:
        try:
            exclude_pattern = re.compile(exclude_layers)
            info(f"Layer exclusion enabled: pattern '{exclude_layers}'")
        except re.error as e:
            error(f"ERROR: Invalid regex pattern '{exclude_layers}': {e}")
            if calib_cache_dir and os.path.exists(calib_cache_dir):
                shutil.rmtree(calib_cache_dir)
            return

    lora_key_map = {}
    if actcal_lora and os.path.exists(actcal_lora):
        try:
            from ..calibrate_activation_scales import build_lora_key_map, load_lora_tensors
            info(f"Loading LoRA for weight optimization calibration: {actcal_lora}")
            actcal_lora_tensors = load_lora_tensors(actcal_lora)
            all_linear_bases = [k[:-7] for k in all_keys if k.endswith(".weight") and loader.get_ndim(k) == 2]
            lora_key_map = build_lora_key_map(all_linear_bases, actcal_lora_tensors)
            info(f"Matched {len(lora_key_map)} layers with LoRA calibration data for weight optimization")
        except Exception as e:
            warning(f"Failed to load LoRA for weight optimization calibration: {e}")

    calibration_data_cache = {}
    # Generate calibration data for bias correction (always, even in simple mode)
    if lora_key_map:
        minimal("Scanning model and preparing LoRA-informed calibration data for weight optimization...")
    else:
        minimal("Scanning model and generating simulated calibration data...")
    for key in all_keys:
        if key.endswith(".weight"):
            shape = loader.get_shape(key)
            if len(shape) == 2:
                in_features = shape[1]
                if in_features not in calibration_data_cache:
                    verbose(f"  - Found new input dimension: {in_features}.")
                    calib_tensor = torch.randn(calib_samples, in_features, dtype=COMPUTE_DTYPE, generator=seed_generator, device=seed_device)

                    if calib_cache_dir:
                        # Save to disk as safetensors
                        cache_path = os.path.join(calib_cache_dir, f"calib_{in_features}.safetensors")
                        save_file({"calib_data": calib_tensor}, cache_path)
                        calibration_data_cache[in_features] = cache_path
                        del calib_tensor
                    else:
                        # Store in CPU memory
                        calibration_data_cache[in_features] = calib_tensor
    info("Simulated calibration data generated.\n")

    new_tensors: Dict[str, torch.Tensor] = {}
    lora_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted([key for key in all_keys if key.endswith(".weight") and loader.get_ndim(key) == 2])
    total_weights = len(weight_keys)
    skipped_count = 0
    processed_count = 0
    custom_count = 0
    dequant_w = None  # Track dequantized weights for bias correction
    fallback_count = 0

    info(f"Found {total_weights} weight tensors to potentially process.")
    info("-" * 60)

    work_items = list(enumerate(weight_keys))

    def process_layer_item(item: Tuple[int, str], dev: str) -> Dict[str, Any]:
        i, key = item
        exclusion_reason = ""
        use_custom = False
        use_fallback = False
        use_layer_config = False
        layer_format = target_format
        layer_settings = None

        text_encoder_filter = filter_flags.get("t5xxl") or filter_flags.get("mistral") or filter_flags.get("visual") or filter_flags.get("generic_text")

        if filter_flags.get("t5xxl") and any(n in key for n in T5XXL_REMOVE_KEY_NAMES):
            info(f"[{dev}] ({i + 1}/{total_weights}) Removing T5XXL decoder tensor: {key}")
            return {"key": key, "removed": True, "skipped": True}

        if layer_config:
            layer_settings = get_layer_settings(key, layer_config, fullmatch=layer_config_fullmatch)
            if layer_settings:
                if layer_settings.get("skip", False):
                    info(f"[{dev}] ({i + 1}/{total_weights}) Skipping (layer-config): {key}")
                    original_tensor = loader.get_tensor(key)
                    return {"key": key, "skipped": True, "tensors": {key: original_tensor.to(device="cpu", dtype=original_tensor.dtype)}}
                use_layer_config = True
                fmt = layer_settings["format"]
                if fmt.startswith("float8"):
                    layer_format = "fp8"
                elif fmt.startswith("int8"):
                    layer_format = "int8"
                elif fmt == "mxfp8":
                    layer_format = "mxfp8"
                elif fmt == "nvfp4":
                    layer_format = "nvfp4"
                elif fmt in ("int4", "convrot_w4a4"):
                    layer_format = "convrot_w4a4"
                else:
                    layer_format = "fp8"

        if not use_layer_config and custom_pattern and custom_pattern.search(key):
            use_custom = True
            layer_format = custom_type

        if not use_custom and not use_layer_config and exclude_pattern and exclude_pattern.search(key):
            exclusion_reason = "regex exclusion (--exclude-layers)"

        if not use_custom and not use_layer_config:
            active_filters = filter_flags
            for filter_name, is_active in active_filters.items():
                if not is_active:
                    continue
                cfg = MODEL_FILTERS[filter_name]
                skip_patterns = cfg.get("exclude", []) + cfg.get("highprec", [])
                if skip_patterns and any(n in key for n in skip_patterns):
                    exclusion_reason = f"{filter_name} skip"
                    break

        if exclusion_reason and not use_custom and not use_layer_config:
            if fallback:
                use_fallback = True
                layer_format = fallback
                info(f"[{dev}] ({i + 1}/{total_weights}) Processing (fallback {fallback.upper()}): {key} (was: {exclusion_reason})")
            else:
                info(f"[{dev}] ({i + 1}/{total_weights}) Skipping tensor: {key} (Reason: {exclusion_reason})")
                original_tensor = loader.get_tensor(key)
                return {"key": key, "skipped": True, "tensors": {key: original_tensor.to(device="cpu", dtype=original_tensor.dtype)}}

        if use_layer_config:
            fmt = layer_settings["format"]
            info(f"[{dev}] ({i + 1}/{total_weights}) Processing (config {fmt}): {key}")
        elif use_custom:
            info(f"[{dev}] ({i + 1}/{total_weights}) Processing (custom {custom_type.upper()}): {key}")
        elif use_fallback:
            info(f"[{dev}] ({i + 1}/{total_weights}) Processing (fallback {fallback.upper()}): {key}")
        else:
            info(f"[{dev}] ({i + 1}/{total_weights}) Processing ({format_name}): {key}")

        original_tensor = loader.get_tensor(key)
        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            info(f"  - Skipping empty or non-2D tensor: {key}")
            return {"key": key, "skipped": True, "tensors": {key: original_tensor.to(device="cpu", dtype=original_tensor.dtype)}}

        apply_heur = custom_heur if use_custom else skip_inefficient_layers
        if apply_heur:
            active_block_size = block_size
            if use_layer_config and layer_settings:
                active_block_size = layer_settings.get("block_size", active_block_size)
            elif use_custom and custom_block_size is not None:
                active_block_size = custom_block_size
            elif use_fallback and fallback_block_size is not None:
                active_block_size = fallback_block_size
            should_skip, skip_perf_reason = should_skip_layer_for_performance(original_tensor, active_block_size)
            if should_skip:
                info(f"  - Skipping for performance: {skip_perf_reason}")
                return {"key": key, "skipped": True, "tensors": {key: original_tensor.to(device="cpu", dtype=original_tensor.dtype)}}

        if use_layer_config:
            cfg_overrides = {}
            cfg_block_size = layer_settings.get("block_size")
            cfg_scaling_mode = layer_settings.get("scaling_mode")
            cfg_simple = layer_settings.get("simple", False)
            if cfg_block_size is not None:
                cfg_overrides["block_size"] = cfg_block_size
            if cfg_scaling_mode is not None:
                cfg_overrides["scaling_mode"] = cfg_scaling_mode
            if cfg_simple:
                cfg_overrides["no_learned_rounding"] = True
            converter = create_converter_for_format(layer_format, cfg_overrides if cfg_overrides else None, target_device=dev)
        elif use_custom:
            converter = create_converter_for_format(custom_type, is_primary=False, target_device=dev)
        elif use_fallback:
            converter = create_converter_for_format(fallback, is_primary=False, target_device=dev)
        else:
            converter = create_converter_for_format(target_format, is_primary=True, target_device=dev)

        is_int8 = layer_format == "int8"
        is_mxfp8 = layer_format == "mxfp8"
        is_nvfp4 = layer_format == "nvfp4"
        is_int4 = layer_format in ("int4", "convrot_w4a4")

        depth = -1
        depth_match = re.search(r"\.(\d+)\.", key)
        if depth_match:
            depth = int(depth_match.group(1))

        convrot_applied = False
        convrot_group_size = 256
        if is_int4 or (hasattr(converter, "convrot") and getattr(converter, "convrot") and getattr(converter, "scaling_mode", "") == "row"):
            in_features = original_tensor.shape[1]
            dynamic_convrot = getattr(converter, "dynamic_convrot", False)
            if dynamic_convrot:
                from ..utils.convrot import find_max_compatible_group_size

                min_gs = getattr(converter, "convrot_group_size", 256)
                layer_gs = find_max_compatible_group_size(in_features, min_group_size=min_gs)
                if layer_gs is not None:
                    convrot_group_size = layer_gs
                    convrot_applied = True
            else:
                convrot_group_size = getattr(converter, "convrot_group_size", 256)
                if in_features % convrot_group_size == 0:
                    convrot_applied = True

        temp_base_name = key[: key.rfind(".weight")]
        has_bias = f"{temp_base_name}.bias" in all_keys

        res_tensors = {}
        res_lora = {}

        if is_mxfp8:
            q_tensor, block_scales, dequant_w, extra_tensors = converter.convert(original_tensor, key=key, depth=depth, has_bias=has_bias)
            dequant_s = block_scales
        elif is_nvfp4:
            q_tensor, block_scales, per_tensor_scale, dequant_w, extra_tensors = converter.convert(original_tensor, key=key, depth=depth, has_bias=has_bias)
            dequant_s = block_scales
        else:
            in_features = original_tensor.shape[1]
            base_name = key[: key.rfind(".weight")]

            lora_calib = None
            if lora_key_map and base_name in lora_key_map and "lora_A" in lora_key_map[base_name]:
                lora_A = lora_key_map[base_name]["lora_A"]
                x_base = lora_A.to(dtype=COMPUTE_DTYPE, device="cpu")
                gen = torch.Generator(device="cpu").manual_seed(seed)
                n1 = torch.randn(x_base.shape, generator=gen, dtype=COMPUTE_DTYPE)
                n2 = torch.randn(x_base.shape, generator=gen, dtype=COMPUTE_DTYPE)
                x_calib = torch.cat([x_base, x_base + 0.1 * n1, x_base + 0.2 * n2, x_base * -1])
                lora_calib = x_calib / x_calib.std().clamp(min=1e-6)

            if lora_calib is not None:
                calibration_data = lora_calib
                calib_data_loaded = False
            else:
                cache_entry = calibration_data_cache.get(in_features)
                calib_data_loaded = False
                if isinstance(cache_entry, str):
                    with MemoryEfficientSafeOpen(cache_entry, low_memory=True) as calib_loader:
                        calibration_data = calib_loader.get_tensor("calib_data")
                    calib_data_loaded = True
                else:
                    calibration_data = cache_entry

            q_tensor, dequant_s, dequant_w, extra_tensors = converter.convert(original_tensor, key=key, depth=depth, calibration_data=calibration_data, has_bias=has_bias)

            if calib_data_loaded and calibration_data is not None:
                del calibration_data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        res_tensors[key] = q_tensor.to(device="cpu")
        base_name = key[: key.rfind(".weight")]
        bias_key = f"{base_name}.bias"

        if comfy_quant is True:
            layer_block_size = converter.block_size
            layer_full_precision_mm = full_precision_matrix_mult
            if use_layer_config and "full_precision_matrix_mult" in layer_settings:
                layer_full_precision_mm = layer_settings["full_precision_matrix_mult"]
            elif use_custom and custom_full_precision_mm:
                layer_full_precision_mm = True

            comfy_quant_format = None
            block_size_for_meta = None

            if is_mxfp8:
                res_tensors[f"{base_name}.weight_scale"] = block_scales.to(device="cpu")
                comfy_quant_format = "mxfp8"
                block_size_for_meta = 32
                comfy_quant_tensor = create_comfy_quant_tensor("mxfp8", block_size=32, full_precision_matrix_mult=layer_full_precision_mm if layer_full_precision_mm else None)
            elif is_nvfp4:
                res_tensors[f"{base_name}.weight_scale"] = block_scales.to(device="cpu")
                res_tensors[f"{base_name}.weight_scale_2"] = per_tensor_scale.to(device="cpu", dtype=torch.float32)
                comfy_quant_format = "nvfp4"
                block_size_for_meta = 16
                comfy_quant_tensor = create_comfy_quant_tensor("nvfp4", block_size=16, full_precision_matrix_mult=layer_full_precision_mm if layer_full_precision_mm else None)
            elif is_int4:
                res_tensors[f"{base_name}.weight_scale"] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                comfy_quant_format = "convrot_w4a4"
                block_size_for_meta = layer_block_size
                comfy_quant_tensor = create_comfy_quant_tensor(
                    "convrot_w4a4",
                    block_size=block_size_for_meta,
                    full_precision_matrix_mult=layer_full_precision_mm if layer_full_precision_mm else None,
                    convrot=True,
                    convrot_groupsize=convrot_group_size if convrot_applied else 256,
                )
            elif is_int8:
                res_tensors[f"{base_name}.weight_scale"] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                per_row = False
                if getattr(converter, "scaling_mode", "") in ("tensor", "row"):
                    comfy_quant_format = "int8_tensorwise"
                    block_size_for_meta = None
                    if getattr(converter, "scaling_mode", "") == "row":
                        per_row = True
                else:
                    comfy_quant_format = "int8_blockwise"
                    block_size_for_meta = layer_block_size

                comfy_quant_tensor = create_comfy_quant_tensor(
                    comfy_quant_format,
                    block_size=block_size_for_meta,
                    full_precision_matrix_mult=layer_full_precision_mm if layer_full_precision_mm else None,
                    convrot=convrot_applied,
                    convrot_groupsize=convrot_group_size if convrot_applied else None,
                    per_row=per_row if getattr(converter, "scaling_mode", "") == "row" else None,
                )
                if comfy_quant_format == "int8_blockwise":
                    res_tensors[f"{base_name}.input_scale"] = torch.tensor(1.0, dtype=torch.float32, device="cpu")
            else:
                res_tensors[f"{base_name}.weight_scale"] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                if use_layer_config:
                    fp8_format = layer_settings["format"]
                    fp8_block_size = layer_settings.get("block_size", layer_block_size)
                elif getattr(converter, "scaling_mode", "") == "row":
                    fp8_format = "float8_e4m3fn_rowwise"
                    fp8_block_size = None
                elif getattr(converter, "scaling_mode", "") in ("block", "block2d"):
                    fp8_format = "float8_e4m3fn_blockwise"
                    fp8_block_size = layer_block_size
                elif getattr(converter, "scaling_mode", "") == "block3d":
                    fp8_format = "float8_e4m3fn"
                    fp8_block_size = None
                else:
                    fp8_format = "float8_e4m3fn"
                    fp8_block_size = None

                comfy_quant_format = fp8_format
                block_size_for_meta = fp8_block_size

                comfy_quant_tensor = create_comfy_quant_tensor(fp8_format, block_size=fp8_block_size, full_precision_matrix_mult=layer_full_precision_mm if layer_full_precision_mm else None)
                if input_scales and base_name in input_scales:
                    val = input_scales[base_name]
                    if isinstance(val, torch.Tensor):
                        val = val.item() if val.numel() == 1 else val
                    res_tensors[f"{base_name}.input_scale"] = torch.tensor(val, dtype=torch.float32, device="cpu")
                elif include_input_scale or text_encoder_filter:
                    if text_encoder_filter:
                        res_tensors[f"{base_name}.input_scale"] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                    else:
                        res_tensors[f"{base_name}.input_scale"] = torch.tensor(1.0, dtype=torch.float32, device="cpu")

            res_tensors[f"{base_name}.comfy_quant"] = comfy_quant_tensor.to(device="cpu")

            meta_entry = None
            if quant_metadata_layers is not None:
                meta_entry = {"format": comfy_quant_format}
                block_based_formats = {"int8_blockwise", "float8_e4m3fn_blockwise", "mxfp8", "nvfp4", "convrot_w4a4"}
                if block_size_for_meta is not None and comfy_quant_format in block_based_formats:
                    meta_entry["group_size"] = block_size_for_meta
                if layer_full_precision_mm:
                    meta_entry["full_precision_matrix_mult"] = True
                if is_int4 or convrot_applied:
                    meta_entry["convrot"] = True
                    meta_entry["convrot_groupsize"] = convrot_group_size if convrot_applied else 256

        else:
            res_tensors[f"{base_name}.scale_weight"] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
            if input_scales and base_name in input_scales:
                val = input_scales[base_name]
                if isinstance(val, torch.Tensor):
                    val = val.item() if val.numel() == 1 else val
                res_tensors[f"{base_name}.scale_input"] = torch.tensor(val, dtype=SCALE_DTYPE, device="cpu")
            elif include_input_scale or text_encoder_filter:
                if text_encoder_filter:
                    res_tensors[f"{base_name}.scale_input"] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                else:
                    res_tensors[f"{base_name}.scale_input"] = torch.ones_like(dequant_s, dtype=SCALE_DTYPE, device="cpu")
            meta_entry = None

        if "bias_correction" in extra_tensors:
            if bias_key in all_keys:
                with torch.no_grad():
                    bias_correction = extra_tensors["bias_correction"].cpu()
                    info(f"  - Adjusting corresponding bias using ConvRot-specific calibration: {bias_key}")
                    original_bias = loader.get_tensor(bias_key)
                    b_new = (original_bias.to(dtype=COMPUTE_DTYPE) + bias_correction.to(dtype=COMPUTE_DTYPE)).to(dtype=original_bias.dtype)
                    res_tensors[bias_key] = b_new
        elif bias_key in all_keys:
            if dequant_w is not None:
                info(f"  - Adjusting corresponding bias: {bias_key}")
                with torch.no_grad():
                    original_bias = loader.get_tensor(bias_key)
                    in_features = original_tensor.shape[1]
                    if lora_calib is not None:
                        calib_data = lora_calib
                    elif in_features not in calibration_data_cache:
                        warning(f"  - WARNING: No calibration data for bias correction of {bias_key}.")
                        res_tensors[bias_key] = original_bias.to("cpu")
                    else:
                        cache_entry = calibration_data_cache[in_features]
                        if isinstance(cache_entry, str):
                            with MemoryEfficientSafeOpen(cache_entry, low_memory=True) as calib_loader:
                                calib_data = calib_loader.get_tensor("calib_data")
                        else:
                            calib_data = cache_entry

                        total_samples = calib_data.shape[0]
                        current_samples = total_samples
                        min_samples = max(1, int(total_samples * 0.1))
                        retry_count = 0
                        max_retries = 10

                        while True:
                            try:
                                X_calib_dev = calib_data[:current_samples].to(device=dev)
                                W_orig_dev = original_tensor.to(device=dev, dtype=COMPUTE_DTYPE)
                                W_dequant_dev = dequant_w.to(device=dev, dtype=COMPUTE_DTYPE)
                                b_orig_dev = original_bias.to(device=dev, dtype=COMPUTE_DTYPE)
                                weight_error = W_orig_dev - W_dequant_dev
                                output_error = X_calib_dev @ weight_error.T
                                bias_correction = output_error.mean(dim=0)
                                b_new = b_orig_dev - bias_correction
                                res_tensors[bias_key] = b_new.to(device="cpu", dtype=original_bias.dtype)
                                del (W_orig_dev, W_dequant_dev, X_calib_dev, b_orig_dev, weight_error, output_error, bias_correction, b_new)
                                if str(dev).startswith("cuda"):
                                    torch.cuda.empty_cache()
                                break
                            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                                is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
                                if not is_oom:
                                    raise

                                for var in ["X_calib_dev", "W_orig_dev", "W_dequant_dev", "b_orig_dev", "weight_error", "output_error", "bias_correction", "b_new"]:
                                    if var in locals():
                                        try:
                                            del locals()[var]
                                        except KeyError:
                                            pass
                                gc.collect()
                                if str(dev).startswith("cuda"):
                                    torch.cuda.empty_cache()

                                retry_count += 1
                                if retry_count > max_retries or current_samples <= min_samples:
                                    warning("  - WARNING: OOM during bias correction even after reducing samples. Giving up.")
                                    raise

                                current_samples = max(min_samples, int(current_samples * 0.7))

                        if isinstance(cache_entry, str):
                            del calib_data
                            gc.collect()
            else:
                res_tensors[bias_key] = loader.get_tensor(bias_key).to("cpu")

        if text_encoder_filter:
            if comfy_quant and f"{base_name}.input_scale" not in res_tensors:
                res_tensors[f"{base_name}.input_scale"] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
            elif not comfy_quant and f"{base_name}.scale_input" not in res_tensors:
                res_tensors[f"{base_name}.scale_input"] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()

        info("-" * 60)

        del original_tensor, q_tensor, dequant_s, dequant_w
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

        return {
            "base_name": base_name,
            "tensors": res_tensors,
            "lora_tensors": res_lora,
            "meta_entry": meta_entry,
            "skipped": False,
            "use_custom": use_custom,
            "use_fallback": use_fallback,
            "use_layer_config": use_layer_config,
        }

    layer_results = run_parallel_layer_processing(work_items, process_layer_item, target_devices)

    for r in layer_results:
        if r.get("skipped"):
            skipped_count += 1
        else:
            processed_count += 1
            if r.get("use_custom") or r.get("use_layer_config"):
                custom_count += 1
            elif r.get("use_fallback"):
                fallback_count += 1

        if "tensors" in r:
            new_tensors.update(r["tensors"])
        if "lora_tensors" in r:
            lora_tensors.update(r["lora_tensors"])
        if "base_name" in r and r.get("meta_entry") and quant_metadata_layers is not None:
            quant_metadata_layers[r["base_name"]] = r["meta_entry"]

    # Copy remaining tensors (bias, norms, etc.)
    for key in all_keys:
        if any(n in key for n in T5XXL_REMOVE_KEY_NAMES) and filter_flags.get("t5xxl"):
            continue
        if key not in new_tensors:
            new_tensors[key] = loader.get_tensor(key)

    # Close loader to release file handle
    loader.close()

    # Free calibration data and force garbage collection before save
    calibration_data_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Add scaled_fp8 marker only for legacy non-comfy_quant FP8 format
    # Use empty((0)) when input_scale is present (t5xxl, mistral, or --input_scale flag)
    if not comfy_quant and not int8 and not custom_layers and "scaled_fp8" not in new_tensors:
        has_text_encoder_filter = filter_flags.get("t5xxl") or filter_flags.get("mistral") or filter_flags.get("visual")
        new_tensors["scaled_fp8"] = torch.empty((0), dtype=TARGET_FP8_DTYPE) if (has_text_encoder_filter or include_input_scale or bool(input_scales)) else torch.empty((2), dtype=TARGET_FP8_DTYPE)

    info(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        # Prepare metadata args - preserve original metadata and merge with quant metadata
        output_metadata = dict(original_metadata)  # Start with original file metadata
        if save_quant_metadata and quant_metadata_layers:
            full_metadata = {"format_version": "1.0", "layers": quant_metadata_layers}
            output_metadata["_quantization_metadata"] = json.dumps(full_metadata)
            info(f"  Adding quantization metadata for {len(quant_metadata_layers)} layers")
        save_kwargs = {"metadata": output_metadata} if output_metadata else {}

        # Normalize any 1-element scale tensors to scalars
        new_tensors, normalized_count = normalize_tensorwise_scales(new_tensors, NORMALIZE_SCALES_ENABLED)
        if normalized_count > 0:
            info(f"  Normalized {normalized_count} scale tensors to scalars")
        save_file(new_tensors, output_file, **save_kwargs)

        # Save extracted LoRA adapter if any
        if lora_tensors:
            if not lora_save_path:
                lora_save_path = lora_output or output_file.replace(".safetensors", "_lora.safetensors")

            info(f"Saving {len(lora_tensors)} LoRA tensors to {lora_save_path}")
            save_file(lora_tensors, lora_save_path)

        info("Conversion complete!")
    except Exception as e:
        error(f"FATAL: Error saving file '{output_file}': {e}")
        if calib_cache_dir and os.path.exists(calib_cache_dir):
            shutil.rmtree(calib_cache_dir)
        return

    if calib_cache_dir and os.path.exists(calib_cache_dir):
        shutil.rmtree(calib_cache_dir)
        verbose(f"Cleaned up calibration cache: {calib_cache_dir}")

    info("-" * 60)
    info("Summary:")
    summary_parts = [f"  - Original tensor count : {len(all_keys)}", f"  - Weights processed     : {processed_count}"]
    if custom_count > 0:
        summary_parts.append(f"    - Custom type layers  : {custom_count}")
    if fallback_count > 0:
        summary_parts.append(f"    - Fallback type layers: {fallback_count}")
    summary_parts.extend([f"  - Weights skipped       : {skipped_count}", f"  - Final tensor count    : {len(new_tensors)}"])
    info("\n".join(summary_parts))
    info("-" * 60)
