"""
INT4 ConvRot W4A4 conversion functions for convert_to_quant.

Main quantization pipeline function that processes safetensors files and applies
INT4 W4A4 quantization with group-wise Hadamard rotation (ConvRot) and learned rounding optimization.
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
from ..constants import COMPUTE_DTYPE, INT4_MAX, INT4_MIN, MODEL_FILTERS, NORMALIZE_SCALES_ENABLED, SCALE_DTYPE, T5XXL_REMOVE_KEY_NAMES
from ..converters.learned_int4 import LearnedINT4Converter
from ..utils.comfy_quant import create_comfy_quant_tensor, should_skip_layer_for_performance
from ..utils.logging import error, info, log_debug, minimal, verbose, warning
from ..utils.memory_efficient_loader import MemoryEfficientSafeOpen
from ..utils.tensor_utils import normalize_tensorwise_scales


@log_debug
def convert_to_int4_convrot(
    input_file: str,
    output_file: str,
    comfy_quant: bool = True,
    filter_flags: Optional[Dict[str, bool]] = None,
    calib_samples: int = 8192,
    seed: int = -1,
    calib_cpu: bool = False,
    custom_layers: Optional[str] = None,
    exclude_layers: Optional[str] = None,
    block_size: int = 64,
    convrot_group_size: int = 256,
    dynamic_convrot: bool = False,
    w4a4_untouched_activations: bool = False,
    full_precision_matrix_mult: bool = False,
    custom_full_precision_mm: bool = False,
    skip_inefficient_layers: bool = False,
    no_learned_rounding: bool = False,
    save_quant_metadata: bool = True,
    layer_config: Optional[Dict[str, Any]] = None,
    layer_config_fullmatch: bool = False,
    low_memory: bool = False,
    device: Optional[str] = None,
    devices: Optional[Union[str, List[str]]] = None,
    # LoRA extraction options
    extract_lora: bool = False,
    lora_rank: int = 32,
    lora_target: Optional[str] = None,
    lora_depth: int = -1,
    lora_ar_threshold: float = 0.0,
    lora_save_path: Optional[str] = None,
    lora_output: Optional[str] = None,
    actcal_lora: Optional[str] = None,
    # Checkpointing & Resume options
    resume: bool = False,
    sidecar_path: Optional[str] = None,
    max_shard_size: Optional[Union[str, int]] = None,
    no_checkpoint: bool = False,
    **converter_kwargs,
):
    """
    Main conversion entry point for INT4 ConvRot W4A4 quantization.
    """
    filter_flags = filter_flags or {}

    from ..utils.checkpoint import QuantCheckpointManager
    from ..utils.parallel_utils import parse_devices, run_parallel_layer_processing

    target_devices = parse_devices(device=device, devices=devices)
    device = target_devices[0]

    target_format = "convrot_w4a4"

    checkpoint_mgr = QuantCheckpointManager(
        output_file=output_file,
        input_file=input_file,
        primary_format=target_format,
        resume=resume,
        sidecar_path=sidecar_path,
        max_shard_size=max_shard_size,
        no_checkpoint=no_checkpoint,
    )

    info(f"Processing: {input_file}\nOutput will be saved to: {output_file}")
    info("-" * 60)
    info("Target format: INT4 ConvRot W4A4 (4-bit signed quantization with group Hadamard rotation)")
    info(f"INT4 Range: [{INT4_MIN}, {INT4_MAX}]")
    info(f"Block size: {block_size}, ConvRot group size: {convrot_group_size}")
    info("-" * 60)

    # Calibration cache configuration
    calib_cache_dir = None
    if low_memory and not calib_cpu:
        out_dir = os.path.dirname(os.path.abspath(output_file)) or "."
        os.makedirs(out_dir, exist_ok=True)
        calib_cache_dir = tempfile.mkdtemp(prefix="ctq_calib_", dir=out_dir)
        info(f"Using disk-based calibration cache: {calib_cache_dir}")
        seed_device = "cpu"
    else:
        seed_device = "cpu"

    seed_generator = torch.Generator(device=seed_device)
    if seed != -1:
        seed_generator.manual_seed(seed)

    try:
        loader = MemoryEfficientSafeOpen(input_file, low_memory=low_memory)
    except Exception as e:
        error(f"FATAL: Error loading '{input_file}': {e}")
        if calib_cache_dir and os.path.exists(calib_cache_dir):
            shutil.rmtree(calib_cache_dir)
        return

    all_keys = loader.keys()
    original_metadata = loader.metadata()
    quant_metadata_layers = {} if save_quant_metadata else None

    # Configure converter parameters for INT4 ConvRot W4A4
    converter_kwargs["target_format"] = "int4"
    converter_kwargs["scaling_mode"] = "row"
    converter_kwargs["block_size"] = block_size
    converter_kwargs["convrot"] = True
    converter_kwargs["convrot_group_size"] = convrot_group_size
    converter_kwargs["dynamic_convrot"] = dynamic_convrot
    converter_kwargs["w4a4_untouched_activations"] = w4a4_untouched_activations
    converter_kwargs["no_learned_rounding"] = no_learned_rounding
    converter_kwargs["device"] = device
    converter_kwargs["extract_lora"] = extract_lora
    converter_kwargs["lora_rank"] = lora_rank
    converter_kwargs["lora_target"] = lora_target
    converter_kwargs["lora_depth"] = lora_depth
    converter_kwargs["lora_ar_threshold"] = lora_ar_threshold

    def create_int4_converter(target_device: Optional[str] = None):
        kwargs = converter_kwargs.copy()
        if target_device:
            kwargs["device"] = target_device
        return LearnedINT4Converter(**kwargs)

    # Compile regex patterns
    custom_pattern = None
    if custom_layers:
        try:
            custom_pattern = re.compile(custom_layers)
        except re.error as e:
            error(f"ERROR: Invalid regex pattern '{custom_layers}': {e}")
            if calib_cache_dir and os.path.exists(calib_cache_dir):
                shutil.rmtree(calib_cache_dir)
            return

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
    minimal("Scanning model and generating calibration data...")
    for key in all_keys:
        if key.endswith(".weight"):
            shape = loader.get_shape(key)
            if len(shape) == 2:
                in_features = shape[1]
                if in_features not in calibration_data_cache:
                    verbose(f"  - Found new input dimension: {in_features}.")
                    calib_tensor = torch.randn(calib_samples, in_features, dtype=COMPUTE_DTYPE, generator=seed_generator, device=seed_device)
                    if calib_cache_dir:
                        cache_path = os.path.join(calib_cache_dir, f"calib_{in_features}.safetensors")
                        save_file({"calib_data": calib_tensor}, cache_path)
                        calibration_data_cache[in_features] = cache_path
                        del calib_tensor
                    else:
                        calibration_data_cache[in_features] = calib_tensor

    info("Calibration data prepared.\n")

    new_tensors: Dict[str, torch.Tensor] = {}
    lora_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted([key for key in all_keys if key.endswith(".weight") and loader.get_ndim(key) == 2])
    total_weights = len(weight_keys)
    skipped_count = 0
    processed_count = 0

    info(f"Found {total_weights} weight tensors to process.")
    info("-" * 60)

    work_items = list(enumerate(weight_keys))

    def process_layer_item(item: Tuple[int, str], dev: str) -> Dict[str, Any]:
        i, key = item
        if checkpoint_mgr.is_layer_completed(key):
            info(f"[{dev}] ({i + 1}/{total_weights}) Skipping (loaded from sidecar checkpoint): {key}")
            loaded_res = checkpoint_mgr.load_completed_layer(key)
            if loaded_res is not None:
                return loaded_res

        exclusion_reason = ""
        use_custom = False

        if filter_flags.get("t5xxl") and any(n in key for n in T5XXL_REMOVE_KEY_NAMES):
            info(f"[{dev}] ({i + 1}/{total_weights}) Removing T5XXL decoder tensor: {key}")
            return {"key": key, "removed": True, "skipped": True}

        if layer_config:
            layer_settings = get_layer_settings(key, layer_config, fullmatch=layer_config_fullmatch)
            if layer_settings and layer_settings.get("skip", False):
                info(f"[{dev}] ({i + 1}/{total_weights}) Skipping (layer-config): {key}")
                original_tensor = loader.get_tensor(key)
                return {"key": key, "skipped": True, "tensors": {key: original_tensor.to(device="cpu", dtype=original_tensor.dtype)}}

        base_name = key[: key.rfind(".weight")] if key.endswith(".weight") else key
        if custom_pattern and (custom_pattern.search(key) or custom_pattern.search(base_name)):
            use_custom = True

        if not use_custom and exclude_pattern and exclude_pattern.search(key):
            exclusion_reason = "regex exclusion (--exclude-layers)"

        if not use_custom:
            for filter_name, is_active in filter_flags.items():
                if not is_active:
                    continue
                cfg = MODEL_FILTERS[filter_name]
                skip_patterns = cfg.get("exclude", []) + cfg.get("highprec", [])
                if skip_patterns and any(n in key for n in skip_patterns):
                    exclusion_reason = f"{filter_name} skip"
                    break

        if exclusion_reason and not use_custom:
            info(f"[{dev}] ({i + 1}/{total_weights}) Skipping tensor: {key} (Reason: {exclusion_reason})")
            original_tensor = loader.get_tensor(key)
            return {"key": key, "skipped": True, "tensors": {key: original_tensor.to(device="cpu", dtype=original_tensor.dtype)}}

        info(f"[{dev}] ({i + 1}/{total_weights}) Processing (INT4 ConvRot W4A4): {key}")

        original_tensor = loader.get_tensor(key)
        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            info(f"  - Skipping empty or non-2D tensor: {key}")
            return {"key": key, "skipped": True, "tensors": {key: original_tensor.to(device="cpu", dtype=original_tensor.dtype)}}

        if skip_inefficient_layers:
            should_skip, skip_perf_reason = should_skip_layer_for_performance(original_tensor, block_size)
            if should_skip:
                info(f"  - Skipping for performance: {skip_perf_reason}")
                return {"key": key, "skipped": True, "tensors": {key: original_tensor.to(device="cpu", dtype=original_tensor.dtype)}}

        converter = create_int4_converter(target_device=dev)

        depth = -1
        depth_match = re.search(r"\.(\d+)\.", key)
        if depth_match:
            depth = int(depth_match.group(1))

        # Check ConvRot compatibility with feature dimensions
        if any(dim < 256 for dim in original_tensor.shape) or any(dim % 256 != 0 for dim in original_tensor.shape):
            info(f"  - Dimension ({list(original_tensor.shape)}) not compatible with convrot (requires feature dim >= 256 and divisible by 256); keeping untouched in bf16")
            return {"key": key, "skipped": True, "tensors": {key: original_tensor.to(device="cpu", dtype=torch.bfloat16)}}

        convrot_group_size_layer = convrot_group_size

        temp_base_name = key[: key.rfind(".weight")]
        has_bias = f"{temp_base_name}.bias" in all_keys

        res_tensors = {}
        res_lora = {}

        in_features = original_tensor.shape[1]
        base_name = key[: key.rfind(".weight")]

        lora_calib = None
        if lora_key_map and base_name in lora_key_map and "lora_A" in lora_key_map[base_name]:
            lora_A = lora_key_map[base_name]["lora_A"]
            x_base = lora_A.to(dtype=COMPUTE_DTYPE, device="cpu")
            gen = torch.Generator(device="cpu").manual_seed(seed if seed != -1 else 42)
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

        q_tensor, dequant_s, dequant_w, extra_tensors = converter.convert(
            original_tensor, key=key, depth=depth, calibration_data=calibration_data, has_bias=has_bias
        )

        if calib_data_loaded and calibration_data is not None:
            del calibration_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if extra_tensors:
            base_key = key[: key.rfind(".weight")]
            for lora_key, lora_tensor in extra_tensors.items():
                if lora_key in ("lora_up", "lora_down"):
                    if base_key.startswith("model.diffusion_model."):
                        clean_base = base_key[6:]
                        full_lora_key = f"{clean_base}.{lora_key}.weight"
                    elif base_key.startswith("diffusion_model.") or base_key.startswith("text_encoders."):
                        full_lora_key = f"{base_key}.{lora_key}.weight"
                    elif base_key.startswith("model."):
                        clean_base = base_key[6:]
                        full_lora_key = f"diffusion_model.{clean_base}.{lora_key}.weight"
                    else:
                        full_lora_key = f"diffusion_model.{base_key}.{lora_key}.weight"
                    res_lora[full_lora_key] = lora_tensor.cpu()

        res_tensors[key] = q_tensor.to(device="cpu")
        bias_key = f"{base_name}.bias"

        layer_fpmm = full_precision_matrix_mult or (use_custom and custom_full_precision_mm)
        res_tensors[f"{base_name}.weight_scale"] = dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
        comfy_quant_tensor = create_comfy_quant_tensor(
            "convrot_w4a4",
            block_size=block_size,
            full_precision_matrix_mult=layer_fpmm if layer_fpmm else None,
            convrot=True,
            convrot_groupsize=convrot_group_size_layer,
        )
        res_tensors[f"{base_name}.comfy_quant"] = comfy_quant_tensor.to(device="cpu")

        meta_entry = {
            "format": "convrot_w4a4",
            "group_size": block_size,
            "convrot": True,
            "convrot_groupsize": convrot_group_size_layer,
        }
        if layer_fpmm:
            meta_entry["full_precision_matrix_mult"] = True

        if "bias_correction" in extra_tensors:
            if bias_key in all_keys:
                with torch.no_grad():
                    bias_correction = extra_tensors["bias_correction"].cpu()
                    info(f"  - Adjusting corresponding bias using ConvRot calibration: {bias_key}")
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

                        while True:
                            try:
                                X_calib_dev = calib_data[:current_samples].to(device=dev)
                                W_orig_dev = original_tensor.to(device=dev, dtype=COMPUTE_DTYPE)
                                W_dequant_dev = dequant_w.to(device=dev, dtype=COMPUTE_DTYPE)
                                b_orig_dev = original_bias.to(device=dev, dtype=COMPUTE_DTYPE)
                                weight_error = W_orig_dev - W_dequant_dev
                                output_error = X_calib_dev @ weight_error.T
                                # Outlier-robust 5% trimmed mean bias correction
                                sorted_err, _ = output_error.sort(dim=0)
                                trim_n = max(1, int(sorted_err.shape[0] * 0.05))
                                if sorted_err.shape[0] > 2 * trim_n:
                                    trimmed_err = sorted_err[trim_n:-trim_n]
                                else:
                                    trimmed_err = sorted_err
                                bias_correction = trimmed_err.mean(dim=0)
                                b_new = b_orig_dev - bias_correction
                                res_tensors[bias_key] = b_new.to(device="cpu", dtype=original_bias.dtype)
                                del (W_orig_dev, W_dequant_dev, X_calib_dev, b_orig_dev, weight_error, output_error, sorted_err, trimmed_err, bias_correction, b_new)
                                if str(dev).startswith("cuda"):
                                    torch.cuda.empty_cache()
                                break
                            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                                is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
                                if not is_oom:
                                    raise
                                retry_count += 1
                                if retry_count > 10 or current_samples <= min_samples:
                                    warning("  - WARNING: OOM during bias correction even after reducing samples. Giving up.")
                                    raise
                                current_samples = max(min_samples, int(current_samples * 0.7))

                        if isinstance(cache_entry, str):
                            del calib_data
                            gc.collect()
            else:
                res_tensors[bias_key] = loader.get_tensor(bias_key).to("cpu")

        info("-" * 60)

        del original_tensor, q_tensor, dequant_s, dequant_w
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

        res_item = {
            "key": key,
            "base_name": base_name,
            "tensors": res_tensors,
            "lora_tensors": res_lora,
            "meta_entry": meta_entry,
            "skipped": False,
        }
        checkpoint_mgr.save_layer_checkpoint(res_item)
        return res_item

    layer_results = run_parallel_layer_processing(work_items, process_layer_item, target_devices)

    for r in layer_results:
        checkpoint_mgr.save_layer_checkpoint(r)
        if r.get("skipped"):
            skipped_count += 1
        else:
            processed_count += 1

        if "tensors" in r:
            new_tensors.update(r["tensors"])
        if "lora_tensors" in r:
            lora_tensors.update(r["lora_tensors"])
        if "base_name" in r and r.get("meta_entry") and quant_metadata_layers is not None:
            quant_metadata_layers[r["base_name"]] = r["meta_entry"]

    # Copy remaining tensors (norms, etc.)
    passthrough_tensors: Dict[str, torch.Tensor] = {}
    for key in all_keys:
        if any(n in key for n in T5XXL_REMOVE_KEY_NAMES) and filter_flags.get("t5xxl"):
            continue
        if key not in new_tensors and not checkpoint_mgr.is_layer_completed(key):
            passthrough_tensors[key] = loader.get_tensor(key)

    loader.close()
    calibration_data_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    passthrough_tensors.update(new_tensors)
    passthrough_tensors, _ = normalize_tensorwise_scales(passthrough_tensors, NORMALIZE_SCALES_ENABLED)

    success = checkpoint_mgr.assemble_final_output(
        passthrough_tensors=passthrough_tensors,
        original_metadata=original_metadata,
        lora_tensors=lora_tensors,
        lora_save_path=lora_save_path or lora_output,
    )
    if not success:
        error(f"FATAL: Error assembling final output file '{output_file}'")
        if calib_cache_dir and os.path.exists(calib_cache_dir):
            shutil.rmtree(calib_cache_dir)
        return

    if calib_cache_dir and os.path.exists(calib_cache_dir):
        shutil.rmtree(calib_cache_dir)

    info("-" * 60)
    info("Summary:")
    info(f"  - Original tensor count : {len(all_keys)}")
    info(f"  - Weights processed     : {processed_count}")
    info(f"  - Weights skipped       : {skipped_count}")
    info(f"  - Final tensor count    : {len(new_tensors)}")
    info("Conversion complete!")
