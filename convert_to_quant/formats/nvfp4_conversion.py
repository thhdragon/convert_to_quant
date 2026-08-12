"""
NVFP4 conversion functions for convert_to_quant.

Converts safetensors models to NVFP4 (FP4 E2M1) quantized format with
per-tensor + per-block scaling for Blackwell GPU inference.

Uses LearnedNVFP4Converter (SVD optimization) by default.
Use --simple to switch to raw NVFP4Converter.
"""

import gc
import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from safetensors.torch import save_file

from ..constants import (
    AVOID_KEY_NAMES,
    COMPUTE_DTYPE,
    FP4_BLOCK_SIZE,
    MODEL_FILTERS,
    NORMALIZE_SCALES_ENABLED,
)
from ..converters.learned_nvfp4 import LearnedNVFP4Converter
from ..converters.nvfp4_converter import NVFP4Converter
from ..utils.comfy_quant import (
    should_skip_layer_for_performance,
)
from ..utils.logging import (
    error,
    info,
    log_debug,
    minimal,
    verbose,
    warning,
)
from ..utils.memory_efficient_loader import (
    UnifiedSafetensorsLoader,
)
from ..utils.tensor_utils import (
    dict_to_tensor,
    normalize_tensorwise_scales,
)


@log_debug
def convert_to_nvfp4(
    input_file: str,
    output_file: str,
    # Filter flags (validated dict from CLI)
    filter_flags: Dict[str, bool] = None,
    exclude_layers: Optional[str] = None,
    # Quantization options
    simple: bool = False,
    num_iter: int = 2000,
    heur: bool = False,
    verbose_output: bool = True,  # kept for API compat, actual logging uses logging.py
    # Calibration options (for bias correction)
    calib_samples: int = 3072,
    seed: int = 42,
    # Optimizer/LR options (passed to LearnedNVFP4Converter)
    optimizer: str = "prodigy",
    lr: float = 1.0,
    lr_schedule: str = "plateau",
    top_p: float = 0.2,
    min_k: int = 128,
    max_k: int = 1280,
    full_matrix: bool = False,
    # LR schedule tuning
    lr_gamma: float = 0.99,
    lr_patience: int = 1,
    lr_factor: float = 0.95,
    lr_min: float = 1e-8,
    lr_cooldown: int = 0,
    lr_threshold: float = 0.0,
    lr_adaptive_mode: str = "simple-reset",
    lr_shape_influence: float = 1.0,
    lr_threshold_mode: str = "rel",
    # Early stopping
    early_stop_loss: float = 5e-9,
    early_stop_lr: float = 1.01e-8,
    early_stop_stall: int = 2000,
    # Scale optimization
    scale_refinement_rounds: int = 1,
    scale_optimization: str = "fixed",
    # Input scales (optional, from calibration or another NVFP4 model)
    input_scales: Optional[dict] = None,
    # Memory mode
    low_memory: bool = False,
    # Prodigy
    use_speed: bool = False,
    # LoRA extraction options
    extract_lora: bool = False,
    lora_rank: int = 16,
    lora_target: Optional[str] = None,
    lora_depth: int = -1,
    lora_ar_threshold: float = 0.0,
    lora_save_path: Optional[str] = None,
    lora_output: Optional[str] = None,
    # Device options
    device: Optional[str] = None,
    devices: Optional[Union[str, List[str]]] = None,
    # Checkpointing options
    resume: bool = False,
    sidecar_path: Optional[str] = None,
    max_shard_size: Optional[Union[str, int]] = None,
    no_checkpoint: bool = False,
) -> None:
    """
    Convert safetensors model to NVFP4 (FP4 E2M1) quantized format.

    Uses LearnedNVFP4Converter with SVD optimization by default.
    Pass simple=True for raw quantization without optimization.

    Always creates .comfy_quant metadata tensors and _quantization_metadata header.
    """
    info(f"Processing: {input_file}\nOutput will be saved to: {output_file}")
    info("-" * 60)
    info("Target format: NVFP4 (FP4 E2M1 block quantization)")
    info(f"Block size: {FP4_BLOCK_SIZE}")
    info("-" * 60)

    from ..utils.checkpoint import QuantCheckpointManager
    from ..utils.parallel_utils import parse_devices, run_parallel_layer_processing

    target_devices = parse_devices(device=device, devices=devices)
    checkpoint_mgr = QuantCheckpointManager(
        output_file=output_file,
        input_file=input_file,
        primary_format="nvfp4",
        resume=resume,
        sidecar_path=sidecar_path,
        max_shard_size=max_shard_size,
        no_checkpoint=no_checkpoint,
    )
    seed_device = "cpu"
    seed_generator = torch.Generator(device=seed_device)
    seed_generator.manual_seed(seed)

    # Build exclusion list from filter flags using MODEL_FILTERS registry
    exclude_patterns = list(AVOID_KEY_NAMES)  # Base exclusions

    # Use filter_flags dict passed from CLI (or empty if not provided)
    active_filters = filter_flags or {}

    # Add patterns from active filters
    for filter_name, is_active in active_filters.items():
        if not is_active:
            continue
        cfg = MODEL_FILTERS[filter_name]
        exclude_patterns.extend(cfg.get("exclude", []))
        exclude_patterns.extend(cfg.get("highprec", []))

    # Compile --exclude-layers regex pattern
    exclude_regex_pattern = None
    if exclude_layers:
        import re

        try:
            exclude_regex_pattern = re.compile(exclude_layers)
            info(f"Layer exclusion enabled: pattern '{exclude_layers}'")
        except re.error as e:
            error(f"ERROR: Invalid regex pattern '{exclude_layers}': {e}")
            return

    def create_converter_for_device(dev: str):
        if simple:
            return NVFP4Converter(block_size=FP4_BLOCK_SIZE, pad_to_16x=True)
        else:
            return LearnedNVFP4Converter(
                optimizer=optimizer,
                num_iter=num_iter,
                top_p=top_p,
                min_k=min_k,
                max_k=max_k,
                block_size=FP4_BLOCK_SIZE,
                pad_to_16x=True,
                full_matrix=full_matrix,
                no_learned_rounding=False,
                lr_schedule=lr_schedule,
                lr_gamma=lr_gamma,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                lr_min=lr_min,
                lr_cooldown=lr_cooldown,
                lr_threshold=lr_threshold,
                lr_adaptive_mode=lr_adaptive_mode,
                lr_shape_influence=lr_shape_influence,
                lr_threshold_mode=lr_threshold_mode,
                early_stop_loss=early_stop_loss,
                early_stop_lr=early_stop_lr,
                early_stop_stall=early_stop_stall,
                scale_refinement_rounds=scale_refinement_rounds,
                scale_optimization=scale_optimization,
                lr=lr,
                use_speed=use_speed,
                extract_lora=extract_lora,
                lora_rank=lora_rank,
                lora_target=lora_target,
                lora_depth=lora_depth,
                lora_ar_threshold=lora_ar_threshold,
                device=dev,
            )

    output_tensors: Dict[str, torch.Tensor] = {}
    lora_tensors: Dict[str, torch.Tensor] = {}
    quant_metadata = {}
    quantized_count = 0
    skipped_count = 0

    # Load tensors using unified loader (handles both standard and low-memory modes)
    try:
        loader = UnifiedSafetensorsLoader(input_file, low_memory=low_memory)
    except Exception as e:
        error(f"FATAL: Error loading '{input_file}': {e}")
        return

    all_keys = loader.keys()
    original_metadata = loader.metadata()

    # Filter to only weight tensors for quantization
    weight_keys = sorted([k for k in all_keys if k.endswith(".weight") and loader.get_ndim(k) == 2])
    total_weights = len(weight_keys)

    # Generate calibration data for bias correction
    calibration_data_cache = {}
    minimal("Scanning model and generating simulated calibration data...")
    for key in weight_keys:
        shape = loader.get_shape(key)
        if len(shape) == 2:
            in_features = shape[1]
            if in_features not in calibration_data_cache:
                verbose(f"  - Found new input dimension: {in_features}.")
                calibration_data_cache[in_features] = torch.randn(
                    calib_samples, in_features, dtype=COMPUTE_DTYPE, generator=seed_generator, device=seed_device
                )
    info("Simulated calibration data generated.\n")

    info(f"Found {total_weights} weight tensors to potentially process.")
    info("-" * 60)

    work_items = list(enumerate(weight_keys))

    def process_layer_item(item: Tuple[int, str], dev: str) -> Dict[str, Any]:
        i, key = item

        if checkpoint_mgr.is_layer_completed(key):
            info(f"[{dev}] ({i + 1}/{total_weights}) Skipping (loaded from sidecar checkpoint): {key}")
            loaded_res = checkpoint_mgr.load_completed_layer(key)
            if loaded_res is not None:
                return loaded_res

        tensor = loader.get_tensor(key)
        base_key = key.rsplit(".weight", 1)[0]
        exclusion_reason = ""

        if any(pattern in key for pattern in exclude_patterns):
            exclusion_reason = "Exclusion pattern match"

        if not exclusion_reason and exclude_regex_pattern and exclude_regex_pattern.search(key):
            exclusion_reason = "regex exclusion (--exclude-layers)"

        if tensor.dim() != 2:
            info(f"[{dev}] ({i + 1}/{total_weights}) Skipping tensor: {key} (Reason: non-2D tensor)")
            return {"key": key, "skipped": True, "tensors": {key: tensor.to("cpu")}}

        if exclusion_reason:
            info(f"[{dev}] ({i + 1}/{total_weights}) Skipping tensor: {key} (Reason: {exclusion_reason})")
            return {"key": key, "skipped": True, "tensors": {key: tensor.to("cpu")}}

        if heur:
            should_skip, skip_reason = should_skip_layer_for_performance(tensor, FP4_BLOCK_SIZE)
            if should_skip:
                info(f"[{dev}] ({i + 1}/{total_weights}) Skipping tensor: {key} (Reason: {skip_reason})")
                return {"key": key, "skipped": True, "tensors": {key: tensor.to("cpu")}}

        info(f"[{dev}] ({i + 1}/{total_weights}) Processing tensor: {key}")

        depth = -1
        import re

        depth_match = re.search(r"\.(\d+)\.", key)
        if depth_match:
            depth = int(depth_match.group(1))

        bias_key = f"{base_key}.bias"
        has_bias = bias_key in all_keys

        converter_inst = create_converter_for_device(dev)
        if simple:
            tensor_gpu = tensor.to(device=dev, dtype=torch.float32)
            qdata, block_scales, per_tensor_scale = converter_inst.quantize(tensor_gpu)
            if has_bias:
                dequant_w = converter_inst.dequantize(qdata, per_tensor_scale, block_scales, output_dtype=torch.float32)
                if dequant_w.shape != tensor.shape:
                    dequant_w = dequant_w[:tensor.shape[0], :tensor.shape[1]]
            else:
                dequant_w = None
            del tensor_gpu
            extra_tensors = {}
        else:
            qdata, block_scales, per_tensor_scale, dequant_w, extra_tensors = converter_inst.convert(
                tensor, key=key, depth=depth, has_bias=has_bias
            )
            if dequant_w is not None and dequant_w.shape != tensor.shape:
                dequant_w = dequant_w[:tensor.shape[0], :tensor.shape[1]]

        res_tensors = {}
        res_lora = {}

        if extra_tensors:
            for lora_key, lora_tensor in extra_tensors.items():
                if lora_key in ("lora_up", "lora_down"):
                    if base_key.startswith("diffusion_model.") or base_key.startswith("text_encoders."):
                        full_lora_key = f"{base_key}.{lora_key}.weight"
                    else:
                        full_lora_key = f"diffusion_model.{base_key}.{lora_key}.weight"
                    res_lora[full_lora_key] = lora_tensor.cpu()

        res_tensors[key] = qdata.cpu()
        res_tensors[f"{base_key}.weight_scale_2"] = per_tensor_scale.cpu().to(torch.float32)
        res_tensors[f"{base_key}.weight_scale"] = block_scales.cpu()

        if input_scales and base_key in input_scales:
            res_tensors[f"{base_key}.input_scale"] = torch.tensor(input_scales[base_key], dtype=torch.float32)

        if has_bias and bias_key in all_keys:
            verbose(f"  - Adjusting corresponding bias: {bias_key}")
            with torch.no_grad():
                original_bias = loader.get_tensor(bias_key)
                in_features = tensor.shape[1]
                if in_features not in calibration_data_cache:
                    warning("  - WARNING: No calibration data for bias correction.")
                    res_tensors[bias_key] = original_bias.to("cpu")
                else:
                    X_calib_dev = calibration_data_cache[in_features].to(device=dev)
                    W_orig_dev = tensor.to(device=dev, dtype=COMPUTE_DTYPE)
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

        meta = {
            "format": "nvfp4",
            "group_size": FP4_BLOCK_SIZE,
            "orig_dtype": str(tensor.dtype),
            "orig_shape": list(tensor.shape),
        }
        res_tensors[f"{base_key}.comfy_quant"] = dict_to_tensor(meta)

        info(f"    - Final Weight shape      : {list(qdata.shape)}")
        info(f"    - Final Block Scale shape : {list(block_scales.shape)}")
        info("-" * 60)

        del tensor, dequant_w
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

        res_item = {
            "key": key,
            "base_key": base_key,
            "tensors": res_tensors,
            "lora_tensors": res_lora,
            "meta_entry": meta,
            "metadata": meta,
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
            quantized_count += 1

        if "tensors" in r:
            output_tensors.update(r["tensors"])
        if "lora_tensors" in r:
            lora_tensors.update(r["lora_tensors"])
        if "base_key" in r and "metadata" in r:
            quant_metadata[r["base_key"]] = r["metadata"]

    # Copy non-weight tensors (bias handled above, copy others)
    passthrough_tensors: Dict[str, torch.Tensor] = {}
    for key in all_keys:
        if key not in output_tensors and not checkpoint_mgr.is_layer_completed(key):
            passthrough_tensors[key] = loader.get_tensor(key)

    # Close loader
    loader.close()

    passthrough_tensors.update(output_tensors)
    if NORMALIZE_SCALES_ENABLED:
        passthrough_tensors, normalized_count = normalize_tensorwise_scales(passthrough_tensors)

    checkpoint_mgr.assemble_final_output(
        passthrough_tensors=passthrough_tensors,
        original_metadata=original_metadata,
        lora_tensors=lora_tensors,
        lora_save_path=lora_save_path or lora_output,
    )

    info("-" * 60)
    info("Summary:")
    info(f"  - Original tensor count : {len(all_keys)}")
    info(f"  - Weights processed     : {quantized_count}")
    info(f"  - Weights skipped       : {skipped_count}")
    info(f"  - Final tensor count    : {len(output_tensors)}")
    info("-" * 60)
    info("Conversion complete!")
