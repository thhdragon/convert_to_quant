"""
CLI main function for convert_to_quant.

Entry point that handles argument parsing and dispatches to INT4 ConvRot W4A4 conversion.
"""

import argparse
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ..config.layer_config import generate_config_template, get_layer_settings, load_layer_config
from ..constants import MODEL_FILTERS
from ..formats.int4_conversion import convert_int4_to_comfy_quant
from ..formats.int4_convrot_conversion import convert_to_int4_convrot
from ..pinned_transfer import set_verbose as set_pinned_verbose
from ..utils.comfy_quant import edit_comfy_quant
from ..utils.logging import setup_logging
from ..utils.parallel_utils import parse_devices
from .argument_parser import (
    ADVANCED_ARGS,
    EXPERIMENTAL_ARGS,
    FILTER_ARGS,
    LEARNED_ROUNDING_ARGS,
    LORA_ARGS,
    MODES_ARGS,
    MultiHelpArgumentParser,
)


def extract_filter_flags(args) -> dict:
    """Extract model filter flags from parsed args with validation."""
    flags = {}
    for name in MODEL_FILTERS.keys():
        if hasattr(args, name) and getattr(args, name):
            flags[name] = True
    return flags


def analyze_dry_run(args) -> None:
    """Report the exact layer routing plan without loading weights or writing output."""
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    custom_pattern = re.compile(args.custom_layers) if args.custom_layers else None
    exclude_pattern = re.compile(args.exclude_layers) if args.exclude_layers else None
    layer_config = load_layer_config(args.layer_config) if args.layer_config else None
    filter_flags = extract_filter_flags(args)

    primary_format = "convrot_w4a4"

    routes = {}
    passthrough_count = 0
    with safe_open(args.input, framework="pt", device="cpu") as checkpoint:
        all_keys = list(checkpoint.keys())
        candidates = []
        for key in all_keys:
            if not key.endswith(".weight"):
                passthrough_count += 1
                continue
            tensor_slice = checkpoint.get_slice(key)
            shape = tuple(tensor_slice.get_shape())
            if len(shape) != 2:
                passthrough_count += 1
                continue
            candidates.append((key, shape))

        print("Dry-run analysis (no conversion will be performed)")
        print(f"Input: {args.input}")
        print(f"Primary format: {primary_format}")
        print(f"2D weight candidates: {len(candidates)}")
        print("-" * 60)

        for index, (key, shape) in enumerate(candidates, start=1):
            route = primary_format
            route_kind = "primary"
            remove_reason = None
            for filter_name, enabled in filter_flags.items():
                if not enabled:
                    continue
                remove_patterns = MODEL_FILTERS[filter_name].get("remove", [])
                if any(pattern in key for pattern in remove_patterns):
                    remove_reason = f"{filter_name} remove"
                    break

            if remove_reason:
                route_kind = "remove"
                route = remove_reason
            else:
                settings = (
                    get_layer_settings(key, layer_config, fullmatch=args.layer_config_fullmatch)
                    if layer_config
                    else None
                )
                if settings:
                    if settings.get("skip", False):
                        route_kind = "skip"
                        route = "layer-config skip"
                    else:
                        route_kind = "config"
                        route = str(settings["format"])
                elif custom_pattern and custom_pattern.search(key):
                    route_kind = "custom"
                    route = "convrot_w4a4"
                else:
                    exclusion_reason = None
                    if exclude_pattern and exclude_pattern.search(key):
                        exclusion_reason = "--exclude-layers"
                    if exclusion_reason is None:
                        for filter_name, enabled in filter_flags.items():
                            if not enabled:
                                continue
                            config = MODEL_FILTERS[filter_name]
                            patterns = config.get("exclude", []) + config.get("highprec", [])
                            if any(pattern in key for pattern in patterns):
                                exclusion_reason = f"{filter_name} filter"
                                break
                    if exclusion_reason:
                        route_kind = "skip"
                        route = exclusion_reason

            route_label = f"{route_kind}:{route}"
            routes[route_label] = routes.get(route_label, 0) + 1
            print(f"[{index}/{len(candidates)}] {key} {list(shape)} -> {route_label}")

    print("-" * 60)
    print("Dry-run summary:")
    for route_label, count in sorted(routes.items()):
        print(f"  {route_label}: {count}")
    print(f"  passthrough tensors: {passthrough_count}")
    print("No output file was written.")


def get_parser() -> MultiHelpArgumentParser:
    """Create and return the argument parser for INT4 ConvRot W4A4 conversion."""
    parser = MultiHelpArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Convert safetensors weights to INT4 ConvRot W4A4 format.\n\n"
        "Default behavior: INT4 W4A4 quantization with group-wise Hadamard rotation (ConvRot).\n"
        "For model-specific layer exclusions, see --help-filters.\n"
        "For advanced LR tuning and early stopping, see --help-advanced.\n"
        "For utility modes, see --help-modes.",
        experimental_args=EXPERIMENTAL_ARGS,
        filter_args=FILTER_ARGS,
        advanced_args=ADVANCED_ARGS,
        learned_rounding_args=LEARNED_ROUNDING_ARGS,
        modes_args=MODES_ARGS,
        lora_args=LORA_ARGS,
    )

    parser.add_argument("-i", "--input", type=str, required=True, help="Input safetensors file path.")
    parser.add_argument("-o", "--output", type=str, help="Output safetensors file path. Auto-generated if not provided.")
    parser.add_argument("--resume", action="store_true", help="Resume quantization from existing sidecar progress file if present.")
    parser.add_argument("--sidecar-path", "--sidecar_path", type=str, default=None, dest="sidecar_path", help="Custom path for the sidecar progress JSON file.")
    parser.add_argument("--max-shard-size", "--max_shard_size", type=str, default=None, dest="max_shard_size", help="Maximum size per output safetensors shard (e.g. '5GB').")
    parser.add_argument("--no-checkpoint", "--no_checkpoint", action="store_true", dest="no_checkpoint", help="Disable sidecar progress tracking and per-layer checkpoint saving.")
    parser.add_argument("--comfy_quant", "--comfy-quant", action="store_true", default=True, dest="comfy_quant", help="Use Comfy quantization format (enabled by default).")
    parser.add_argument("-4", "--int4", "--w4a4", action="store_true", default=True, dest="int4", help="Use INT4 W4A4 ConvRot quantization (default).")
    parser.add_argument("--convrot", action="store_true", default=True, help="Enable group-wise Hadamard rotation (ConvRot) for INT4 row-wise quantization.")
    parser.add_argument("--convrot-group-size", "--convrot_group_size", type=int, default=256, dest="convrot_group_size", help="Group size for ConvRot (default: 256).")
    parser.add_argument("--dynamic-convrot", "--dynamic_convrot", action="store_true", dest="dynamic_convrot", help="Enable dynamic ConvRot group sizing.")
    parser.add_argument("--w4a4-untouched-activations", "--w4a4_untouched_activations", action="store_true", dest="w4a4_untouched_activations", help="Leave calibration activations untouched during weight optimization.")
    parser.add_argument("--custom-layers", "--custom_layers", type=str, default=None, dest="custom_layers", help="Regex pattern for layers to quantize with custom options.")
    parser.add_argument("--exclude-layers", "--exclude_layers", type=str, default=None, dest="exclude_layers", help="Regex pattern for layers to exclude from quantization.")
    parser.add_argument("--custom-block-size", "--custom_block_size", type=int, default=None, dest="custom_block_size", help="Block size for custom layers.")
    parser.add_argument("--custom-simple", "--custom_simple", action="store_true", dest="custom_simple", help="Use simple quantization for custom layers.")
    parser.add_argument("--custom-heur", "--custom_heur", action="store_true", dest="custom_heur", help="Apply performance heuristics to custom layers.")
    parser.add_argument("--custom-full-precision-mm", "--custom-fpmm", action="store_true", dest="custom_full_precision_mm", help="Enable full_precision_matrix_mult=True in .comfy_quant metadata for custom layers.")
    parser.add_argument("--custom-convrot", action="store_true", dest="custom_convrot", help="Enable group-wise Hadamard rotation for custom layers.")
    parser.add_argument("--custom-convrot-group-size", type=int, default=256, dest="custom_convrot_group_size", help="Group size for custom layer ConvRot (default: 256).")
    parser.add_argument("--simple", action="store_true", help="Skip SVD optimization, use simple quantization.")
    parser.add_argument("--full_precision_matrix_mult", "--full-precision-matrix-mult", "-fpmm", action="store_true", dest="full_precision_matrix_mult", help="Add full_precision_matrix_mult=True to .comfy_quant metadata.")
    parser.add_argument("--heur", action="store_true", help="Skip layers with poor quantization characteristics.")
    parser.add_argument("--device", type=str, default=None, help="Device to use for quantization (e.g., 'cpu', 'cuda').")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated list of devices for multi-GPU parallel quantization.")
    parser.add_argument("--num_gpus", "--num-gpus", type=int, default=None, dest="num_gpus", help="Number of GPUs to use.")
    parser.add_argument("--verbose", type=str, default="NORMAL", choices=["DEBUG", "VERBOSE", "NORMAL", "MINIMAL"], help="Set verbosity.")

    for filter_name, filter_cfg in MODEL_FILTERS.items():
        parser.add_argument(f"--{filter_name}", action="store_true", help=filter_cfg.get("help", f"Apply {filter_name} model exclusions"))

    parser.add_argument("--full_matrix", "--full-matrix", action="store_true", dest="full_matrix", help="Use full matrices for SVD.")
    parser.add_argument("--scaling_mode", "--scaling-mode", type=str, default="row", dest="scaling_mode", choices=["row", "block"], help="Quantization scaling mode (default: row).")
    parser.add_argument("--block_size", "--block-size", "--group_size", "--group-size", type=int, default=64, dest="block_size", help="Block/group size (default: 64).")
    parser.add_argument("--calib_samples", "--calib-samples", type=int, default=3072, dest="calib_samples", help="Number of random samples for calibration.")
    parser.add_argument("--calib_cpu", "--calib-cpu", action="store_true", dest="calib_cpu", help="Store calibration data cache on CPU.")
    parser.add_argument("--manual_seed", "--manual-seed", type=int, default=-1, dest="manual_seed", help="Manual seed for reproducibility.")
    parser.add_argument("--optimizer", type=str, default="prodigy", choices=["original", "adamw", "radam", "prodigy"], help="Optimization algorithm.")
    parser.add_argument("--num_iter", "--num-iter", type=int, default=4000, dest="num_iter", help="Total optimization iterations per tensor.")
    parser.add_argument("--lr", type=float, default=1.0, help="Initial learning rate.")
    parser.add_argument("--use_speed", "--use-speed", action="store_true", dest="use_speed", help="Enable use_speed parameter for Prodigy optimizer.")
    parser.add_argument("--lr_schedule", "--lr-schedule", type=str, default="plateau", dest="lr_schedule", choices=["adaptive", "exponential", "plateau"], help="LR schedule.")
    parser.add_argument("--lr_gamma", "--lr-gamma", type=float, default=0.99, dest="lr_gamma", help="Decay factor per step for exponential schedule.")
    parser.add_argument("--lr_patience", "--lr-patience", type=int, default=1, dest="lr_patience", help="Steps before decay for plateau schedule.")
    parser.add_argument("--lr_factor", "--lr-factor", type=float, default=0.95, dest="lr_factor", help="LR reduction factor.")
    parser.add_argument("--lr_min", "--lr-min", type=float, default=1e-8, dest="lr_min", help="Minimum LR bound.")
    parser.add_argument("--lr_cooldown", "--lr-cooldown", type=int, default=0, dest="lr_cooldown", help="Cooldown steps after LR reduction.")
    parser.add_argument("--lr_threshold", "--lr-threshold", type=float, default=0.0, dest="lr_threshold", help="Min improvement to reset patience.")
    parser.add_argument("--lr_adaptive_mode", "--lr-adaptive-mode", type=str, default="simple-reset", dest="lr_adaptive_mode", choices=["simple-reset", "no-reset"], help="Counter reset behavior.")
    parser.add_argument("--lr-threshold-mode", "--lr_threshold_mode", type=str, default="rel", choices=["rel", "abs"], dest="lr_threshold_mode", help="Threshold interpretation mode.")
    parser.add_argument("--lr-shape-influence", "--lr_shape_influence", type=float, default=1.0, dest="lr_shape_influence", help="Scale factor based on tensor aspect ratio.")
    parser.add_argument("--early-stop-loss", "--early_stop_loss", "-esloss", type=float, default=5e-9, dest="early_stop_loss", help="Early stop loss threshold.")
    parser.add_argument("--early-stop-lr", "--early_stop_lr", "-eslr", type=float, default=1.01e-8, dest="early_stop_lr", help="Early stop LR threshold.")
    parser.add_argument("--early-stop-stall", "--early_stop_stall", "-esstall", type=int, default=2000, dest="early_stop_stall", help="Early stop stall count threshold.")
    parser.add_argument("--scale-optimization", "--scale_optimization", type=str, default="fixed", dest="scale_optimization", choices=["fixed", "iterative", "joint", "dualround"], help="Scale optimization mode.")
    parser.add_argument("--top_p", "--top-p", type=float, default=0.2, dest="top_p", help="Proportion of principal components to use.")
    parser.add_argument("--min_k", "--min-k", type=int, default=256, dest="min_k", help="Minimum number of principal components.")
    parser.add_argument("--max_k", "--max-k", type=int, default=1280, dest="max_k", help="Maximum number of principal components.")
    parser.add_argument("--extract-lora", "--extract_lora", action="store_true", dest="extract_lora", help="Extract quantization error into separate LoRA adapter layers.")
    parser.add_argument("--lora-rank", "--lora_rank", type=int, default=32, dest="lora_rank", help="Rank for extracted LoRA layers.")
    parser.add_argument("--lora-target", "--lora_target", type=str, default=None, dest="lora_target", help="Regex pattern for LoRA target layers.")
    parser.add_argument("--lora-depth", "--lora_depth", type=int, default=1, dest="lora_depth", help="Maximum block depth for LoRA extraction.")
    parser.add_argument("--lora-ar-threshold", "--lora_ar_threshold", type=float, default=0.0, dest="lora_ar_threshold", help="Aspect ratio threshold for LoRA extraction.")
    parser.add_argument("--lora-output", "--lora_output", type=str, default=None, dest="lora_output", help="Path to save extracted LoRA adapter.")
    parser.add_argument("--convert-int4", "--convert-int4-to-comfy-quant", action="store_true", dest="convert_int4", help="Convert INT4 model to comfy_quant format.")
    parser.add_argument("--layer_config", "--layer-config", type=str, default=None, dest="layer_config", help="Path to layer configuration JSON file.")
    parser.add_argument("--layer_config_fullmatch", "--layer-config-fullmatch", action="store_true", dest="layer_config_fullmatch", help="Use full matching for layer config regex.")
    parser.add_argument("--save_quant_metadata", "--save-quant-metadata", action="store_true", default=True, dest="save_quant_metadata", help="Save quantization metadata in header.")
    parser.add_argument("--low-memory", "--low_memory", action="store_true", dest="low_memory", help="Use memory-efficient loading.")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", dest="dry_run", help="Report layer routing plan without converting.")
    parser.add_argument("--actcal_lora", "--actcal-lora", type=str, default=None, dest="actcal_lora", help="Path to LoRA file for calibration.")

    return parser


def run_conversion(args) -> None:
    """Execute conversion based on parsed arguments."""
    setup_logging(args.verbose)
    if hasattr(args, "verbose"):
        set_pinned_verbose(args.verbose in ("VERBOSE", "DEBUG"))

    if args.dry_run:
        analyze_dry_run(args)
        return

    if getattr(args, "convert_int4", False):
        output_file = args.output
        if not output_file:
            base = os.path.splitext(args.input)[0]
            output_file = f"{base}_comfy_quant.safetensors"
        convert_int4_to_comfy_quant(
            args.input,
            output_file,
            block_size=args.block_size or 64,
            convrot_group_size=args.convrot_group_size or 256,
            save_quant_metadata=args.save_quant_metadata,
        )
        return

    target_devices = parse_devices(device=args.device, devices=getattr(args, "devices", None), num_gpus=getattr(args, "num_gpus", None))

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    if not args.output:
        base = os.path.splitext(args.input)[0]
        prefix = "simple_" if args.simple else "learned_"
        output_file = f"{base}_{prefix}convrot_w4a4.safetensors"
    else:
        output_file = args.output

    filter_flags = extract_filter_flags(args)

    if os.path.abspath(args.input) == os.path.abspath(output_file):
        print("Error: Output file cannot be same as input.")
        return

    seed = int(torch.randint(0, 2**32 - 1, ()).item()) if args.manual_seed == -1 else args.manual_seed
    print(f"Using seed: {seed}")

    layer_config_data = load_layer_config(args.layer_config) if args.layer_config else None

    convert_to_int4_convrot(
        args.input,
        output_file,
        comfy_quant=args.comfy_quant,
        filter_flags=filter_flags,
        calib_samples=args.calib_samples,
        calib_cpu=args.calib_cpu,
        seed=seed,
        custom_layers=args.custom_layers,
        exclude_layers=args.exclude_layers,
        block_size=args.block_size,
        convrot_group_size=args.convrot_group_size,
        dynamic_convrot=args.dynamic_convrot,
        w4a4_untouched_activations=getattr(args, "w4a4_untouched_activations", False),
        full_precision_matrix_mult=args.full_precision_matrix_mult,
        custom_full_precision_mm=getattr(args, "custom_full_precision_mm", False),
        skip_inefficient_layers=args.heur,
        no_learned_rounding=args.simple,
        layer_config=layer_config_data,
        layer_config_fullmatch=args.layer_config_fullmatch,
        save_quant_metadata=args.save_quant_metadata,
        low_memory=args.low_memory,
        device=args.device,
        devices=target_devices,
        optimizer=args.optimizer,
        num_iter=args.num_iter,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        top_p=args.top_p,
        min_k=args.min_k,
        max_k=args.max_k,
        full_matrix=args.full_matrix,
        scaling_mode=args.scaling_mode,
        lr_gamma=args.lr_gamma,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
        lr_cooldown=args.lr_cooldown,
        lr_threshold=args.lr_threshold,
        lr_adaptive_mode=args.lr_adaptive_mode,
        lr_shape_influence=args.lr_shape_influence,
        lr_threshold_mode=args.lr_threshold_mode,
        early_stop_loss=args.early_stop_loss,
        early_stop_lr=args.early_stop_lr,
        early_stop_stall=args.early_stop_stall,
        scale_optimization=args.scale_optimization,
        use_speed=getattr(args, "use_speed", False),
        extract_lora=args.extract_lora,
        lora_rank=args.lora_rank,
        lora_target=args.lora_target,
        lora_depth=args.lora_depth,
        lora_ar_threshold=args.lora_ar_threshold,
        lora_output=args.lora_output,
        actcal_lora=args.actcal_lora,
        resume=args.resume,
        sidecar_path=args.sidecar_path,
        max_shard_size=args.max_shard_size,
        no_checkpoint=args.no_checkpoint,
    )


def main():
    parser = get_parser()
    args = parser.parse_args()
    run_conversion(args)


if __name__ == "__main__":
    main()
