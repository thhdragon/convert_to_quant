#!/usr/bin/env python3
"""
Consolidate and Precompute Calibration Activation Stacks.

This tool scans raw calibration activation safetensors files (such as multi-step
diffusion dumps from ComfyUI-PTQ-Sampler), performs high-diversity subsampling
across all steps/files, converts activations to the target precision (default: bfloat16),
and saves a single, unified, high-performance calibration safetensors file.

Usage:
    python -m convert_to_quant.consolidate_calibration -i /path/to/raw_calib_dir -o calib_stack_bf16.safetensors
    python -m convert_to_quant.consolidate_calibration -i /path/to/raw_calib_dir -o calib_stack_bf16.safetensors -n 4096 --dtype bf16 --strategy balanced
"""

import argparse
import glob
import os
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .utils.calibration_loader import normalize_layer_key
from .utils.logging import debug, error, info, minimal, setup_logging, verbose, warning


def parse_dtype(dtype_arg: Union[str, torch.dtype]) -> torch.dtype:
    """Parse string or torch.dtype into standard torch floating-point dtype."""
    if isinstance(dtype_arg, torch.dtype):
        return dtype_arg

    dtype_str = str(dtype_arg).lower().strip()
    if dtype_str in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    elif dtype_str in ("fp16", "float16", "torch.float16", "half"):
        return torch.float16
    elif dtype_str in ("fp32", "float32", "torch.float32", "float"):
        return torch.float32
    else:
        raise ValueError(
            f"Unsupported dtype '{dtype_arg}'. Supported dtypes: 'bfloat16' ('bf16'), 'float16' ('fp16'), 'float32' ('fp32')."
        )


def discover_safetensors_files(input_path: Union[str, List[str]]) -> List[str]:
    """Find all .safetensors files from a directory, file path, glob pattern, or file list."""
    if isinstance(input_path, list):
        files = []
        for p in input_path:
            files.extend(discover_safetensors_files(p))
        return sorted(list(set(files)))

    if os.path.isfile(input_path):
        if input_path.endswith(".safetensors"):
            return [os.path.abspath(input_path)]
        return []

    if os.path.isdir(input_path):
        pattern = os.path.join(input_path, "**", "*.safetensors")
        return sorted(glob.glob(pattern, recursive=True))

    # Pattern / glob matching
    matches = sorted(glob.glob(input_path, recursive=True))
    return [os.path.abspath(m) for m in matches if m.endswith(".safetensors")]


def format_bytes(size_bytes: int) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


def consolidate_calibration_data(
    input_path: Union[str, List[str]],
    output_path: str,
    samples_per_layer: int = 4096,
    dtype: Union[str, torch.dtype] = torch.bfloat16,
    strategy: str = "balanced",
    seed: int = 42,
    layer_filter: Optional[str] = None,
    keep_raw_keys: bool = False,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Consolidate raw calibration activation safetensors files into a unified stack.

    Args:
        input_path: Path to directory, glob pattern, or file list containing raw calibration safetensors.
        output_path: Path to write consolidated .safetensors file.
        samples_per_layer: Number of sample points/tokens to collect per layer (default: 4096).
        dtype: Target data type for output tensors (default: torch.bfloat16).
        strategy: Subsampling strategy ('balanced' extracts ~N/K tokens per file; 'random' pools and subsamples).
        seed: Random seed for reproducible subsampling.
        layer_filter: Optional regex pattern to filter layer keys.
        keep_raw_keys: If True, retains original raw keys instead of normalized keys.
        show_progress: Whether to log detailed progress.

    Returns:
        Dictionary of consolidated activation tensors {layer_key: tensor}.
    """
    target_dtype = parse_dtype(dtype)
    files = discover_safetensors_files(input_path)

    if not files:
        raise FileNotFoundError(f"No .safetensors files found for input path: '{input_path}'")

    total_input_bytes = sum(os.path.getsize(f) for f in files if os.path.exists(f))
    info(f"Discovered {len(files)} calibration files ({format_bytes(total_input_bytes)} total on disk).")

    # Step 1: Index layer keys across all files (metadata only, zero tensor allocation)
    key_to_file_entries: Dict[str, List[Tuple[str, str]]] = {}
    normalized_to_target_key: Dict[str, str] = {}

    filter_re = re.compile(layer_filter) if layer_filter else None

    info("Indexing calibration layers across input files...")
    for fpath in files:
        try:
            with safe_open(fpath, framework="pt") as f:
                for k in f.keys():
                    norm_k = normalize_layer_key(k)
                    target_k = k if keep_raw_keys else norm_k

                    if filter_re and not filter_re.search(target_k) and not filter_re.search(k):
                        continue

                    if target_k not in key_to_file_entries:
                        key_to_file_entries[target_k] = []
                        normalized_to_target_key[norm_k] = target_k

                    key_to_file_entries[target_k].append((fpath, k))
        except Exception as e:
            warning(f"Error inspecting calibration file '{fpath}': {e}")

    unique_layers = sorted(list(key_to_file_entries.keys()))
    info(f"Indexed {len(unique_layers)} unique layer targets across {len(files)} files.")

    if not unique_layers:
        raise ValueError("No matching layer keys found across input files.")

    # Step 2: Stream layer-by-layer extraction
    start_time = time.time()
    consolidated_dict: Dict[str, torch.Tensor] = {}
    gen = torch.Generator().manual_seed(seed)

    info(f"Extracting and consolidating up to {samples_per_layer} samples/layer in {target_dtype} (strategy='{strategy}')...")

    for idx, layer_key in enumerate(unique_layers, 1):
        file_entries = key_to_file_entries[layer_key]
        num_source_files = len(file_entries)

        collected_tensors: List[torch.Tensor] = []

        if strategy == "balanced":
            # Determine target samples per file (distribute uniformly across files)
            base_samples_per_file = max(1, samples_per_layer // num_source_files)
            remainder = samples_per_layer % num_source_files

            for f_idx, (fpath, raw_key) in enumerate(file_entries):
                target_count = base_samples_per_file + (1 if f_idx < remainder else 0)
                try:
                    with safe_open(fpath, framework="pt") as f:
                        if raw_key in f.keys():
                            t = f.get_tensor(raw_key)
                            if t.ndim > 2:
                                t = t.view(-1, t.shape[-1])
                            elif t.ndim == 1:
                                t = t.unsqueeze(0)

                            n_avail = t.shape[0]
                            if n_avail > target_count:
                                file_gen = torch.Generator().manual_seed(seed + f_idx * 1000 + idx)
                                perm = torch.randperm(n_avail, generator=file_gen)[:target_count]
                                t_slice = t[perm]
                            else:
                                t_slice = t

                            collected_tensors.append(t_slice.to(dtype=target_dtype, device="cpu"))
                except Exception as e:
                    debug(f"Failed to read '{raw_key}' from '{fpath}': {e}")
        else:
            # Random pooling strategy
            for f_idx, (fpath, raw_key) in enumerate(file_entries):
                try:
                    with safe_open(fpath, framework="pt") as f:
                        if raw_key in f.keys():
                            t = f.get_tensor(raw_key)
                            if t.ndim > 2:
                                t = t.view(-1, t.shape[-1])
                            elif t.ndim == 1:
                                t = t.unsqueeze(0)
                            collected_tensors.append(t.to(dtype=target_dtype, device="cpu"))
                except Exception as e:
                    debug(f"Failed to read '{raw_key}' from '{fpath}': {e}")

        if not collected_tensors:
            warning(f"No activations could be extracted for layer '{layer_key}', skipping.")
            continue

        combined = torch.cat(collected_tensors, dim=0)

        # Final shuffle and exact slice to samples_per_layer
        total_tokens = combined.shape[0]
        if total_tokens > samples_per_layer:
            layer_gen = torch.Generator().manual_seed(seed + idx * 31)
            perm = torch.randperm(total_tokens, generator=layer_gen)[:samples_per_layer]
            combined = combined[perm]

        consolidated_dict[layer_key] = combined

        if show_progress and (idx == 1 or idx % 10 == 0 or idx == len(unique_layers)):
            verbose(
                f"  [{idx}/{len(unique_layers)}] '{layer_key}': shape={tuple(combined.shape)}, "
                f"from {num_source_files} files"
            )

    # Step 3: Save consolidated safetensors file
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    metadata = {
        "format": "consolidated_ptq_calibration",
        "samples_per_layer": str(samples_per_layer),
        "dtype": str(target_dtype),
        "strategy": strategy,
        "source_files_count": str(len(files)),
        "num_layers": str(len(consolidated_dict)),
        "seed": str(seed),
    }

    info(f"Saving {len(consolidated_dict)} consolidated layers to '{output_path}'...")
    save_file(consolidated_dict, output_path, metadata=metadata)

    elapsed = time.time() - start_time
    out_size_bytes = os.path.getsize(output_path)

    minimal(
        f"\nConsolidation Complete in {elapsed:.2f}s!\n"
        f"  - Input: {len(files)} files ({format_bytes(total_input_bytes)})\n"
        f"  - Output: '{output_path}' ({format_bytes(out_size_bytes)})\n"
        f"  - Layers: {len(consolidated_dict)}\n"
        f"  - Target samples/layer: {samples_per_layer}\n"
        f"  - Precision: {target_dtype}\n"
        f"  - Compression: {total_input_bytes / max(1, out_size_bytes):.1f}x reduction\n"
    )

    return consolidated_dict


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for calibration consolidator."""
    parser = argparse.ArgumentParser(
        description="Precompute and consolidate raw activation calibration safetensors into a unified BF16 stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        "--input-dir",
        "--input_dir",
        type=str,
        required=True,
        dest="input_path",
        help="Path to directory containing raw calibration .safetensors, or glob pattern, or single file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        dest="output_path",
        help="Path for output consolidated .safetensors file.",
    )
    parser.add_argument(
        "-n",
        "--samples",
        "--max-tokens",
        "--max_tokens",
        type=int,
        default=4096,
        dest="samples_per_layer",
        help="Number of calibration samples (tokens) to retain per layer.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        dest="dtype",
        help="Data type for output tensors: 'bfloat16' ('bf16'), 'float16' ('fp16'), 'float32' ('fp32').",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["balanced", "random"],
        default="balanced",
        dest="strategy",
        help="Sampling strategy: 'balanced' extracts evenly across all input files; 'random' pools all tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        dest="seed",
        help="Random seed for reproducible subsampling.",
    )
    parser.add_argument(
        "--layer-filter",
        "--layer_filter",
        type=str,
        default=None,
        dest="layer_filter",
        help="Optional regex pattern to filter layer keys.",
    )
    parser.add_argument(
        "--keep-raw-keys",
        "--keep_raw_keys",
        action="store_true",
        dest="keep_raw_keys",
        help="Keep raw unnormalized layer keys instead of normalizing them.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        dest="quiet",
        help="Suppress all logging except errors.",
    )
    return parser


def main():
    """CLI entry point for calibration consolidation tool."""
    parser = build_parser()
    args = parser.parse_args()

    if args.quiet:
        verb_level = "MINIMAL"
    elif args.verbose:
        verb_level = "VERBOSE"
    else:
        verb_level = "NORMAL"

    setup_logging(verb_level)

    try:
        consolidate_calibration_data(
            input_path=args.input_path,
            output_path=args.output_path,
            samples_per_layer=args.samples_per_layer,
            dtype=args.dtype,
            strategy=args.strategy,
            seed=args.seed,
            layer_filter=args.layer_filter,
            keep_raw_keys=args.keep_raw_keys,
            show_progress=not args.quiet,
        )
    except Exception as e:
        error(f"Consolidation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
