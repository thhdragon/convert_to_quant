#!/usr/bin/env python3
"""
convert_wan_checkpoint.py

Detects whether a Wan video-diffusion-transformer checkpoint (single
.safetensors file OR a sharded checkpoint described by a
*.safetensors.index.json) is in "native" (original repo) format or
"diffusers" format, and converts it to the other format.

Native key examples:      blocks.0.self_attn.q.weight, head.head.weight
Diffusers key examples:   blocks.0.attn1.to_q.weight,  proj_out.weight

Usage:
    # Just detect the format, don't convert
    python convert_wan_checkpoint.py detect /path/to/checkpoint

    # Convert (auto-detects source format, writes the opposite)
    python convert_wan_checkpoint.py convert /path/to/checkpoint /path/to/output_dir

    # Force a direction instead of auto-detecting
    python convert_wan_checkpoint.py convert /path/to/checkpoint /path/to/output_dir --to diffusers
    python convert_wan_checkpoint.py convert /path/to/checkpoint /path/to/output_dir --to native

`/path/to/checkpoint` can be either:
    - a single *.safetensors file, or
    - a *.safetensors.index.json file (sharded checkpoint;
      the shard files must sit alongside it), or
    - a directory containing either of the above (the script will look
      for a *.index.json first, then a single *.safetensors file).

Output is always written as a single, unsharded model.safetensors file
(plus a matching model.safetensors.index.json only if you pass
--keep-shards, in which case it re-shards using the same shard sizes as
the input). For most fine-tuning / inference use a single merged file is
simplest, so that's the default.

Requires: pip install safetensors
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
except ImportError:
    sys.exit(
        "This script needs the `safetensors` and `torch` packages.\n"
        "Install with:  pip install safetensors torch"
    )


# --------------------------------------------------------------------------
# Key-mapping table (native <-> diffusers), derived from the Wan
# transformer block structure.
# --------------------------------------------------------------------------

BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")

# Sub-mappings applied *within* a block, native -> diffusers.
NATIVE_TO_DIFFUSERS_BLOCK = [
    (re.compile(r"^self_attn\.(.+)$"), r"attn1.\1"),
    (re.compile(r"^cross_attn\.(.+)$"), r"attn2.\1"),
    (re.compile(r"^ffn\.0\.(.+)$"), r"ffn.net.0.proj.\1"),
    (re.compile(r"^ffn\.2\.(.+)$"), r"ffn.net.2.\1"),
    (re.compile(r"^modulation$"), r"scale_shift_table"),
    (re.compile(r"^norm3\.(.+)$"), r"norm2.\1"),
]

DIFFUSERS_TO_NATIVE_BLOCK = [
    (re.compile(r"^attn1\.(.+)$"), r"self_attn.\1"),
    (re.compile(r"^attn2\.(.+)$"), r"cross_attn.\1"),
    (re.compile(r"^ffn\.net\.0\.proj\.(.+)$"), r"ffn.0.\1"),
    (re.compile(r"^ffn\.net\.2\.(.+)$"), r"ffn.2.\1"),
    (re.compile(r"^scale_shift_table$"), r"modulation"),
    (re.compile(r"^norm2\.(.+)$"), r"norm3.\1"),
]

# Top-level (non-block) renames, native -> diffusers.
NATIVE_TO_DIFFUSERS_TOP = {
    "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
    "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
    "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
    "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
    "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
    "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
    "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
    "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
    "time_projection.1.bias": "condition_embedder.time_proj.bias",
    "time_projection.1.weight": "condition_embedder.time_proj.weight",
    "head.head.bias": "proj_out.bias",
    "head.head.weight": "proj_out.weight",
    "head.modulation": "scale_shift_table",
    # patch_embedding.* is unchanged in both formats.
}

DIFFUSERS_TO_NATIVE_TOP = {v: k for k, v in NATIVE_TO_DIFFUSERS_TOP.items()}

# Distinctive marker substrings used purely for format detection.
NATIVE_MARKERS = ("self_attn", "cross_attn", "head.head", "time_projection")
DIFFUSERS_MARKERS = ("attn1", "attn2", "proj_out", "condition_embedder")

# Substrings that training wrappers (FSDP, DDP, torch.compile, etc.) commonly
# splice into every key. These carry no architectural meaning, so we strip
# them before mapping and can optionally restore them afterward.
WRAPPER_NOISE_SUBSTRINGS = (
    "_fsdp_wrapped_module.",
    "_orig_mod.",       # torch.compile
    "module.",          # DDP / DataParallel -- only stripped as a leading token, see below
)


def strip_wrapper_noise(key):
    """
    Remove known training-wrapper substrings (FSDP/compile/DDP) from a key,
    which can appear multiple times since these wrappers nest recursively
    per submodule. Returns (clean_key, list_of_removed_fragments_in_order)
    so the exact original key can be reconstructed if needed.
    """
    removed = []
    remaining = key
    changed = True
    while changed:
        changed = False
        for noise in ("_fsdp_wrapped_module.", "_orig_mod."):
            if noise in remaining:
                remaining = remaining.replace(noise, "", 1)
                removed.append(noise)
                changed = True
    return remaining, removed


def strip_known_leading_prefix(key):
    """
    Some checkpoints (this included) prefix every key with a wrapper attribute
    name like 'model.' before the real architecture keys start. Detect and
    strip a single leading 'model.' segment if present.
    """
    if key.startswith("model."):
        return key[len("model."):], "model."
    return key, None


# --------------------------------------------------------------------------
# Loading (handles single-file and sharded checkpoints)
# --------------------------------------------------------------------------

def find_checkpoint_files(path: Path):
    """
    Given a path that may be a directory, a .safetensors file, or a
    .safetensors.index.json file, return (index_json_path_or_None, [shard_paths]).
    """
    if path.is_dir():
        index_candidates = sorted(path.glob("*.safetensors.index.json"))
        if index_candidates:
            return find_checkpoint_files(index_candidates[0])
        st_candidates = sorted(path.glob("*.safetensors"))
        if not st_candidates:
            sys.exit(f"No .safetensors or .safetensors.index.json found in {path}")
        if len(st_candidates) > 1:
            sys.exit(
                f"Multiple .safetensors files found in {path} with no index.json "
                f"to tie them together. Point directly at the file you want."
            )
        return None, st_candidates

    if path.name.endswith(".index.json"):
        with open(path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shard_names = sorted(set(weight_map.values()))
        shard_paths = [path.parent / name for name in shard_names]
        missing = [p for p in shard_paths if not p.exists()]
        if missing:
            sys.exit(
                "Index file references shards that aren't present next to it:\n  "
                + "\n  ".join(str(m) for m in missing)
            )
        return path, shard_paths

    if path.suffix == ".safetensors":
        return None, [path]

    sys.exit(f"Don't know how to interpret checkpoint path: {path}")


def load_all_tensors(shard_paths):
    """Load every tensor from one or more .safetensors shards into a dict."""
    tensors = {}
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                if key in tensors:
                    sys.exit(f"Duplicate key '{key}' found across shards -- refusing to continue.")
                tensors[key] = f.get_tensor(key)
    return tensors


# --------------------------------------------------------------------------
# Format detection
# --------------------------------------------------------------------------

def detect_format(keys):
    keys_joined = "\n".join(keys)
    native_hits = sum(1 for m in NATIVE_MARKERS if m in keys_joined)
    diffusers_hits = sum(1 for m in DIFFUSERS_MARKERS if m in keys_joined)

    if native_hits and not diffusers_hits:
        return "native"
    if diffusers_hits and not native_hits:
        return "diffusers"
    if native_hits and diffusers_hits:
        sys.exit(
            "Checkpoint has markers from BOTH formats -- it may already be "
            "partially converted or is a different architecture. Aborting "
            "rather than guessing."
        )
    sys.exit(
        "Could not confidently detect the checkpoint format: found none of the "
        "expected marker keys (self_attn/cross_attn/head.head for native, "
        "attn1/attn2/proj_out/condition_embedder for diffusers). This may not "
        "be a Wan transformer checkpoint."
    )


# --------------------------------------------------------------------------
# Key conversion
# --------------------------------------------------------------------------

def map_clean_key(clean_key, direction):
    """
    Map an already-normalized key (no wrapper noise, no leading 'model.')
    from one format to the other. Returns None if unrecognized instead of
    exiting, so the caller can report it with full context.
    """
    top_map = NATIVE_TO_DIFFUSERS_TOP if direction == "native_to_diffusers" else DIFFUSERS_TO_NATIVE_TOP
    block_rules = NATIVE_TO_DIFFUSERS_BLOCK if direction == "native_to_diffusers" else DIFFUSERS_TO_NATIVE_BLOCK

    if clean_key in top_map:
        return top_map[clean_key]

    if clean_key in ("patch_embedding.weight", "patch_embedding.bias"):
        return clean_key  # unchanged in both formats

    m = BLOCK_RE.match(clean_key)
    if m:
        idx, rest = m.groups()
        for pattern, repl in block_rules:
            new_rest, n = pattern.subn(repl, rest)
            if n:
                return f"blocks.{idx}.{new_rest}"
    return None


def convert_tensors(tensors, direction, keep_prefix=True):
    """
    Normalize each key (strip FSDP/compile wrapper noise and a leading
    'model.' prefix if present), map the clean key to the target format,
    then re-apply the original prefix/wrapper noise unless keep_prefix=False.
    """
    converted = {}
    unrecognized = []
    stripped_examples = set()

    for key, tensor in tensors.items():
        clean, noise_fragments = strip_wrapper_noise(key)
        clean, leading_prefix = strip_known_leading_prefix(clean)

        if noise_fragments or leading_prefix:
            stripped_examples.add(key)

        new_clean = map_clean_key(clean, direction)
        if new_clean is None:
            unrecognized.append(key)
            continue

        if keep_prefix:
            # Re-apply exactly what we stripped, in the same relative
            # position: leading_prefix first, then the wrapper noise
            # re-inserted at the same depth (before 'blocks.N.' or at the
            # very front for top-level keys). We approximate this by
            # putting all noise fragments back at the front, which matches
            # how FSDP/compile wrappers actually nest in practice.
            out_key = "".join(noise_fragments) + new_clean
            if leading_prefix:
                out_key = leading_prefix + out_key
        else:
            out_key = new_clean

        if out_key in converted:
            sys.exit(f"Key collision after conversion: '{out_key}' produced by both an "
                      f"earlier key and '{key}'. Aborting.")
        converted[out_key] = tensor

    if unrecognized:
        preview = "\n  ".join(unrecognized[:15])
        more = f"\n  ... and {len(unrecognized) - 15} more" if len(unrecognized) > 15 else ""
        sys.exit(
            f"Don't know how to convert {len(unrecognized)} key(s), e.g.:\n  {preview}{more}\n\n"
            "These didn't match any known native/diffusers pattern even after stripping "
            "common training-wrapper noise (_fsdp_wrapped_module., _orig_mod., a leading "
            "'model.'). This checkpoint may use a wrapper prefix this script doesn't know "
            "about yet, or isn't a Wan transformer checkpoint."
        )

    if stripped_examples:
        example = next(iter(stripped_examples))
        if keep_prefix:
            print(f"Note: detected and preserved a training-wrapper prefix on {len(stripped_examples)} "
                  f"key(s), e.g. '{example}'. Use --strip-prefix if you want clean keys instead "
                  f"(needed for most inference loaders).")
        else:
            print(f"Note: detected and dropped a training-wrapper prefix on {len(stripped_examples)} "
                  f"key(s), e.g. '{example}'.")

    return converted


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_detect(args):
    path = Path(args.checkpoint)
    _, shard_paths = find_checkpoint_files(path)
    # For detection we only need the key names, not the tensor data.
    all_keys = []
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt") as f:
            all_keys.extend(f.keys())
    fmt = detect_format(all_keys)
    n_shards = len(shard_paths)
    print(f"Format:  {fmt}")
    print(f"Tensors: {len(all_keys)}")
    print(f"Shards:  {n_shards} file{'s' if n_shards != 1 else ''}")


def cmd_convert(args):
    in_path = Path(args.checkpoint)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, shard_paths = find_checkpoint_files(in_path)
    print(f"Loading {len(shard_paths)} shard(s)...")
    tensors = load_all_tensors(shard_paths)

    detected = detect_format(list(tensors.keys()))
    if args.to:
        target = args.to
        if target == detected:
            sys.exit(f"Checkpoint is already in '{detected}' format; nothing to do.")
    else:
        target = "diffusers" if detected == "native" else "native"

    direction = f"{detected}_to_{target}"
    print(f"Detected source format: {detected}")
    print(f"Converting to:          {target}")

    converted = convert_tensors(tensors, direction, keep_prefix=not args.strip_prefix)

    out_file = out_dir / "model.safetensors"
    print(f"Writing merged checkpoint ({len(converted)} tensors) to {out_file} ...")
    save_file(converted, str(out_file), metadata={"format": "pt"})
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_detect = sub.add_parser("detect", help="Detect checkpoint format without converting")
    p_detect.add_argument("checkpoint", help="Path to .safetensors file, .index.json, or directory")
    p_detect.set_defaults(func=cmd_detect)

    p_convert = sub.add_parser("convert", help="Convert checkpoint to the other format")
    p_convert.add_argument("checkpoint", help="Path to .safetensors file, .index.json, or directory")
    p_convert.add_argument("output", help="Output directory (will contain model.safetensors)")
    p_convert.add_argument(
        "--to",
        choices=["native", "diffusers"],
        default=None,
        help="Force target format instead of auto-detecting the opposite of the source",
    )
    p_convert.add_argument(
        "--strip-prefix",
        action="store_true",
        help="Drop any detected training-wrapper prefix (e.g. 'model.', '_fsdp_wrapped_module.') "
             "instead of preserving it on the converted keys. Recommended for checkpoints you plan "
             "to load into a normal inference pipeline rather than resume training on.",
    )
    p_convert.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
