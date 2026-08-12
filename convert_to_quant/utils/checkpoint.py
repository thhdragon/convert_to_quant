"""
Quantization Checkpoint & Resume Manager for convert_to_quant.

Provides per-layer checkpointing, sidecar progress JSON tracking,
stop/resume support, and sharded safetensors output assembly.
"""

import glob
import json
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .logging import error, info, verbose, warning


def parse_shard_size(size_str: Optional[Union[str, int]]) -> Optional[int]:
    """
    Parse a human-readable shard size string into bytes.

    Examples:
        "5GB" -> 5 * 1024^3
        "500MB" -> 500 * 1024^2
        5368709120 -> 5368709120

    Returns:
        Number of bytes as integer, or None if disabled/unspecified.
    """
    if size_str is None or size_str == "" or size_str == 0:
        return None
    if isinstance(size_str, (int, float)):
        return int(size_str)

    s = str(size_str).strip().upper()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT]?B?)$", s)
    if not match:
        raise ValueError(f"Invalid shard size format: '{size_str}'. Use e.g. '5GB', '2000MB'.")

    value, unit = match.groups()
    val = float(value)
    if unit in ("K", "KB"):
        return int(val * 1024)
    elif unit in ("M", "MB"):
        return int(val * 1024**2)
    elif unit in ("G", "GB") or unit == "":
        return int(val * 1024**3)
    elif unit in ("T", "TB"):
        return int(val * 1024**4)
    else:
        return int(val)


def get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Get approximate size in bytes of a PyTorch tensor."""
    return tensor.element_size() * tensor.numel()


class QuantCheckpointManager:
    """
    Manages per-layer checkpointing, sidecar progress JSON tracking,
    resuming interrupted runs, and assembling output safetensors.
    """

    def __init__(
        self,
        output_file: str,
        input_file: str,
        primary_format: str = "fp8",
        resume: bool = False,
        sidecar_path: Optional[str] = None,
        max_shard_size: Optional[Union[str, int]] = None,
        no_checkpoint: bool = False,
    ):
        self.output_file = os.path.abspath(output_file)
        self.input_file = os.path.abspath(input_file)
        self.primary_format = primary_format
        self.disabled = no_checkpoint
        self.max_shard_size_bytes = parse_shard_size(max_shard_size)

        out_dir = os.path.dirname(self.output_file) or "."
        os.makedirs(out_dir, exist_ok=True)

        if sidecar_path:
            self.sidecar_path = os.path.abspath(sidecar_path)
        else:
            self.sidecar_path = f"{self.output_file}.progress.json"

        # Checkpoint directory holds per-layer tensors
        self.checkpoint_dir = f"{self.output_file}.checkpoint"

        self.state: Dict[str, Any] = {
            "version": "1.0",
            "input_file": self.input_file,
            "output_file": self.output_file,
            "primary_format": self.primary_format,
            "status": "in_progress",
            "completed_layers": {},  # key -> {"status": "completed", "tensors_file": ..., "meta_entry": ...}
            "quant_metadata_layers": {},
        }

        self.resumed = False
        if not self.disabled:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            if resume or os.path.exists(self.sidecar_path):
                self._load_existing_sidecar()

    def _load_existing_sidecar(self) -> None:
        """Load state from an existing sidecar progress JSON."""
        if not os.path.exists(self.sidecar_path):
            verbose(f"No existing sidecar found at: {self.sidecar_path}")
            return

        try:
            with open(self.sidecar_path, "r", encoding="utf-8") as f:
                loaded_state = json.load(f)

            # Validate input file matching
            if loaded_state.get("input_file") == self.input_file or os.path.basename(
                loaded_state.get("input_file", "")
            ) == os.path.basename(self.input_file):
                self.state = loaded_state
                self.resumed = True
                completed_count = len(self.state.get("completed_layers", {}))
                info(
                    f"Resuming quantization session from sidecar: {self.sidecar_path} "
                    f"({completed_count} layers already completed)"
                )
            else:
                warning(
                    f"Sidecar input file mismatch ({loaded_state.get('input_file')} vs {self.input_file}). Starting fresh session."
                )
        except Exception as e:
            warning(f"Failed to parse existing sidecar JSON '{self.sidecar_path}': {e}. Starting fresh session.")

    def _flush_sidecar(self) -> None:
        """Save the current state atomically to the sidecar progress JSON file."""
        if self.disabled:
            return

        temp_sidecar = f"{self.sidecar_path}.tmp"
        try:
            with open(temp_sidecar, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
            os.replace(temp_sidecar, self.sidecar_path)
        except Exception as e:
            warning(f"Could not save progress sidecar: {e}")

    def is_layer_completed(self, key: str) -> bool:
        """Check if a weight layer has already been completed in a previous run."""
        if self.disabled:
            return False
        layer_entry = self.state["completed_layers"].get(key)
        if not layer_entry:
            return False
        # Check if tensor file exists on disk
        tensors_file = layer_entry.get("tensors_file")
        if tensors_file and os.path.exists(tensors_file):
            return True
        return False

    def load_completed_layer(self, key: str) -> Optional[Dict[str, Any]]:
        """Load tensors and metadata for a previously completed layer."""
        if not self.is_layer_completed(key):
            return None

        layer_entry = self.state["completed_layers"][key]
        tensors_file = layer_entry["tensors_file"]

        try:
            tensors = load_file(tensors_file)
            return {
                "key": key,
                "base_name": layer_entry.get("base_name"),
                "tensors": tensors,
                "lora_tensors": {},
                "meta_entry": layer_entry.get("meta_entry"),
                "skipped": layer_entry.get("skipped", False),
                "use_custom": layer_entry.get("use_custom", False),
                "use_fallback": layer_entry.get("use_fallback", False),
                "use_layer_config": layer_entry.get("use_layer_config", False),
            }
        except Exception as e:
            error(f"Error loading completed layer checkpoint '{tensors_file}': {e}")
            return None

    def save_layer_checkpoint(self, result: Dict[str, Any]) -> None:
        """
        Immediately save a layer's output tensors and update sidecar progress.
        Called as soon as a single layer finishes quantization.
        """
        if self.disabled or not result:
            return

        key = result.get("key")
        if not key:
            return

        # Clean key filename for checkpoint storage
        safe_name = re.sub(r"[^\w\-.]", "_", key)
        ckpt_file = os.path.join(self.checkpoint_dir, f"{safe_name}.safetensors")

        tensors = result.get("tensors", {})
        if tensors:
            save_file(tensors, ckpt_file)

        base_name = result.get("base_name")
        meta_entry = result.get("meta_entry")

        self.state["completed_layers"][key] = {
            "status": "completed",
            "base_name": base_name,
            "tensors_file": ckpt_file,
            "meta_entry": meta_entry,
            "skipped": result.get("skipped", False),
            "use_custom": result.get("use_custom", False),
            "use_fallback": result.get("use_fallback", False),
            "use_layer_config": result.get("use_layer_config", False),
        }

        if base_name and meta_entry:
            self.state["quant_metadata_layers"][base_name] = meta_entry

        self._flush_sidecar()

    def assemble_final_output(
        self,
        passthrough_tensors: Dict[str, torch.Tensor],
        original_metadata: Optional[Dict[str, str]] = None,
        lora_tensors: Optional[Dict[str, torch.Tensor]] = None,
        lora_save_path: Optional[str] = None,
    ) -> bool:
        """
        Collect all layer checkpoints, combine with remaining passthrough tensors,
        and save to final output file(s). Supports sharded output if max_shard_size is set.
        """
        info(f"Assembling final output tensors into: {self.output_file}")
        all_tensors: Dict[str, torch.Tensor] = {}

        # 1. Load all layer checkpoints
        for key, layer_entry in self.state["completed_layers"].items():
            ckpt_file = layer_entry.get("tensors_file")
            if ckpt_file and os.path.exists(ckpt_file):
                try:
                    loaded = load_file(ckpt_file)
                    all_tensors.update(loaded)
                except Exception as e:
                    error(f"Failed loading checkpoint file '{ckpt_file}': {e}")
                    return False

        # 2. Add passthrough tensors (bias, norms, embeddings)
        all_tensors.update(passthrough_tensors)

        # 3. Prepare metadata
        output_metadata = dict(original_metadata or {})
        quant_meta = self.state.get("quant_metadata_layers", {})
        if quant_meta:
            full_metadata = {"format_version": "1.0", "layers": quant_meta}
            output_metadata["_quantization_metadata"] = json.dumps(full_metadata)

        save_kwargs = {"metadata": output_metadata} if output_metadata else {}

        # 4. Save single or sharded safetensors
        try:
            if self.max_shard_size_bytes and self.max_shard_size_bytes > 0:
                self._save_sharded(all_tensors, save_kwargs)
            else:
                info(f"Saving {len(all_tensors)} tensors to single file: {self.output_file}")
                save_file(all_tensors, self.output_file, **save_kwargs)

            # Save LoRA adapter if present
            if lora_tensors:
                target_lora_path = lora_save_path or self.output_file.replace(".safetensors", "_lora.safetensors")
                info(f"Saving {len(lora_tensors)} extracted LoRA tensors to: {target_lora_path}")
                save_file(lora_tensors, target_lora_path)

            # Mark state completed and clean up checkpoint directory
            self.state["status"] = "completed"
            self._flush_sidecar()
            self._cleanup_checkpoint_dir()
            return True
        except Exception as e:
            error(f"FATAL: Error saving final assembled file '{self.output_file}': {e}")
            return False

    def _save_sharded(self, tensors: Dict[str, torch.Tensor], save_kwargs: dict) -> None:
        """Save model split across multiple safetensors shards + index JSON."""
        total_size = sum(get_tensor_size_bytes(t) for t in tensors.values())
        max_size = self.max_shard_size_bytes

        info(
            f"Saving sharded safetensors: total size ~{total_size / (1024**3):.2f} GB "
            f"(max shard size: {max_size / (1024**3):.2f} GB)"
        )

        base_name, ext = os.path.splitext(self.output_file)
        shards: List[Dict[str, torch.Tensor]] = []
        current_shard: Dict[str, torch.Tensor] = {}
        current_shard_bytes = 0

        for key, tensor in tensors.items():
            t_bytes = get_tensor_size_bytes(tensor)
            if current_shard and (current_shard_bytes + t_bytes > max_size):
                shards.append(current_shard)
                current_shard = {}
                current_shard_bytes = 0

            current_shard[key] = tensor
            current_shard_bytes += t_bytes

        if current_shard:
            shards.append(current_shard)

        num_shards = len(shards)
        weight_map: Dict[str, str] = {}

        for index, shard in enumerate(shards, start=1):
            shard_filename = f"{base_name}-{index:05d}-of-{num_shards:05d}{ext}"
            info(f"  - Writing shard {index}/{num_shards} ({len(shard)} tensors) to: {shard_filename}")
            save_file(shard, shard_filename, **save_kwargs)
            for k in shard.keys():
                weight_map[k] = os.path.basename(shard_filename)

        # Write standard HuggingFace index JSON
        index_file = f"{base_name}.safetensors.index.json"
        index_data = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
        info(f"  - Index manifest written to: {index_file}")

    def _cleanup_checkpoint_dir(self) -> None:
        """Clean up temporary layer checkpoint directory after successful assembly."""
        if os.path.exists(self.checkpoint_dir):
            try:
                shutil.rmtree(self.checkpoint_dir)
                verbose(f"Cleaned up checkpoint directory: {self.checkpoint_dir}")
            except Exception as e:
                warning(f"Could not remove checkpoint directory '{self.checkpoint_dir}': {e}")
