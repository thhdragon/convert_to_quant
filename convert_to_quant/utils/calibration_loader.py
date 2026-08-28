"""
Calibration data loader for Post-Training Quantization (PTQ).

Loads real-world activation tensors collected from diffusion sampler runs
(e.g., ComfyUI-PTQ-Sampler) and matches them to model layer weights.
"""

import glob
import os
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from safetensors import safe_open

from .logging import debug, info, verbose, warning


def normalize_layer_key(key: str) -> str:
    """
    Normalize layer key by removing common prefixes and suffixes for matching.
    e.g. 'diffusion_model.double_blocks.0.img_attn.proj.weight' -> 'double_blocks.0.img_attn.proj'
    """
    if key.endswith(".weight"):
        key = key[:-7]
    elif key.endswith(".bias"):
        key = key[:-5]

    for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break

    return key


class CalibrationDataLoader:
    """
    Loads and manages real calibration activation tensors from safetensors files.
    """

    def __init__(self, calib_path: str, max_tokens: int = 4096, seed: int = 42):
        self.calib_path = calib_path
        self.max_tokens = max_tokens
        self.seed = seed
        self.files: List[str] = []
        self.key_to_files: Dict[str, List[str]] = {}
        self.normalized_to_raw_keys: Dict[str, Set[str]] = {}

        self._discover_files()
        self._index_keys()

    def _discover_files(self) -> None:
        """Find all safetensors calibration files."""
        if os.path.isfile(self.calib_path):
            if self.calib_path.endswith(".safetensors"):
                self.files = [self.calib_path]
        elif os.path.isdir(self.calib_path):
            self.files = sorted(glob.glob(os.path.join(self.calib_path, "**", "*.safetensors"), recursive=True))

        if not self.files:
            warning(f"No .safetensors calibration files found in '{self.calib_path}'")
        else:
            info(f"Discovered {len(self.files)} calibration files from '{self.calib_path}'")

    def _index_keys(self) -> None:
        """Index all layer keys present across the calibration files."""
        for fpath in self.files:
            try:
                with safe_open(fpath, framework="pt") as f:
                    for k in f.keys():
                        if k not in self.key_to_files:
                            self.key_to_files[k] = []
                        self.key_to_files[k].append(fpath)

                        norm_k = normalize_layer_key(k)
                        if norm_k not in self.normalized_to_raw_keys:
                            self.normalized_to_raw_keys[norm_k] = set()
                        self.normalized_to_raw_keys[norm_k].add(k)
            except Exception as e:
                warning(f"Error inspecting calibration file '{fpath}': {e}")

        info(f"Indexed {len(self.normalized_to_raw_keys)} unique calibration layer targets across {len(self.files)} files.")

    def has_layer(self, weight_key: str) -> bool:
        """Check if calibration data is available for a given weight key."""
        norm_k = normalize_layer_key(weight_key)
        return norm_k in self.normalized_to_raw_keys or weight_key in self.key_to_files

    def get_calibration_tensor(
        self,
        weight_key: str,
        max_tokens: Optional[int] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve and consolidate activation calibration data for a specific layer.

        Args:
            weight_key: Model weight key (e.g. 'double_blocks.0.img_attn.proj.weight')
            max_tokens: Maximum number of token samples to return
            device: Target device for output tensor
            dtype: Target dtype for output tensor

        Returns:
            torch.Tensor of shape (N_samples, in_features) or None if not found
        """
        norm_k = normalize_layer_key(weight_key)
        raw_keys = self.normalized_to_raw_keys.get(norm_k)

        if not raw_keys and weight_key in self.key_to_files:
            raw_keys = {weight_key}

        if not raw_keys:
            return None

        # Find matching raw key in indexed files
        target_raw_key = None
        for rk in raw_keys:
            if rk in self.key_to_files:
                target_raw_key = rk
                break

        if not target_raw_key:
            return None

        matching_files = self.key_to_files[target_raw_key]
        tokens_limit = max_tokens or self.max_tokens

        collected_tensors: List[torch.Tensor] = []
        for fpath in matching_files:
            try:
                with safe_open(fpath, framework="pt") as f:
                    if target_raw_key in f.keys():
                        t = f.get_tensor(target_raw_key)
                        # Flatten to 2D if needed: (N_tokens, in_features)
                        if t.ndim > 2:
                            t = t.view(-1, t.shape[-1])
                        collected_tensors.append(t.to(dtype=dtype, device="cpu"))
            except Exception as e:
                debug(f"Failed to read tensor '{target_raw_key}' from '{fpath}': {e}")

        if not collected_tensors:
            return None

        combined = torch.cat(collected_tensors, dim=0)

        # Subsample if total tokens exceed limit
        total_tokens = combined.shape[0]
        if total_tokens > tokens_limit:
            gen = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(total_tokens, generator=gen)[:tokens_limit]
            combined = combined[indices]

        return combined.to(device=device)
