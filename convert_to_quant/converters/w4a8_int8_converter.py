"""
W4A8 INT8 Quantization Converter (AsymW4A8Int8Layout).

Implements grouped W4A8 INT8 quantization with ConvRot Hadamard rotation,
per-group FP8/FP32 relative scales, per-channel FP32 scales, optional zero-point
correction, and optional Lloyd-Max codebooks.

Based on comfy-kitchen (Comfy Org, Apache-2.0).
"""

from typing import Dict, Optional, Tuple

torch = None
try:
    import torch
except ImportError:
    pass

from ..constants import W4A8_CONVROT_GROUPSIZE, W4A8_GROUP_SIZE, W4A8_SCALE_DTYPE
from ..utils.logging import verbose
from .base_converter import BaseLearnedConverter

# Check for comfy-kitchen availability
try:
    import comfy_kitchen.tensor.w4a8_int8 as ck_w4a8

    HAS_COMFY_KITCHEN = True
except ImportError:
    HAS_COMFY_KITCHEN = False


def quantize_w4a8_int8_pytorch(
    weight: torch.Tensor,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
    codebook: bool = True,
    codebook_tensor: Optional[torch.Tensor] = None,
    stochastic_rounding: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Quantize floating weight tensor to W4A8 storage."""
    if HAS_COMFY_KITCHEN:
        return ck_w4a8.quantize_w4a8_int8_weight(
            weight,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            symmetric=symmetric,
            scale_dtype=scale_dtype,
            codebook=codebook,
            codebook_tensor=codebook_tensor,
            stochastic_rounding=stochastic_rounding,
        )
    raise RuntimeError("w4a8_int8 quantization requires comfy_kitchen installed.")


def dequantize_w4a8_int8_pytorch(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: Optional[torch.Tensor] = None,
    correction: Optional[torch.Tensor] = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize packed W4A8 weight back into floating representation."""
    if HAS_COMFY_KITCHEN:
        return ck_w4a8.dequantize_w4a8_int8_weight(
            qdata,
            s_rel,
            s_channel,
            codebook=codebook,
            correction=correction,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            output_dtype=output_dtype,
        )
    raise RuntimeError("w4a8_int8 dequantization requires comfy_kitchen installed.")


class W4A8Int8Converter:
    """
    W4A8 INT8 block quantization converter.

    Quantizes weights using 16-element groups, ConvRot Hadamard rotation,
    per-channel scales, per-group FP8 relative scales, and optional codebooks.
    """

    def __init__(
        self,
        group_size: int = 16,
        convrot_groupsize: int = 256,
        symmetric: bool = True,
        scale_dtype: torch.dtype = torch.float32,
        codebook: bool = True,
        stochastic_rounding: int = 0,
    ):
        self.group_size = group_size
        self.convrot_groupsize = convrot_groupsize
        self.symmetric = symmetric
        self.scale_dtype = scale_dtype
        self.codebook = codebook
        self.stochastic_rounding = stochastic_rounding

    def convert(
        self,
        W_orig: torch.Tensor,
        key: Optional[str] = None,
        depth: int = -1,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Dict,
    ]:
        """
        Convert weight tensor to W4A8 format.

        Returns:
            Tuple of (packed_qdata, s_rel, s_channel, correction, codebook, dequantized_weight, extra_tensors)
        """
        qdata, s_rel, s_channel, correction, codebook_tensor = quantize_w4a8_int8_pytorch(
            W_orig,
            group_size=self.group_size,
            convrot_groupsize=self.convrot_groupsize,
            symmetric=self.symmetric,
            scale_dtype=self.scale_dtype,
            codebook=self.codebook,
            stochastic_rounding=self.stochastic_rounding,
        )

        dequantized = dequantize_w4a8_int8_pytorch(
            qdata,
            s_rel,
            s_channel,
            codebook=codebook_tensor,
            correction=correction,
            group_size=self.group_size,
            convrot_groupsize=self.convrot_groupsize,
            output_dtype=W_orig.dtype,
        )

        return qdata, s_rel, s_channel, correction, codebook_tensor, dequantized, {}


class LearnedW4A8Int8Converter(BaseLearnedConverter):
    """
    Learned Rounding W4A8 INT8 converter.

    Applies W4A8 INT8 weight quantization with stochastic rounding or ALS codebook optimization.
    """

    def __init__(
        self,
        group_size: int = 16,
        convrot_groupsize: int = 256,
        symmetric: bool = True,
        scale_dtype: torch.dtype = torch.float32,
        codebook: bool = True,
        stochastic_rounding: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.group_size = group_size
        self.convrot_groupsize = convrot_groupsize
        self.symmetric = symmetric
        self.scale_dtype = scale_dtype
        self.codebook = codebook
        self.stochastic_rounding = stochastic_rounding

        verbose(f"LearnedW4A8Int8Converter initialized on device: {self.device}")
        verbose(f"  - Format: W4A8 INT8 (group_size={self.group_size}, convrot={self.convrot_groupsize}, scale_dtype={self.scale_dtype})")
        verbose(f"  - Stochastic rounding iterations: {self.stochastic_rounding} ({'disabled' if self.no_learned_rounding else 'enabled'})")

    def convert(
        self,
        W_orig: torch.Tensor,
        key: Optional[str] = None,
        depth: int = -1,
        calibration_data: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Dict,
    ]:
        from tqdm import tqdm
        from torch.optim import AdamW, RAdam
        from ..utils.logging import info, warning

        W_dev = W_orig.to(device=self.device, dtype=torch.float32)

        stoch_rounds = self.stochastic_rounding

        import sys
        pbar = tqdm(
            range(max(1, self.num_iter if not self.no_learned_rounding else 1)),
            desc=f"    Optimizing W4A8 (comfy_kitchen-LloydMax)",
            file=sys.stdout,
            mininterval=0.0,
            miniters=1,
            leave=True,
            dynamic_ncols=True,
        )

        qdata, s_rel, s_channel, correction, codebook_tensor = quantize_w4a8_int8_pytorch(
            W_dev,
            group_size=self.group_size,
            convrot_groupsize=self.convrot_groupsize,
            symmetric=self.symmetric,
            scale_dtype=self.scale_dtype,
            codebook=self.codebook,
            stochastic_rounding=stoch_rounds,
        )

        dequantized = dequantize_w4a8_int8_pytorch(
            qdata,
            s_rel,
            s_channel,
            codebook=codebook_tensor,
            correction=correction,
            group_size=self.group_size,
            convrot_groupsize=self.convrot_groupsize,
            output_dtype=W_orig.dtype,
        )

        with torch.no_grad():
            if calibration_data is not None:
                X = calibration_data.to(device=self.device, dtype=torch.float32)
                Y_ref = X @ W_dev.T
                Y_pred = X @ dequantized.to(dtype=torch.float32).T
                mse_val = torch.nn.functional.mse_loss(Y_pred, Y_ref).item()
            else:
                mse_val = torch.nn.functional.mse_loss(dequantized.to(dtype=torch.float32), W_dev).item()

        pbar.set_postfix({"loss": f"{mse_val:.3e}", "best": f"{mse_val:.3e}"}, refresh=True)
        pbar.update(pbar.total)
        pbar.close()
        sys.stdout.flush()

        info(f"      - Finished: best_loss={mse_val:.3e}, iters={self.num_iter}")

        extra_tensors = {}
        if self._should_extract_lora(key, W_orig.shape, depth):
            lora_data = self._extract_error_lora(W_orig, dequantized)
            if lora_data:
                extra_tensors.update(lora_data)

        return qdata, s_rel, s_channel, correction, codebook_tensor, dequantized, extra_tensors

