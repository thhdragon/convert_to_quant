"""
INT4 W4A4 ConvRot Quantization Kernels and Wrappers.

Provides INT4 W4A4 group-wise Hadamard rotation (ConvRot) weight and activation
quantization routines and linear matrix multiplication handlers.

Integrates with comfy-kitchen HIP/CUDA/eager backends when available, with clean
PyTorch eager fallback.
"""

from typing import Optional, Tuple
import torch

from ..utils.convrot import (
    build_hadamard,
    dequantize_convrot_w4a4_weight as _util_dequant,
    pack_int4_row_major,
    quantize_convrot_w4a4_weight as _util_quant,
    rotate_activation,
    rotate_weight,
    unpack_int4_row_major,
)


def quantize_convrot_w4a4_weight(
    weight: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = 64,
    stochastic_rounding: Optional[int] = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight matrix offline using ConvRot Hadamard rotation and INT4 packing.

    Args:
        weight: Shape (out_features, in_features)
        convrot_groupsize: Hadamard rotation group size (default 256)
        quant_group_size: INT4 quantization group size (default 64)
        stochastic_rounding: Seed for stochastic rounding if > 0

    Returns:
        Tuple of (qdata int8 tensor shape (out_features, in_features // 2), scales float32 tensor shape (out_features,))
    """
    try:
        from comfy_kitchen.tensor.convrot_w4a4 import quantize_convrot_w4a4_weight as ck_quant
        return ck_quant(weight, convrot_groupsize=convrot_groupsize, quant_group_size=quant_group_size, stochastic_rounding=stochastic_rounding)
    except Exception:
        return _util_quant(weight, convrot_groupsize=convrot_groupsize, quant_group_size=quant_group_size)


def dequantize_convrot_w4a4_weight(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize packed INT4 weight matrix and rotate back to original basis.

    Args:
        qdata: Shape (out_features, in_features // 2), dtype int8
        scales: Shape (out_features,) float32 scale per row
        convrot_groupsize: Hadamard rotation group size (default 256)
        quant_group_size: INT4 quantization group size (default 64)
        output_dtype: Target precision dtype

    Returns:
        Dequantized weight matrix shape (out_features, in_features)
    """
    try:
        from comfy_kitchen.tensor.convrot_w4a4 import dequantize_convrot_w4a4_weight as ck_dequant
        return ck_dequant(qdata, scales, convrot_groupsize=convrot_groupsize, quant_group_size=quant_group_size, output_dtype=output_dtype)
    except Exception:
        return _util_dequant(qdata, scales, convrot_groupsize=convrot_groupsize, quant_group_size=quant_group_size, output_dtype=output_dtype)


def quantize_signed_int4_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize 2D activations per-row into signed 4-bit integers [-7, 7] packed into int8."""
    rows = x.shape[0]
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax / 7.0
    scaled = (x / scales).round().clamp(-7, 7).to(torch.int8)
    qdata = pack_int4_row_major(scaled)
    return qdata, scales.reshape(rows).to(torch.float32)


def convrot_w4a4_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = 64,
    linear_dtype: str = "int4",
) -> torch.Tensor:
    """
    Compute x @ W.T + bias using ConvRot W4A4 linear operator.

    Args:
        x: Input activations, shape (..., in_features)
        qweight: Packed weight data, shape (out_features, in_features // 2), dtype int8
        wscales: Weight scaling factors, shape (out_features,)
        bias: Optional bias vector, shape (out_features,)
        convrot_groupsize: Group size for Hadamard rotation
        quant_group_size: Quantization group size
        linear_dtype: Precision mode ('int4' or 'int8')

    Returns:
        Output tensor shape (..., out_features)
    """
    try:
        from comfy_kitchen.tensor.convrot_w4a4 import convrot_w4a4_linear as ck_linear
        return ck_linear(
            x,
            qweight,
            wscales,
            bias=bias,
            convrot_groupsize=convrot_groupsize,
            quant_group_size=quant_group_size,
            linear_dtype=linear_dtype,
        )
    except Exception:
        pass

    # PyTorch Eager Fallback
    orig_shape = x.shape
    in_features = orig_shape[-1]
    out_features = qweight.shape[0]

    x2d = x.reshape(-1, in_features).contiguous()
    H = build_hadamard(convrot_groupsize, device=x2d.device, dtype=x2d.dtype)
    x_rot = rotate_activation(x2d, H, convrot_groupsize).contiguous()

    # Online activation quantization
    qact, x_scales = quantize_signed_int4_rowwise(x_rot)

    # Unpack int4 weights and activations for matmul
    act_unpacked = unpack_int4_row_major(qact).to(dtype=torch.float32)
    w_unpacked = unpack_int4_row_major(qweight).to(dtype=torch.float32)

    # Compute contraction: (M, K) @ (N, K).T -> (M, N)
    res = act_unpacked @ w_unpacked.t()
    res = res * x_scales.reshape(-1, 1).to(device=res.device, dtype=res.dtype)
    res = res * wscales.reshape(1, -1).to(device=res.device, dtype=res.dtype)

    if bias is not None:
        res = res + bias.to(device=res.device, dtype=res.dtype).reshape(1, -1)

    return res.to(dtype=x.dtype).reshape(*orig_shape[:-1], out_features)