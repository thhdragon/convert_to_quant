"""Group-wise Hadamard rotation for INT8 quantization quality improvement.

Originally from: https://github.com/newgrit1004/ComfyUI-ZImage-Triton
License: MIT

Spreads activation outliers across channels using orthogonal Hadamard matrices.
Based on QuaRot (2024) and ConvRot (2025) approaches, adapted for DiT models
with group-wise rotation to avoid row-wise outlier amplification.
"""

# INT4 W4A4 ConvRot functions are included for full parity with INT8 ConvRot.

import torch

# Cache Hadamard matrices by (size, device, dtype) to avoid recomputation
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def pack_int4_row_major(x: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values [-7, 7] into int8 bytes (2 values per byte).

    Uses int32 intermediates (matching comfy_kitchen reference) for safe
    bitwise operations across all PyTorch versions and platforms.
    """
    lo = x[..., 0::2].to(torch.int32) & 0x0F
    hi = x[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def unpack_int4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int8 bytes into signed int4 values stored in int8 [-7, 7].

    Uses int32 intermediates and torch.stack+reshape (matching comfy_kitchen
    reference) for correct sign-extension and safe bitwise operations.
    """
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


def quantize_convrot_w4a4_weight(
    weight: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate weight offline with ConvRot Hadamard and quantize to packed signed INT4."""
    try:
        from comfy_kitchen.tensor.convrot_w4a4 import quantize_convrot_w4a4_weight as ck_quant
        return ck_quant(weight, convrot_groupsize=convrot_groupsize, quant_group_size=quant_group_size)
    except Exception:
        pass

    out_f, in_f = weight.shape
    H = build_hadamard(convrot_groupsize, device=weight.device, dtype=weight.dtype)
    weight_rot = rotate_weight(weight, H, convrot_groupsize)

    # Per-row scale calculation for INT4 [-7, 7]
    absmax = weight_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax / 7.0
    scaled = (weight_rot / scales).round().clamp(-7, 7).to(torch.int8)
    qdata = pack_int4_row_major(scaled)
    return qdata, scales.reshape(out_f).to(torch.float32)


def dequantize_convrot_w4a4_weight(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize packed INT4 weights and rotate back using inverse Hadamard."""
    try:
        from comfy_kitchen.tensor.convrot_w4a4 import dequantize_convrot_w4a4_weight as ck_dequant
        return ck_dequant(qdata, scales, convrot_groupsize=convrot_groupsize, quant_group_size=quant_group_size, output_dtype=output_dtype)
    except Exception:
        pass

    unpacked = unpack_int4_row_major(qdata).to(torch.float32)
    w_rot = unpacked * scales.to(device=qdata.device, dtype=torch.float32).reshape(-1, 1)
    H = build_hadamard(convrot_groupsize, device=qdata.device, dtype=torch.float32)
    return rotate_weight(w_rot, H, convrot_groupsize).to(output_dtype)



def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot).

    Size must be a power of 4 (e.g., 4, 16, 64, 256, 1024...).
    Uses the Kronecker construction from Theorem 3.3 to avoid the all-1s
    column of standard Sylvester Hadamard matrices, which amplifies
    row-wise outliers in diffusion models.
    """
    import math
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(
            f"ConvRot Hadamard size must be a power of 4 (e.g. 4, 16, 64, 256, 1024), got {size}. "
            f"The standard Sylvester Hadamard fallback was removed because it has an all-ones DC "
            f"column that amplifies row-wise outliers in diffusion models. "
            f"Use a convrot_group_size that is a power of 4, or disable ConvRot for this layer."
        )

    # Base H4 from Theorem 3.3 (Eq 9 in the paper)
    # Notice how every row and column sums to exactly 2
    H4 = torch.tensor([[ 1,  1,  1, -1],
        [ 1,  1, -1,  1],[ 1, -1,  1,  1],[-1,  1,  1,  1]
    ], dtype=dtype, device=device)

    H = H4
    current_size = 4

    # Kronecker construction for larger sizes: H_{4^{k+1}} = H_{4^k} \otimes H_4
    while current_size < size:
        H = torch.kron(H, H4)
        current_size *= 4

    # Normalize to make it orthogonal
    H_normalized = H / (size**0.5)
    _HADAMARD_CACHE[cache_key] = H_normalized

    return H_normalized

def rotate_weight(
    weight: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate weight matrix offline: W_rot = W @ H_block^T.

    For Linear(in, out) with weight shape (out, in):
    Each row of W is split into groups of group_size and rotated by H^T.

    Args:
        weight: Shape (out_features, in_features).
        H: Normalized Hadamard matrix, shape (group_size, group_size).
        group_size: Group size for block-diagonal rotation.

    Returns:
        Rotated weight, same shape as input.
    """
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size

    # (out, in) → (out, n_groups, group_size)
    W_grouped = weight.view(out_f, n_groups, group_size)
    # Apply H^T to each group: (..., group_size) @ (group_size, group_size)
    H_t = H.T.to(dtype=weight.dtype, device=weight.device)
    W_rot = torch.matmul(W_grouped, H_t)
    return W_rot.reshape(out_f, in_f)


def rotate_activation(
    x: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate activation online: x_rot = x @ H_block.

    Group-wise Hadamard spreads outliers across channels within each group.

    Args:
        x: Shape (..., features). Last dim must be divisible by group_size.
        H: Normalized Hadamard matrix, shape (group_size, group_size).
        group_size: Group size for block-diagonal rotation.

    Returns:
        Rotated activation, same shape as input.
    """
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(
            f"features {features} not divisible by group_size {group_size}"
        )
    n_groups = features // group_size

    # (..., features) → (..., n_groups, group_size)
    x_grouped = x.view(*orig_shape[:-1], n_groups, group_size)
    H_dev = H.to(dtype=x.dtype, device=x.device)
    x_rot = torch.matmul(x_grouped, H_dev)
    return x_rot.view(orig_shape)


def find_max_compatible_group_size(in_features: int, min_group_size: int = 256) -> int | None:
    """Return static group size (default 256) if in_features >= min_group_size and divisible by group_size, else None."""
    group_size = min_group_size
    if in_features >= group_size and in_features % group_size == 0:
        return group_size
    return None


def balance_channels_smoothquant(
    weight: torch.Tensor,
    x_calib: torch.Tensor,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform SmoothQuant per-channel scale balancing prior to Hadamard rotation.

    s_c = max(|X_c|)^alpha / max(|W_c|)^(1-alpha)
    Returns: (weight_balanced, smooth_scales)
    """
    with torch.no_grad():
        act_max = x_calib.abs().amax(dim=0).clamp_min(1e-5)
        weight_max = weight.abs().amax(dim=0).clamp_min(1e-5)
        scales = (act_max.pow(alpha) / weight_max.pow(1.0 - alpha)).clamp_min(1e-5)
        scales = scales / scales.mean()
        weight_balanced = weight * scales.unsqueeze(0)
    return weight_balanced, scales


