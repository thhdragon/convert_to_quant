"""
Test Codebook AdaRound optimization for W4A8 INT8.
"""

import pytest
import torch

from convert_to_quant.converters.w4a8_int8_converter import (
    HAS_COMFY_KITCHEN,
    quantize_w4a8_int8_pytorch,
    dequantize_w4a8_int8_pytorch,
)
from convert_to_quant.utils.convrot import build_hadamard, rotate_weight, pack_int4_row_major


def test_codebook_adaround():
    if not HAS_COMFY_KITCHEN:
        print("comfy_kitchen not installed")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    M, N = 3072, 3072
    group_size = 16
    convrot_groupsize = 256

    W_dev = torch.randn(M, N, dtype=torch.float32, device=device) * 0.02
    X = torch.randn(128, N, dtype=torch.float32, device=device)
    Y_ref = X @ W_dev.T

    # 1. Seed quantization to get initial scales and codebook
    qdata0, s_rel0, s_channel0, corr0, cb0 = quantize_w4a8_int8_pytorch(
        W_dev,
        group_size=group_size,
        convrot_groupsize=convrot_groupsize,
        symmetric=True,
        scale_dtype=torch.float32,
        codebook=True,
        stochastic_rounding=0,
    )

    dequant0 = dequantize_w4a8_int8_pytorch(
        qdata0, s_rel0, s_channel0, codebook=cb0, correction=corr0,
        group_size=group_size, convrot_groupsize=convrot_groupsize, output_dtype=torch.float32,
    )
    init_mse = torch.nn.functional.mse_loss(X @ dequant0.T, Y_ref).item()
    print(f"Seed Baseline MSE: {init_mse:.6e}")

    # 2. Setup Codebook AdaRound
    H = build_hadamard(convrot_groupsize, device=device, dtype=torch.float32)
    W_rot = rotate_weight(W_dev, H, convrot_groupsize)

    s_rel_expanded = s_rel0.to(torch.float32).repeat_interleave(group_size, dim=1)
    s_channel_bc = s_channel0.view(-1, 1).to(torch.float32)
    s_total = s_channel_bc * s_rel_expanded

    # Normalized float weights
    z = (W_rot / s_total.clamp_min(1e-12)).clamp(-1.0, 1.0)

    # Bounding codebook indices
    # cb0 has 16 sorted values e.g. [-0.98, ..., +0.98]
    cb = cb0.to(device=device, dtype=torch.float32)
    # Find lower bounding index k for each z value
    # z shape [M, N], cb shape [16]
    # diff = z.unsqueeze(-1) - cb (shape [M, N, 16])
    diff = z.unsqueeze(-1) - cb
    # For z >= cb[k], diff >= 0
    valid_mask = diff >= 0
    k_lower = valid_mask.sum(dim=-1) - 1
    k_lower = k_lower.clamp(0, 14)
    k_upper = k_lower + 1

    c_lower = cb[k_lower]
    c_upper = cb[k_upper]
    gap = (c_upper - c_lower).clamp_min(1e-8)

    # Initial fraction alpha in [0, 1]
    alpha = ((z - c_lower) / gap).clamp(1e-6, 1.0 - 1e-6)

    T_start, T_end = 20.0, 2.0
    V_init = -torch.log((1.0 / alpha) - 1.0) * T_start
    V = V_init.clone().detach().requires_grad_(True)

    optimizer = torch.optim.AdamW([V], lr=1.0)
    best_loss = float("inf")
    best_V = V.clone().detach()

    for i in range(100):
        optimizer.zero_grad()
        temp = T_start + (T_end - T_start) * (i / 100)
        h_V = torch.sigmoid(V / temp)

        c_soft = c_lower + h_V * gap
        W_rot_dequant = c_soft * s_total
        W_dequant_soft = rotate_weight(W_rot_dequant, H, convrot_groupsize)

        Y_pred = X @ W_dequant_soft.T
        loss_mse = torch.nn.functional.mse_loss(Y_pred, Y_ref) / max(init_mse, 1e-12)
        loss_reg = (1.0 - (2.0 * h_V - 1.0).pow(2)).mean()
        loss = loss_mse + 0.1 * loss_reg

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            h_V_hard = (h_V >= 0.5).float()
            c_hard = c_lower + h_V_hard * gap
            W_rot_hard = c_hard * s_total
            W_dequant_hard = rotate_weight(W_rot_hard, H, convrot_groupsize)
            hard_mse = torch.nn.functional.mse_loss(X @ W_dequant_hard.T, Y_ref).item()
            if hard_mse < best_loss:
                best_loss = hard_mse
                best_V = V.clone().detach()

    # Final hard index mapping
    best_h = torch.sigmoid(best_V / T_end)
    chosen_index = torch.where(best_h >= 0.5, k_upper, k_lower).to(torch.int8)

    # Pack chosen_index into qdata byte array (2 indices per byte)
    qdata_learned = pack_int4_row_major(chosen_index)

    dequant_learned = dequantize_w4a8_int8_pytorch(
        qdata_learned, s_rel0, s_channel0, codebook=cb0, correction=corr0,
        group_size=group_size, convrot_groupsize=convrot_groupsize, output_dtype=torch.float32,
    )
    final_mse = torch.nn.functional.mse_loss(X @ dequant_learned.T, Y_ref).item()

    print(f"Seed Baseline MSE:         {init_mse:.6e}")
    print(f"Best Hard AdaRound MSE:    {best_loss:.6e} (ratio: {best_loss / init_mse:.4f})")
    print(f"Final Packed Dequant MSE: {final_mse:.6e} (ratio: {final_mse / init_mse:.4f})")

    assert final_mse <= init_mse * 1.01, f"Final MSE ({final_mse:.6e}) should be <= seed baseline ({init_mse:.6e})"
    print("SUCCESS: Codebook AdaRound verified!")


if __name__ == "__main__":
    test_codebook_adaround()
