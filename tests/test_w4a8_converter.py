"""
Test LearnedW4A8Int8Converter with comfy_kitchen codebook quantization.
"""

import pytest
import torch

from convert_to_quant.converters.w4a8_int8_converter import (
    HAS_COMFY_KITCHEN,
    LearnedW4A8Int8Converter,
    W4A8Int8Converter,
)


def test_w4a8_converter_correctness():
    if not HAS_COMFY_KITCHEN:
        print("comfy_kitchen not installed")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    M, N = 3072, 3072
    W_dev = torch.randn(M, N, dtype=torch.float32, device=device) * 0.02
    X = torch.randn(128, N, dtype=torch.float32, device=device)
    Y_ref = X @ W_dev.T

    # 1. Test basic W4A8Int8Converter
    basic_converter = W4A8Int8Converter(group_size=16, convrot_groupsize=256)
    qdata, s_rel, s_channel, correction, codebook, dequantized, extra = basic_converter.convert(W_dev)

    mse_basic = torch.nn.functional.mse_loss(X @ dequantized.T, Y_ref).item()
    print(f"W4A8Int8Converter baseline MSE: {mse_basic:.6e}")

    # 2. Test LearnedW4A8Int8Converter
    learned_converter = LearnedW4A8Int8Converter(group_size=16, convrot_groupsize=256, num_iter=100)
    qdata_l, s_rel_l, s_chan_l, corr_l, cb_l, dequant_l, extra_l = learned_converter.convert(W_dev, calibration_data=X)

    mse_learned = torch.nn.functional.mse_loss(X @ dequant_l.T, Y_ref).item()
    print(f"LearnedW4A8Int8Converter MSE:  {mse_learned:.6e}")

    assert mse_learned <= mse_basic * 1.05, f"Learned MSE ({mse_learned:.6e}) should be <= basic MSE ({mse_basic:.6e})"
    print("SUCCESS: W4A8 converter test passed!")


if __name__ == "__main__":
    test_w4a8_converter_correctness()
