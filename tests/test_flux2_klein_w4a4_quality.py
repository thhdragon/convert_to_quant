"""
Real-world INT4 ConvRot W4A4 Quantization Quality Benchmark

Evaluates quantization accuracy (MSE, SNR dB, Cosine Similarity) on real layers
from /tests/real_world_test_models/flux-2-klein-4b_bf16.safetensors.

Compares:
  1. Standard W4A4 (No ConvRot, Simple Rounding)
  2. ConvRot W4A4 (Simple Rounding)
  3. ConvRot W4A4 + SVD-guided AdaRound
  4. ConvRot W4A4 + SVD AdaRound + Bias Correction
  5. ConvRot W4A4 + Smooth-ConvRot Pre-Scaling + AdaRound + Bias Correction
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open

# Add repository root to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
from convert_to_quant.utils.convrot import (
    build_hadamard,
    rotate_activation,
    rotate_weight,
    balance_channels_smoothquant,
    pack_int4_row_major,
    unpack_int4_row_major,
)
from convert_to_quant.comfy.int4_kernels import (
    convrot_w4a4_linear,
    quantize_signed_int4_rowwise,
)

def compute_metrics(y_ref: torch.Tensor, y_quant: torch.Tensor) -> dict:
    """Compute detailed accuracy error metrics between reference and quantized outputs."""
    y_ref = y_ref.to(torch.float32)
    y_quant = y_quant.to(torch.float32)

    diff = y_ref - y_quant
    mae = torch.mean(torch.abs(diff)).item()
    mse = torch.mean(diff ** 2).item()
    rmse = np.sqrt(mse)

    ref_norm = torch.norm(y_ref).item()
    err_norm = torch.norm(diff).item()

    if err_norm < 1e-12:
        snr_db = float("inf")
    elif ref_norm < 1e-12:
        snr_db = 0.0
    else:
        snr_db = 20 * np.log10(ref_norm / err_norm)

    if ref_norm > 1e-10 and torch.norm(y_quant).item() > 1e-10:
        cos_sim = torch.nn.functional.cosine_similarity(y_ref.flatten(), y_quant.flatten(), dim=0).item()
    else:
        cos_sim = 1.0 if (ref_norm < 1e-10 and torch.norm(y_quant).item() < 1e-10) else 0.0

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "SNR (dB)": snr_db,
        "CosSim": cos_sim,
    }


def quantize_w4a4_baseline(W: torch.Tensor, X: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Method 1: Naive row-wise INT4 W4A4 without ConvRot or AdaRound."""
    M, N = W.shape
    scale_w = (W.abs().amax(dim=1, keepdim=True).clamp_min(1e-10) / 7.0)
    q_w = (W / scale_w).round().clamp(-7, 7)
    W_dequant = q_w * scale_w

    absmax_x = X.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale_x = absmax_x / 7.0
    q_x = (X / scale_x).round().clamp(-7, 7)
    X_dequant = q_x * scale_x

    Y = X_dequant @ W_dequant.T
    if bias is not None:
        Y = Y + bias
    return Y


def quantize_w4a4_convrot_simple(W: torch.Tensor, X: torch.Tensor, group_size: int = 256, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Method 2: ConvRot W4A4 with simple round-to-nearest."""
    M, N = W.shape
    H = build_hadamard(group_size, device=W.device, dtype=W.dtype)
    W_rot = rotate_weight(W, H, group_size)
    X_rot = rotate_activation(X, H, group_size)

    scale_w = (W_rot.abs().amax(dim=1, keepdim=True).clamp_min(1e-10) / 7.0)
    q_w = (W_rot / scale_w).round().clamp(-7, 7)
    W_dequant_rot = q_w * scale_w

    absmax_x = X_rot.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale_x = absmax_x / 7.0
    q_x = (X_rot / scale_x).round().clamp(-7, 7)
    X_dequant_rot = q_x * scale_x

    Y = X_dequant_rot @ W_dequant_rot.T
    if bias is not None:
        Y = Y + bias
    return Y


from convert_to_quant.constants import FLUX_KLEIN_LAYER_KEYNAMES

def run_evaluation(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", num_samples: int = 256, num_layers: int = 5):
    print("=" * 80)
    print(f"INT4 ConvRot W4A4 Quality Benchmark on Flux 2 Klein 4B")
    print(f"Model Path: {model_path}")
    print(f"Device: {device}")
    print("=" * 80)

    if not Path(model_path).exists():
        print(f"Error: Model path '{model_path}' not found.")
        return

    with safe_open(model_path, framework="pt", device="cpu") as sf:
        all_keys = list(sf.keys())
        # Filter out high-precision blacklist layers for FLUX Klein (they are kept in FP/BF16 in production)
        weight_keys = [
            k for k in all_keys
            if k.endswith(".weight")
            and len(sf.get_slice(k).get_shape()) == 2
            and not any(ex in k for ex in FLUX_KLEIN_LAYER_KEYNAMES)
        ]

        print(f"Total 2D weight tensors found (excluding FLUX Klein high-precision blacklist): {len(weight_keys)}")

        # Select representative transformer block layers (Attention Projections and MLP feed-forward)
        target_keys = []
        for category in ["double_blocks.0.img_attn", "double_blocks.0.txt_attn", "single_blocks.0.linear1", "single_blocks.0.linear2", "double_blocks.1.img_attn"]:
            matches = [k for k in weight_keys if category in k]
            if matches:
                target_keys.extend(matches[:1])

        if len(target_keys) < num_layers:
            remaining = [k for k in weight_keys if k not in target_keys]
            target_keys.extend(remaining[: num_layers - len(target_keys)])

        target_keys = target_keys[:num_layers]

        target_keys = target_keys[:num_layers]

        print(f"Selected {len(target_keys)} benchmark layers:")
        for k in target_keys:
            shape = sf.get_slice(k).get_shape()
            print(f"  - {k} (shape: {shape})")
        print("-" * 80)

        results = []

        for key in target_keys:
            W = sf.get_tensor(key).to(device=device, dtype=torch.float32)
            out_f, in_f = W.shape

            base_name = key[:-7]
            bias_key = f"{base_name}.bias"
            bias = sf.get_tensor(bias_key).to(device=device, dtype=torch.float32) if bias_key in all_keys else None

            group_size = 256
            if in_f < group_size or in_f % group_size != 0:
                print(f"Skipping {key}: input dim {in_f} not compatible with ConvRot group_size {group_size}")
                continue

            torch.manual_seed(42)
            X = torch.randn(num_samples, in_f, device=device, dtype=torch.float32)
            Y_ref = X @ W.T
            if bias is not None:
                Y_ref = Y_ref + bias

            # -----------------------------------------------------------------
            # 1. Baseline W4A4 (No ConvRot)
            # -----------------------------------------------------------------
            Y_m1 = quantize_w4a4_baseline(W, X, bias)
            m1 = compute_metrics(Y_ref, Y_m1)

            # -----------------------------------------------------------------
            # 2. ConvRot W4A4 (Simple Rounding)
            # -----------------------------------------------------------------
            Y_m2 = quantize_w4a4_convrot_simple(W, X, group_size, bias)
            m2 = compute_metrics(Y_ref, Y_m2)

            # -----------------------------------------------------------------
            # 3. ConvRot W4A4 + SVD AdaRound
            # -----------------------------------------------------------------
            conv_adaround = LearnedRoundingConverter(
                target_format="int4",
                scaling_mode="row",
                convrot=True,
                convrot_group_size=group_size,
                num_iter=100,
                device=device,
            )
            qdata_3, scale_3, dequant_w_3, extra_3 = conv_adaround.convert(W, calibration_data=X, has_bias=(bias is not None))
            Y_m3 = convrot_w4a4_linear(X, qdata_3, scale_3, bias=bias, convrot_groupsize=group_size)
            m3 = compute_metrics(Y_ref, Y_m3)

            # -----------------------------------------------------------------
            # 4. ConvRot W4A4 + SVD AdaRound + Bias Correction
            # -----------------------------------------------------------------
            bias_corr = extra_3.get("bias_correction", None)
            bias_4 = (bias + bias_corr.to(device=device)) if (bias is not None and bias_corr is not None) else bias
            if bias is None and bias_corr is not None:
                bias_4 = bias_corr.to(device=device)
            Y_m4 = convrot_w4a4_linear(X, qdata_3, scale_3, bias=bias_4, convrot_groupsize=group_size)
            m4 = compute_metrics(Y_ref, Y_m4)

            # -----------------------------------------------------------------
            # 5. ConvRot W4A4 + Smooth-ConvRot Pre-Scaling + SVD AdaRound + Bias Correction
            # -----------------------------------------------------------------
            W_smooth, s_c = balance_channels_smoothquant(W, X, alpha=0.5)
            X_smooth = X / s_c.unsqueeze(0)
            conv_smooth = LearnedRoundingConverter(
                target_format="int4",
                scaling_mode="row",
                convrot=True,
                convrot_group_size=group_size,
                num_iter=100,
                device=device,
            )
            qdata_5, scale_5, dequant_w_5, extra_5 = conv_smooth.convert(W_smooth, calibration_data=X_smooth, has_bias=(bias is not None))
            bias_corr_5 = extra_5.get("bias_correction", None)
            bias_5 = (bias + bias_corr_5.to(device=device)) if (bias is not None and bias_corr_5 is not None) else bias
            if bias is None and bias_corr_5 is not None:
                bias_5 = bias_corr_5.to(device=device)
            Y_m5 = convrot_w4a4_linear(X_smooth, qdata_5, scale_5, bias=bias_5, convrot_groupsize=group_size)
            m5 = compute_metrics(Y_ref, Y_m5)

            results.append({
                "key": key,
                "shape": (out_f, in_f),
                "Method 1 (Baseline W4A4)": m1,
                "Method 2 (ConvRot Simple)": m2,
                "Method 3 (ConvRot AdaRound)": m3,
                "Method 4 (ConvRot AdaRound + BiasCorr)": m4,
                "Method 5 (SmoothConvRot AdaRound + BiasCorr)": m5,
            })

            print(f"Layer: {key} [{out_f}x{in_f}]")
            print(f"  M1 Baseline W4A4                   : SNR = {m1['SNR (dB)']:.2f} dB | CosSim = {m1['CosSim']:.5f} | MSE = {m1['MSE']:.6f}")
            print(f"  M2 ConvRot Simple                  : SNR = {m2['SNR (dB)']:.2f} dB | CosSim = {m2['CosSim']:.5f} | MSE = {m2['MSE']:.6f}")
            print(f"  M3 ConvRot AdaRound                : SNR = {m3['SNR (dB)']:.2f} dB | CosSim = {m3['CosSim']:.5f} | MSE = {m3['MSE']:.6f}")
            print(f"  M4 ConvRot AdaRound + BiasCorr     : SNR = {m4['SNR (dB)']:.2f} dB | CosSim = {m4['CosSim']:.5f} | MSE = {m4['MSE']:.6f}")
            print(f"  M5 SmoothConvRot AdaRound + BiasCorr: SNR = {m5['SNR (dB)']:.2f} dB | CosSim = {m5['CosSim']:.5f} | MSE = {m5['MSE']:.6f}")
            print("-" * 80)

        if results:
            avg_m1_snr = np.mean([r["Method 1 (Baseline W4A4)"]["SNR (dB)"] for r in results])
            avg_m2_snr = np.mean([r["Method 2 (ConvRot Simple)"]["SNR (dB)"] for r in results])
            avg_m3_snr = np.mean([r["Method 3 (ConvRot AdaRound)"]["SNR (dB)"] for r in results])
            avg_m4_snr = np.mean([r["Method 4 (ConvRot AdaRound + BiasCorr)"]["SNR (dB)"] for r in results])
            avg_m5_snr = np.mean([r["Method 5 (SmoothConvRot AdaRound + BiasCorr)"]["SNR (dB)"] for r in results])

            avg_m1_cs = np.mean([r["Method 1 (Baseline W4A4)"]["CosSim"] for r in results])
            avg_m2_cs = np.mean([r["Method 2 (ConvRot Simple)"]["CosSim"] for r in results])
            avg_m3_cs = np.mean([r["Method 3 (ConvRot AdaRound)"]["CosSim"] for r in results])
            avg_m4_cs = np.mean([r["Method 4 (ConvRot AdaRound + BiasCorr)"]["CosSim"] for r in results])
            avg_m5_cs = np.mean([r["Method 5 (SmoothConvRot AdaRound + BiasCorr)"]["CosSim"] for r in results])

            print("=" * 80)
            print("SUMMARY OF AVERAGE ACCURACY METRICS ACROSS LAYERS:")
            print(f"  Method 1 Baseline W4A4:                    Avg SNR = {avg_m1_snr:.2f} dB | Avg CosSim = {avg_m1_cs:.5f}")
            print(f"  Method 2 ConvRot Simple:                   Avg SNR = {avg_m2_snr:.2f} dB | Avg CosSim = {avg_m2_cs:.5f}")
            print(f"  Method 3 ConvRot AdaRound:                 Avg SNR = {avg_m3_snr:.2f} dB | Avg CosSim = {avg_m3_cs:.5f}")
            print(f"  Method 4 ConvRot AdaRound + BiasCorr:      Avg SNR = {avg_m4_snr:.2f} dB | Avg CosSim = {avg_m4_cs:.5f}")
            print(f"  Method 5 SmoothConvRot AdaRound + BiasCorr: Avg SNR = {avg_m5_snr:.2f} dB | Avg CosSim = {avg_m5_cs:.5f}")
            print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INT4 ConvRot W4A4 Quality Benchmark")
    parser.add_argument("--model", type=str, default="/tests/real_world_test_models/flux-2-klein-4b_bf16.safetensors", help="Path to safetensors model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--num-layers", type=int, default=5, help="Number of benchmark layers to test")
    args = parser.parse_args()

    run_evaluation(args.model, device=args.device, num_layers=args.num_layers)
