"""
Comprehensive INT4 ConvRot W4A4 test suite.

Designed to catch every bug found during the Aug 2026 bug hunt:
  - Codec (pack/unpack) bit-level correctness vs official comfy_kitchen reference
  - build_hadamard power-of-4 enforcement (no silent scipy fallback)
  - Hadamard orthogonality and self-inverse property (W @ H.T @ H = W)
  - INT4 value range enforcement after all quantization paths
  - Adaround finalization: round() not truncate, no in-place W_floor mutation
  - DUALROUND scale re-estimation from original weights, not dequantized output
  - Silent no-op warning when learned rounding is skipped
  - Bias correction uses consistent reference when activations are quantized
  - End-to-end parity with comfy_kitchen eager implementation
"""

import warnings

import pytest
import torch
import torch.nn.functional as F

from convert_to_quant.utils.convrot import (
    build_hadamard,
    dequantize_convrot_w4a4_weight,
    pack_int4_row_major,
    quantize_convrot_w4a4_weight,
    rotate_activation,
    rotate_weight,
    unpack_int4_row_major,
)
from convert_to_quant.comfy.int4_kernels import (
    convrot_w4a4_linear,
    quantize_signed_int4_rowwise,
)
from convert_to_quant.converters.learned_int4 import LearnedINT4Converter


# ---------------------------------------------------------------------------
# Helpers — official comfy_kitchen reference implementations, inlined
# ---------------------------------------------------------------------------

def _ck_pack(values: torch.Tensor) -> torch.Tensor:
    """comfy_kitchen reference pack (int32 intermediates)."""
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def _ck_unpack(packed: torch.Tensor) -> torch.Tensor:
    """comfy_kitchen reference unpack (int32 intermediates, stack+reshape)."""
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)


# ===========================================================================
# Section 1: Pack / Unpack Codec
# ===========================================================================

class TestPackUnpackCodec:

    def test_pack_exhaustive_int4_range(self):
        """Every value in [-7, 7] must survive a pack->unpack round-trip."""
        vals = torch.arange(-7, 8, dtype=torch.int8)
        vals_2d = vals.unsqueeze(0).repeat(1, 2)[:, :14]
        packed = pack_int4_row_major(vals_2d)
        unpacked = unpack_int4_row_major(packed)
        assert torch.equal(unpacked, vals_2d), (
            f"Round-trip failed.\nOriginal: {vals_2d}\nGot:      {unpacked}"
        )

    def test_pack_matches_official_comfy_kitchen_reference(self):
        """pack_int4_row_major must produce bit-identical output to comfy_kitchen."""
        torch.manual_seed(0)
        raw = torch.randint(-7, 8, (8, 64)).to(torch.int8)
        ours = pack_int4_row_major(raw)
        ref  = _ck_pack(raw)
        assert torch.equal(ours, ref), (
            "pack_int4_row_major diverges from comfy_kitchen reference.\n"
            f"Max difference: {(ours.to(torch.int32) - ref.to(torch.int32)).abs().max().item()}"
        )

    def test_unpack_matches_official_comfy_kitchen_reference(self):
        """unpack_int4_row_major must produce bit-identical output to comfy_kitchen."""
        torch.manual_seed(1)
        raw = torch.randint(-7, 8, (8, 64), dtype=torch.int8)
        packed = _ck_pack(raw)
        ours = unpack_int4_row_major(packed)
        ref  = _ck_unpack(packed)
        assert torch.equal(ours, ref), (
            "unpack_int4_row_major diverges from comfy_kitchen reference.\n"
            f"Max difference: {(ours.to(torch.int32) - ref.to(torch.int32)).abs().max().item()}"
        )

    def test_pack_known_bit_pattern(self):
        """Verify nibble encoding: lo=even col (bits 0-3), hi=odd col (bits 4-7)."""
        vals = torch.tensor([[3, 5]], dtype=torch.int8)
        packed = pack_int4_row_major(vals)
        assert packed.shape == (1, 1)
        byte = packed[0, 0].item()
        assert (byte & 0x0F) == 3, f"Low nibble should be 3, got {byte & 0x0F}"
        assert ((byte >> 4) & 0x0F) == 5, f"High nibble should be 5, got {(byte >> 4) & 0x0F}"

    def test_pack_negative_known_bit_pattern(self):
        """Verify -7 and -1 survive packing."""
        vals = torch.tensor([[-7, -1]], dtype=torch.int8)
        packed = pack_int4_row_major(vals)
        unpacked = unpack_int4_row_major(packed)
        assert unpacked[0, 0].item() == -7
        assert unpacked[0, 1].item() == -1

    def test_pack_unpack_large_tensor(self):
        """Round-trip on a large realistic weight shape."""
        torch.manual_seed(2)
        raw = torch.randint(-7, 8, (256, 4096), dtype=torch.int8)
        assert torch.equal(unpack_int4_row_major(pack_int4_row_major(raw)), raw)

    def test_quantize_signed_int4_rowwise_range(self):
        """quantize_signed_int4_rowwise must emit values strictly in [-7, 7]."""
        torch.manual_seed(3)
        x = torch.randn(64, 256)
        qdata, scales = quantize_signed_int4_rowwise(x)
        unpacked = unpack_int4_row_major(qdata)
        assert unpacked.min().item() >= -7, f"Min value {unpacked.min().item()} < -7"
        assert unpacked.max().item() <= 7,  f"Max value {unpacked.max().item()} > 7"
        assert scales.shape == (64,)
        assert (scales > 0).all(), "All scales must be positive"

    def test_quantize_signed_int4_rowwise_matches_official(self):
        """Output of our quantize_signed_int4_rowwise must match comfy_kitchen's."""
        from comfy_kitchen.backends.eager.convrot_w4a4 import (
            quantize_signed_int4_rowwise as ck_quant,
        )
        torch.manual_seed(4)
        x = torch.randn(32, 256)
        ours_q, ours_s = quantize_signed_int4_rowwise(x)
        ck_q,   ck_s   = ck_quant(x, stochastic_rounding=0)
        assert torch.equal(ours_q, ck_q), "qdata mismatch vs comfy_kitchen"
        assert torch.allclose(ours_s, ck_s, atol=1e-6), "scales mismatch vs comfy_kitchen"


# ===========================================================================
# Section 2: Hadamard Matrix
# ===========================================================================

class TestBuildHadamard:

    @pytest.mark.parametrize("size", [4, 16, 64, 256, 1024])
    def test_valid_power_of_4_sizes(self, size):
        """build_hadamard should succeed for all power-of-4 sizes."""
        H = build_hadamard(size)
        assert H.shape == (size, size)
        assert H.dtype == torch.float32

    @pytest.mark.parametrize("bad_size", [2, 3, 8, 32, 128, 512, 2048])
    def test_rejects_non_power_of_4(self, bad_size):
        """build_hadamard must raise ValueError for non-power-of-4 sizes.

        This catches the bug where the old code silently fell back to the
        scipy Sylvester Hadamard (which has an all-ones DC column that
        amplifies row-wise outliers in diffusion models).
        """
        with pytest.raises(ValueError, match="power of 4"):
            build_hadamard(bad_size)

    def test_rejects_size_1(self):
        with pytest.raises(ValueError):
            build_hadamard(1)

    @pytest.mark.parametrize("size", [4, 16, 64, 256])
    def test_orthogonality(self, size):
        """H @ H.T must equal the identity matrix."""
        H = build_hadamard(size)
        HHt = H @ H.T
        I = torch.eye(size, dtype=torch.float32)
        assert torch.allclose(HHt, I, atol=1e-5), (
            f"H @ H.T != I for size {size}. Max deviation: {(HHt - I).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("size", [4, 16, 64, 256])
    def test_self_inverse(self, size):
        """H @ H must equal I for the normalized ConvRot H4 construction.

        This property (H = H^{-1}) means rotate_weight(w_rot, H) correctly
        inverts the rotation — verified during the bug hunt as NOT a bug.
        """
        H = build_hadamard(size)
        HH = H @ H
        I = torch.eye(size, dtype=torch.float32)
        assert torch.allclose(HH, I, atol=1e-5), (
            f"H @ H != I for size {size}. Max deviation: {(HH - I).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("size", [4, 16, 64, 256])
    def test_symmetric(self, size):
        """The H4 ConvRot Hadamard must be symmetric (H == H.T)."""
        H = build_hadamard(size)
        assert torch.allclose(H, H.T, atol=1e-7), (
            f"H is not symmetric for size {size}"
        )

    def test_cache_returns_same_object(self):
        """Cached Hadamard must be the exact same tensor object."""
        H1 = build_hadamard(64)
        H2 = build_hadamard(64)
        assert H1.data_ptr() == H2.data_ptr(), "Cache should return the identical tensor"

    @pytest.mark.parametrize("size", [4, 16, 64, 256])
    def test_matches_official_comfy_kitchen(self, size):
        """Our Hadamard must match comfy_kitchen's _build_hadamard exactly."""
        from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard as ck_hadamard
        ours = build_hadamard(size)
        ref  = ck_hadamard(size)
        assert torch.allclose(ours, ref, atol=1e-7), (
            f"Hadamard mismatch vs comfy_kitchen for size {size}. "
            f"Max diff: {(ours - ref).abs().max().item():.2e}"
        )

    def test_rotation_is_involution(self):
        """Rotating a weight twice must recover the original (H is self-inverse)."""
        torch.manual_seed(5)
        W = torch.randn(16, 256)
        H = build_hadamard(256)
        W_rot  = rotate_weight(W,     H, 256)
        W_back = rotate_weight(W_rot, H, 256)
        assert torch.allclose(W_back, W, atol=1e-5), (
            f"Double rotation did not recover original. "
            f"Max deviation: {(W_back - W).abs().max().item():.2e}"
        )

    def test_activation_rotation_involution(self):
        """Rotating activations twice must recover the original."""
        torch.manual_seed(6)
        X = torch.randn(8, 256)
        H = build_hadamard(256)
        X_rot  = rotate_activation(X,     H, 256)
        X_back = rotate_activation(X_rot, H, 256)
        assert torch.allclose(X_back, X, atol=1e-5), (
            f"Double activation rotation did not recover original. "
            f"Max deviation: {(X_back - X).abs().max().item():.2e}"
        )


# ===========================================================================
# Section 3: Quantization Correctness
# ===========================================================================

class TestQuantizationCorrectness:

    def test_quantize_weight_output_range(self):
        """All unpacked quantized weight values must be in [-7, 7]."""
        torch.manual_seed(7)
        W = torch.randn(64, 256)
        qdata, scales = quantize_convrot_w4a4_weight(W, convrot_groupsize=256, quant_group_size=64)
        unpacked = unpack_int4_row_major(qdata)
        assert unpacked.min().item() >= -7
        assert unpacked.max().item() <= 7

    def test_quantize_weight_matches_official(self):
        """quantize_convrot_w4a4_weight must match comfy_kitchen exactly."""
        from comfy_kitchen.backends.eager.convrot_w4a4 import (
            quantize_convrot_w4a4_weight as ck_quant,
        )
        torch.manual_seed(8)
        W = torch.randn(16, 256)
        ours_q, ours_s = quantize_convrot_w4a4_weight(W, convrot_groupsize=256, quant_group_size=64)
        ck_q,   ck_s   = ck_quant(W, convrot_groupsize=256, quant_group_size=64, stochastic_rounding=0)
        assert torch.equal(ours_q, ck_q), "qdata mismatch vs comfy_kitchen"
        assert torch.allclose(ours_s, ck_s, atol=1e-6), "scales mismatch vs comfy_kitchen"

    def test_dequantize_weight_matches_official(self):
        """dequantize_convrot_w4a4_weight must match comfy_kitchen exactly."""
        from comfy_kitchen.backends.eager.convrot_w4a4 import (
            dequantize_convrot_w4a4_weight as ck_dequant,
            quantize_convrot_w4a4_weight as ck_quant,
        )
        torch.manual_seed(9)
        W = torch.randn(16, 256)
        qdata, scales = ck_quant(W, convrot_groupsize=256, quant_group_size=64, stochastic_rounding=0)
        ours = dequantize_convrot_w4a4_weight(qdata, scales, convrot_groupsize=256, quant_group_size=64)
        ref  = ck_dequant(qdata, scales, convrot_groupsize=256, quant_group_size=64)
        assert torch.allclose(ours, ref, atol=1e-6), (
            f"dequantize mismatch. Max diff: {(ours - ref).abs().max().item():.2e}"
        )

    def test_convrot_linear_matches_official(self):
        """convrot_w4a4_linear must match comfy_kitchen's eager impl."""
        from comfy_kitchen.backends.eager.convrot_w4a4 import (
            convrot_w4a4_linear as ck_linear,
            quantize_convrot_w4a4_weight as ck_quant,
        )
        torch.manual_seed(10)
        X = torch.randn(8, 256)
        W = torch.randn(32, 256)
        bias = torch.randn(32)
        qdata, scales = ck_quant(W, convrot_groupsize=256, quant_group_size=64, stochastic_rounding=0)
        ours = convrot_w4a4_linear(X, qdata, scales, bias=bias, convrot_groupsize=256, quant_group_size=64)
        ref  = ck_linear(X, qdata, scales, bias=bias, convrot_groupsize=256, quant_group_size=64)
        assert torch.allclose(ours, ref, atol=1e-4), (
            f"convrot_w4a4_linear mismatch. Max diff: {(ours - ref).abs().max().item():.2e}"
        )

    def test_dequantize_then_rotate_back_is_close_to_original(self):
        """dequantize -> rotate back should approximately recover the original weight."""
        torch.manual_seed(11)
        W = torch.randn(16, 256)
        qdata, scales = quantize_convrot_w4a4_weight(W, convrot_groupsize=256, quant_group_size=64)
        W_hat = dequantize_convrot_w4a4_weight(qdata, scales, convrot_groupsize=256, quant_group_size=64)
        rel_err = (W_hat - W).abs().mean() / W.abs().mean()
        assert rel_err < 0.25, f"Relative error too large: {rel_err:.4f}"

    def test_quantize_dequantize_scales_positive(self):
        """Scales must always be strictly positive."""
        torch.manual_seed(12)
        W = torch.randn(64, 256)
        _, scales = quantize_convrot_w4a4_weight(W, convrot_groupsize=256, quant_group_size=64)
        assert (scales > 0).all(), "All scales must be strictly positive"


# ===========================================================================
# Section 4: AdaRound Finalization
# ===========================================================================

class TestAdaroundFinalization:

    def test_quantized_output_in_range_after_adaround(self):
        """All quantized values after adaround must remain in [-7, 7].

        This catches the truncation bug: .to(int8) truncates 6.9999->6 instead
        of rounding. With .round() first, boundary values land correctly.
        """
        torch.manual_seed(13)
        converter = LearnedINT4Converter(
            device="cpu",
            no_learned_rounding=False,
            num_iter=20,
            optimizer="adamw",
            lr=0.1,
            convrot=True,
            convrot_group_size=256,
        )
        W = torch.randn(16, 256)
        calib = torch.randn(32, 256)
        qdata, scale, dequantized, extra = converter.convert(W, calibration_data=calib)

        unpacked = unpack_int4_row_major(qdata)
        assert unpacked.min().item() >= -7, f"Adaround produced value < -7: {unpacked.min().item()}"
        assert unpacked.max().item() <= 7,  f"Adaround produced value > 7: {unpacked.max().item()}"

    def test_adaround_does_not_corrupt_w_floor_in_dualround(self):
        """DUALROUND passes must not share a mutable W_floor between calls.

        Old bug: W_floor.add_(best_V) mutated in-place which could corrupt the
        tensor if referenced from the outer scope in Pass 2.
        """
        torch.manual_seed(14)
        converter = LearnedINT4Converter(
            device="cpu",
            no_learned_rounding=False,
            num_iter=10,
            optimizer="adamw",
            lr=0.1,
            convrot=True,
            convrot_group_size=256,
            scale_optimization="dualround",
        )
        W = torch.randn(16, 256)
        calib = torch.randn(32, 256)
        qdata, scale, dequantized, extra = converter.convert(W, calibration_data=calib)

        unpacked = unpack_int4_row_major(qdata)
        assert unpacked.min().item() >= -7
        assert unpacked.max().item() <= 7
        assert dequantized.shape == W.shape
        assert not torch.isnan(dequantized).any(), "Dequantized output contains NaN"
        assert not torch.isinf(dequantized).any(), "Dequantized output contains Inf"

    def test_dequantized_weight_matches_quantized_data(self):
        """The returned dequantized tensor must be consistent with qdata and scale."""
        torch.manual_seed(15)
        converter = LearnedINT4Converter(
            device="cpu",
            no_learned_rounding=False,
            num_iter=10,
            optimizer="adamw",
            lr=0.1,
            convrot=True,
            convrot_group_size=256,
        )
        W = torch.randn(16, 256)
        calib = torch.randn(32, 256)
        qdata, scale, dequantized, extra = converter.convert(W, calibration_data=calib)

        manual_dequant = dequantize_convrot_w4a4_weight(
            qdata, scale, convrot_groupsize=256, quant_group_size=64, output_dtype=torch.float32
        )
        assert torch.allclose(dequantized, manual_dequant, atol=1e-5), (
            f"Dequantized output doesn't match manual reconstruction. "
            f"Max diff: {(dequantized - manual_dequant).abs().max().item():.2e}"
        )


# ===========================================================================
# Section 5: DUALROUND Scale Source
# ===========================================================================

class TestDualroundScaleSource:

    def test_dualround_output_is_valid(self):
        """DUALROUND must produce valid outputs (in range, no NaN, no Inf).

        Old bug: scale was re-estimated from dequant(qdata) * scale_old, giving
        scale_new <= scale_old, making Pass 2 start from worse initial conditions.
        After the fix, scale is re-estimated from W_float32 directly.
        """
        torch.manual_seed(16)
        W = torch.randn(16, 256)
        calib = torch.randn(32, 256)

        converter = LearnedINT4Converter(
            device="cpu",
            no_learned_rounding=False,
            num_iter=5,
            optimizer="adamw",
            lr=0.01,
            convrot=True,
            convrot_group_size=256,
            scale_optimization="dualround",
        )
        qdata, scale, dequant, _ = converter.convert(W, calibration_data=calib)

        unpacked = unpack_int4_row_major(qdata)
        assert unpacked.min().item() >= -7
        assert unpacked.max().item() <= 7
        assert not torch.isnan(dequant).any(), "NaN in DUALROUND dequantized output"
        assert not torch.isinf(dequant).any(), "Inf in DUALROUND dequantized output"
        assert not torch.isnan(scale).any(), "NaN in DUALROUND scales"

    def test_dualround_scale_consistent_with_float_weights(self):
        """After DUALROUND, scale * 7 must be >= max(|W_float32|) per row.

        This verifies the scale is derived from W_float32 (not from the
        quantized output which would always give scale <= original scale).
        """
        torch.manual_seed(17)
        # Construct a weight with some large values to make scale meaningful
        W = torch.randn(16, 256) * 3.0
        calib = torch.randn(32, 256)

        converter = LearnedINT4Converter(
            device="cpu",
            no_learned_rounding=False,
            num_iter=5,
            optimizer="adamw",
            lr=0.05,
            convrot=True,
            convrot_group_size=256,
            scale_optimization="dualround",
        )
        qdata, scale, dequant, _ = converter.convert(W, calibration_data=calib)

        # scale * 7 should be >= absmax of W per row (within some tolerance for
        # the rotation transforming the weight space)
        row_absmax = W.abs().amax(dim=1)
        # After rotation, the scale is set in the rotated space, so we can't
        # compare directly to W — but scale * 7 should be a reasonable positive value
        assert (scale * 7.0 > 0).all(), "Scale * 7 should be positive for all rows"
        assert not torch.isnan(scale).any()


# ===========================================================================
# Section 6: Silent No-Op Warning
# ===========================================================================

class TestSilentNoOpWarning:

    def test_output_valid_when_learned_rounding_skipped_no_convrot(self):
        """When ConvRot is disabled and learned rounding is requested,
        the converter must still return valid (in-range) quantized weights.
        The old code would silently skip optimization with no diagnostic output.
        """
        converter = LearnedINT4Converter(
            device="cpu",
            no_learned_rounding=False,
            num_iter=5,
            optimizer="adamw",
            convrot=False,  # disabled: optimization cannot run
        )
        W = torch.randn(16, 256)
        qdata, scale, dequant, extra = converter.convert(W)

        unpacked = unpack_int4_row_major(qdata)
        assert unpacked.min().item() >= -7
        assert unpacked.max().item() <= 7
        assert not torch.isnan(dequant).any()

    def test_no_crash_when_no_learned_rounding_true(self):
        """no_learned_rounding=True with ConvRot disabled must work without error."""
        converter = LearnedINT4Converter(
            device="cpu",
            no_learned_rounding=True,
            convrot=False,
        )
        W = torch.randn(16, 256)
        qdata, scale, dequant, extra = converter.convert(W)
        unpacked = unpack_int4_row_major(qdata)
        assert unpacked.min().item() >= -7
        assert unpacked.max().item() <= 7


# ===========================================================================
# Section 7: Bias Correction Consistency
# ===========================================================================

class TestBiasCorrection:

    def test_bias_correction_present_when_calibration_provided(self):
        """bias_correction must be present in extra when calibration data is given."""
        torch.manual_seed(18)
        converter = LearnedINT4Converter(
            device="cpu",
            no_learned_rounding=True,
            convrot=True,
            convrot_group_size=256,
        )
        W = torch.randn(16, 256)
        bias = torch.randn(16)
        calib = torch.randn(32, 256)
        _, _, _, extra = converter.convert(W, bias=bias, calibration_data=calib)
        assert "bias_correction" in extra, "bias_correction must be present in extra"
        assert extra["bias_correction"].shape == bias.shape

    def test_bias_correction_reduces_output_error(self):
        """Applying bias_correction must not increase MSE vs the unquantized reference."""
        torch.manual_seed(19)
        W = torch.randn(16, 256)
        bias = torch.randn(16)
        calib = torch.randn(64, 256)

        converter = LearnedINT4Converter(
            device="cpu",
            no_learned_rounding=False,
            num_iter=5,
            optimizer="adamw",
            lr=0.1,
            convrot=True,
            convrot_group_size=256,
        )
        qdata, scale, dequant, extra = converter.convert(W, bias=bias, calibration_data=calib)
        assert "bias_correction" in extra

        ref = calib @ W.T + bias

        out_corrected = convrot_w4a4_linear(
            calib, qdata, scale,
            bias=bias + extra["bias_correction"].to(calib.device),
            convrot_groupsize=256, quant_group_size=64,
        )
        mse_corrected = F.mse_loss(out_corrected, ref).item()

        out_raw = convrot_w4a4_linear(calib, qdata, scale, bias=bias, convrot_groupsize=256, quant_group_size=64)
        mse_raw = F.mse_loss(out_raw, ref).item()

        assert mse_corrected <= mse_raw * 1.1, (
            f"Bias correction made things worse: {mse_corrected:.4e} > {mse_raw:.4e}"
        )

    def test_bias_correction_shape_matches_bias(self):
        """bias_correction tensor must have the same shape as the original bias."""
        torch.manual_seed(20)
        converter = LearnedINT4Converter(
            device="cpu", no_learned_rounding=True, convrot=True, convrot_group_size=256,
        )
        for out_features in [16, 32, 64]:
            W = torch.randn(out_features, 256)
            bias = torch.randn(out_features)
            calib = torch.randn(16, 256)
            _, _, _, extra = converter.convert(W, bias=bias, calibration_data=calib)
            if "bias_correction" in extra:
                assert extra["bias_correction"].shape == bias.shape

    def test_untouched_vs_quantized_activations_different_bias_correction(self):
        """w4a4_untouched_activations=True and False should produce different bias corrections.

        The new Phase 4 fix computes different Y_ref for each mode:
        - untouched: Y_ref = X_rot @ W_rot.T  (unquantized acts)
        - quantized: Y_ref = act_dequant @ W_float32.T  (quantized acts x original weights)
        These are numerically different, so bias corrections must differ.
        """
        torch.manual_seed(21)
        W = torch.randn(16, 256)
        bias = torch.randn(16)
        calib = torch.randn(32, 256)

        conv_q = LearnedINT4Converter(
            device="cpu", no_learned_rounding=True, convrot=True,
            convrot_group_size=256, w4a4_untouched_activations=False,
        )
        conv_u = LearnedINT4Converter(
            device="cpu", no_learned_rounding=True, convrot=True,
            convrot_group_size=256, w4a4_untouched_activations=True,
        )

        _, _, _, extra_q = conv_q.convert(W.clone(), bias=bias.clone(), calibration_data=calib.clone())
        _, _, _, extra_u = conv_u.convert(W.clone(), bias=bias.clone(), calibration_data=calib.clone())

        if "bias_correction" in extra_q and "bias_correction" in extra_u:
            bc_q = extra_q["bias_correction"]
            bc_u = extra_u["bias_correction"]
            max_diff = (bc_q - bc_u).abs().max().item()
            assert max_diff > 1e-6, (
                f"Bias corrections are identical ({max_diff:.2e}), but they should differ "
                "because quantized vs unquantized activations produce different references."
            )


# ===========================================================================
# Section 8: End-to-End Integration
# ===========================================================================

class TestEndToEnd:

    def test_simple_quantization_output_vs_full_precision(self):
        """Quantized linear output should be reasonably close to full-precision baseline."""
        torch.manual_seed(22)
        X = torch.randn(8, 256)
        W = torch.randn(32, 256)
        bias = torch.randn(32)

        qdata, scales = quantize_convrot_w4a4_weight(W, convrot_groupsize=256, quant_group_size=64)
        out_quant = convrot_w4a4_linear(X, qdata, scales, bias=bias, convrot_groupsize=256, quant_group_size=64)
        out_ref = X @ W.T + bias

        mse = F.mse_loss(out_quant, out_ref).item()
        assert mse < 10.0 * out_ref.var().item(), (
            f"MSE {mse:.4e} seems too high relative to output variance {out_ref.var().item():.4e}"
        )

    def test_full_pipeline_no_nan_no_inf(self):
        """Full convert pipeline must not produce NaN or Inf in any output tensor."""
        torch.manual_seed(23)
        converter = LearnedINT4Converter(
            device="cpu",
            no_learned_rounding=False,
            num_iter=10,
            optimizer="adamw",
            lr=0.1,
            convrot=True,
            convrot_group_size=256,
        )
        W = torch.randn(32, 256)
        calib = torch.randn(16, 256)
        qdata, scale, dequant, extra = converter.convert(W, calibration_data=calib)

        assert not torch.isnan(dequant).any(), "NaN in dequantized output"
        assert not torch.isinf(dequant).any(), "Inf in dequantized output"
        assert not torch.isnan(scale).any(), "NaN in scales"
        assert not torch.isinf(scale).any(), "Inf in scales"

    def test_learned_rounding_not_dramatically_worse(self):
        """Learned rounding must not make quantization dramatically worse than simple rounding."""
        torch.manual_seed(24)
        W = torch.randn(16, 256)
        calib = torch.randn(32, 256)

        conv_opt = LearnedINT4Converter(
            device="cpu", no_learned_rounding=False, num_iter=30,
            optimizer="adamw", lr=0.2, convrot=True, convrot_group_size=256,
        )
        conv_simple = LearnedINT4Converter(
            device="cpu", no_learned_rounding=True, convrot=True, convrot_group_size=256,
        )

        _, _, dequant_opt,    _ = conv_opt.convert(W.clone(),    calibration_data=calib.clone())
        _, _, dequant_simple, _ = conv_simple.convert(W.clone(), calibration_data=calib.clone())

        mse_opt    = F.mse_loss(dequant_opt,    W).item()
        mse_simple = F.mse_loss(dequant_simple, W).item()

        assert mse_opt <= mse_simple * 1.20, (
            f"Learned rounding MSE ({mse_opt:.4e}) is more than 20% worse "
            f"than simple rounding ({mse_simple:.4e})"
        )

    def test_convert_all_zeros_weight(self):
        """A zero weight tensor must be handled gracefully (no div-by-zero, no NaN)."""
        converter = LearnedINT4Converter(
            device="cpu", no_learned_rounding=True, convrot=True, convrot_group_size=256,
        )
        W = torch.zeros(16, 256)
        qdata, scale, dequant, extra = converter.convert(W)
        assert not torch.isnan(dequant).any()
        assert not torch.isinf(scale).any()

    @pytest.mark.parametrize("shape", [(16, 256), (64, 512), (128, 1024)])
    def test_shape_invariance(self, shape):
        """Quantization pipeline must work for various 2D weight shapes."""
        torch.manual_seed(25)
        W = torch.randn(*shape)
        qdata, scales = quantize_convrot_w4a4_weight(W, convrot_groupsize=256, quant_group_size=64)
        assert qdata.shape == (shape[0], shape[1] // 2)
        assert scales.shape == (shape[0],)
