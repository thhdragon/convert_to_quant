"""
Unit tests for 4-bit sensitivity detection and automatic BF16 fallback.
"""

import pytest
import torch

from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
from convert_to_quant.formats.int4_convrot_conversion import convert_to_int4_convrot


def test_learned_rounding_sensitivity_fallback_detection():
    """Verify LearnedRoundingConverter flags layers with low SNR or high error."""
    torch.manual_seed(42)
    device = "cpu"

    # Create a converter with high SNR requirement (30 dB) to force fallback
    converter = LearnedRoundingConverter(
        target_format="int4",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=256,
        min_snr_db=30.0,  # Standard INT4 is usually 16-25 dB, so 30 dB will trigger fallback
        device=device,
        num_iter=0,
    )

    out_features, in_features = 256, 256
    W = torch.randn(out_features, in_features, device=device)
    X = torch.randn(128, in_features, device=device)

    qdata, scale, dequant_w, extra = converter.convert(W, calibration_data=X)

    assert "metrics" in extra
    assert "snr_db" in extra["metrics"]
    assert "cos_sim" in extra["metrics"]
    assert "fallback" in extra
    assert extra["fallback"]["should_fallback"] is True
    assert len(extra["fallback"]["reasons"]) > 0


def test_learned_rounding_passes_when_snr_is_acceptable():
    """Verify LearnedRoundingConverter does not trigger fallback when threshold is met."""
    torch.manual_seed(42)
    device = "cpu"

    # Create a converter with low SNR threshold (5 dB)
    converter = LearnedRoundingConverter(
        target_format="int4",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=256,
        min_snr_db=5.0,
        device=device,
        num_iter=0,
    )

    out_features, in_features = 256, 256
    W = torch.randn(out_features, in_features, device=device)
    X = torch.randn(128, in_features, device=device)

    qdata, scale, dequant_w, extra = converter.convert(W, calibration_data=X)

    assert "metrics" in extra
    assert extra["metrics"]["snr_db"] > 5.0
    assert "fallback" not in extra or extra["fallback"]["should_fallback"] is False
