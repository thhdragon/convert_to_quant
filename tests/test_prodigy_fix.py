import torch
from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
from convert_to_quant.converters.w4a8_int8_converter import LearnedW4A8Int8Converter

def test_prodigy_plateau_learned_rounding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M, N = 256, 256
    W = torch.randn(M, N, device=device)

    converter = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=False,
        optimizer="prodigy",
        lr_schedule="plateau",
        num_iter=40,
        lr=1.0,
        device=device
    )

    qdata, scale, dequantized, extra = converter.convert(W)
    assert qdata is not None
    assert dequantized is not None
    assert dequantized.shape == W.shape

def test_prodigy_plateau_w4a8():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M, N = 256, 256
    W = torch.randn(M, N, device=device)
    X = torch.randn(32, N, device=device)

    converter = LearnedW4A8Int8Converter(
        optimizer="prodigy",
        lr_schedule="plateau",
        num_iter=40,
        lr=1.0,
        device=device
    )

    qdata, s_rel, s_channel, correction, codebook_tensor, dequantized, extra = converter.convert(W, calibration_data=X)
    assert qdata is not None
    assert dequantized is not None
    assert dequantized.shape == W.shape
