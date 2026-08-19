import torch
from convert_to_quant.converters.learned_int4 import LearnedINT4Converter


def test_prodigy_plateau_learned_rounding_int4():
    device = "cpu"
    M, N = 256, 256
    W = torch.randn(M, N, device=device)
    calib = torch.randn(32, N, device=device)

    converter = LearnedINT4Converter(
        optimizer="prodigy",
        lr_schedule="plateau",
        num_iter=20,
        lr=1.0,
        device=device,
        convrot=True,
        convrot_group_size=256,
    )

    qdata, scale, dequantized, extra = converter.convert(W, calibration_data=calib)
    assert qdata is not None
    assert dequantized is not None
    assert dequantized.shape == W.shape
