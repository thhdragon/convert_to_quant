import torch
from convert_to_quant.converters.learned_int4 import LearnedINT4Converter


def test_oom_recovery_and_calib_scale_shrinkage(monkeypatch):
    device = "cpu"

    converter = LearnedINT4Converter(
        target_format="int4",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=256,
        num_iter=5,
        device=device,
    )

    original_convert = converter._convert_int4_convrot
    call_count = 0

    def mock_convert(W_float32, calibration_data=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("CUDA out of memory in test")
        else:
            return original_convert(W_float32, calibration_data=calibration_data)

    monkeypatch.setattr(converter, "_convert_int4_convrot", mock_convert)

    W_orig = torch.randn(256, 256, device=device, dtype=torch.float32)
    X = torch.randn(32, 256, device=device, dtype=torch.float32)

    assert converter.calib_scale == 1.0

    qdata, scale, dequant_w, extra_tensors = converter.convert(
        W_orig,
        key="test_layer.weight",
        depth=0,
        calibration_data=X,
    )

    assert call_count == 2
    assert converter.calib_scale == 1.0
    assert qdata.shape == (256, 128)
    assert qdata.dtype == torch.int8
    assert scale.shape == (256,)
