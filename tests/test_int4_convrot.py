import torch
import pytest
from convert_to_quant.utils.convrot import (
    pack_int4_row_major,
    unpack_int4_row_major,
    quantize_convrot_w4a4_weight,
    dequantize_convrot_w4a4_weight,
)
from convert_to_quant.comfy.int4_kernels import convrot_w4a4_linear
from convert_to_quant.utils.comfy_quant import create_comfy_quant_tensor, tensor_to_dict
from convert_to_quant.converters.learned_int4 import LearnedINT4Converter


def test_pack_unpack_int4():
    # Generate test values in [-7, 7]
    original = torch.tensor([
        [0, 1, -1, 7, -7, 3, -4, 2],
        [-5, 6, -2, 4, 0, -3, 5, -6]
    ], dtype=torch.int8)

    packed = pack_int4_row_major(original)
    assert packed.shape == (2, 4)
    assert packed.dtype == torch.int8

    unpacked = unpack_int4_row_major(packed)
    assert unpacked.shape == original.shape
    assert torch.equal(unpacked, original)


def test_quantize_dequantize_convrot_w4a4_weight():
    weight = torch.randn(16, 256, dtype=torch.float32)
    qdata, scales = quantize_convrot_w4a4_weight(weight, convrot_groupsize=256, quant_group_size=64)

    assert qdata.shape == (16, 128)
    assert qdata.dtype == torch.int8
    assert scales.shape == (16,)
    assert scales.dtype == torch.float32

    dequantized = dequantize_convrot_w4a4_weight(qdata, scales, convrot_groupsize=256, quant_group_size=64, output_dtype=torch.float32)
    assert dequantized.shape == weight.shape

    # Check relative error bound
    diff = (dequantized - weight).abs().mean()
    assert diff < 0.5


def test_convrot_w4a4_linear():
    x = torch.randn(4, 256, dtype=torch.float32)
    weight = torch.randn(16, 256, dtype=torch.float32)
    bias = torch.randn(16, dtype=torch.float32)

    qdata, scales = quantize_convrot_w4a4_weight(weight, convrot_groupsize=256, quant_group_size=64)

    out = convrot_w4a4_linear(x, qdata, scales, bias=bias, convrot_groupsize=256, quant_group_size=64)
    assert out.shape == (4, 16)

    # Output baseline check
    ref_out = x @ weight.t() + bias
    assert out.shape == ref_out.shape


def test_create_comfy_quant_tensor_convrot_w4a4():
    t = create_comfy_quant_tensor("convrot_w4a4", block_size=64, convrot=True, convrot_groupsize=256)
    config = tensor_to_dict(t)
    assert config["format"] == "convrot_w4a4"
    assert config["group_size"] == 64
    assert config["convrot"] is True
    assert config["convrot_groupsize"] == 256


def test_learned_int4_converter():
    converter = LearnedINT4Converter(device="cpu", no_learned_rounding=True)
    weight = torch.randn(16, 256, dtype=torch.float32)

    qdata, scale, dequantized, extra = converter.convert(weight)
    assert qdata.shape == (16, 128)
    assert scale.shape == (16,)
    assert dequantized.shape == weight.shape


def test_learned_int4_converter_with_optimization():
    converter = LearnedINT4Converter(
        device="cpu",
        no_learned_rounding=False,
        num_iter=5,
        optimizer="prodigy",
        lr_schedule="plateau",
        convrot=True,
        convrot_group_size=256,
    )
    weight = torch.randn(16, 256, dtype=torch.float32)
    calib = torch.randn(8, 256, dtype=torch.float32)

    qdata, scale, dequantized, extra = converter.convert(weight, calibration_data=calib)
    assert qdata.shape == (16, 128)
    assert scale.shape == (16,)
    assert dequantized.shape == weight.shape


def test_convert_to_fp8_scaled_int4_comfy_quant(tmp_path):
    from safetensors.torch import save_file, safe_open
    from convert_to_quant.formats.fp8_conversion import convert_to_fp8_scaled

    in_file = str(tmp_path / "model.safetensors")
    out_file = str(tmp_path / "model_int4.safetensors")

    weights = {
        "layer1.weight": torch.randn(256, 256, dtype=torch.float32),
        "layer1.bias": torch.randn(256, dtype=torch.float32),
    }
    save_file(weights, in_file)

    convert_to_fp8_scaled(
        input_file=in_file,
        output_file=out_file,
        comfy_quant=True,
        filter_flags={},
        calib_samples=1,
        seed=42,
        primary_format="int4",
        no_learned_rounding=True,
    )

    with safe_open(out_file, framework="pt", device="cpu") as f:
        tensor_keys = list(f.keys())
        assert "layer1.comfy_quant" in tensor_keys
        cq_tensor = f.get_tensor("layer1.comfy_quant")
        cq_dict = tensor_to_dict(cq_tensor)
        assert cq_dict["format"] == "convrot_w4a4"

def test_w4a4_adaround_quantized_activations():
    """Verify that INT4 AdaRound optimization and bias correction account for quantized activations."""
    torch.manual_seed(42)
    converter = LearnedINT4Converter(
        device="cpu",
        no_learned_rounding=False,
        num_iter=10,
        optimizer="adamw",
        lr=0.5,
        convrot=True,
        convrot_group_size=256,
    )
    weight = torch.randn(16, 256, dtype=torch.float32)
    bias = torch.randn(16, dtype=torch.float32)
    calib = torch.randn(32, 256, dtype=torch.float32)

    qdata, scale, dequantized, extra = converter.convert(weight, bias=bias, calibration_data=calib)
    assert qdata.shape == (16, 128)
    assert scale.shape == (16,)
    assert "bias_correction" in extra

    # Evaluate using actual W4A4 linear operator (which quantizes activations to INT4)
    out_quant = convrot_w4a4_linear(calib, qdata, scale, bias=bias + extra["bias_correction"].to(calib.device))
    ref_out = calib @ weight.t() + bias
    mse_opt = torch.nn.functional.mse_loss(out_quant, ref_out).item()

    # Compare with simple rounding (no learned rounding)
    converter_simple = LearnedINT4Converter(
        device="cpu",
        no_learned_rounding=True,
        convrot=True,
        convrot_group_size=256,
    )
    qdata_simple, scale_simple, _, extra_simple = converter_simple.convert(weight, bias=bias, calibration_data=calib)
    bias_simple = bias + extra_simple["bias_correction"].to(calib.device) if "bias_correction" in extra_simple else bias
    out_simple = convrot_w4a4_linear(calib, qdata_simple, scale_simple, bias=bias_simple)
    mse_simple = torch.nn.functional.mse_loss(out_simple, ref_out).item()

    assert mse_opt <= mse_simple



