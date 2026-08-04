"""
Unit tests for model dequantization functionality (--dequantize).
"""

import tempfile
import os
import torch
from safetensors.torch import save_file, load_file
from convert_to_quant.formats.dequantization import dequantize_model
from convert_to_quant.cli.main import get_parser
from convert_to_quant.utils.comfy_quant import create_comfy_quant_tensor


def test_dequantize_fp8_comfy_quant():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "quant_fp8.safetensors")
        output_path = os.path.join(tmpdir, "dequant_bf16.safetensors")

        # Create dummy FP8 weight and scale
        orig_weight = torch.randn((128, 128), dtype=torch.float32)
        scale = torch.tensor(0.01, dtype=torch.float32)
        qdata = (orig_weight / scale).to(torch.float8_e4m3fn)

        comfy_quant = create_comfy_quant_tensor("float8_e4m3fn")

        tensors = {
            "layer1.weight": qdata,
            "layer1.weight_scale": scale,
            "layer1.comfy_quant": comfy_quant,
            "layer1.bias": torch.randn(128, dtype=torch.float32),
        }
        save_file(tensors, input_path)

        # Run dequantization
        dequantize_model(input_path, output_path, dtype="bf16")

        assert os.path.exists(output_path)
        res = load_file(output_path)

        assert "layer1.weight" in res
        assert res["layer1.weight"].dtype == torch.bfloat16
        assert res["layer1.weight"].shape == (128, 128)

        assert "layer1.bias" in res
        assert res["layer1.bias"].dtype == torch.bfloat16

        # Ensure scale and comfy_quant tensors were removed
        assert "layer1.weight_scale" not in res
        assert "layer1.comfy_quant" not in res


def test_dequantize_int8_blockwise():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "quant_int8.safetensors")
        output_path = os.path.join(tmpdir, "dequant_int8_bf16.safetensors")

        M, N = 128, 128
        block_size = 64
        qdata = torch.randint(-127, 127, (M, N), dtype=torch.int8)
        scale = torch.ones((M // block_size, N // block_size), dtype=torch.float32) * 0.05

        comfy_quant = create_comfy_quant_tensor("int8_blockwise", block_size=block_size)

        tensors = {
            "model.layer.weight": qdata,
            "model.layer.weight_scale": scale,
            "model.layer.comfy_quant": comfy_quant,
        }
        save_file(tensors, input_path)

        dequantize_model(input_path, output_path, dtype="bf16")

        res = load_file(output_path)
        assert "model.layer.weight" in res
        assert res["model.layer.weight"].dtype == torch.bfloat16
        assert res["model.layer.weight"].shape == (M, N)
        assert "model.layer.weight_scale" not in res
        assert "model.layer.comfy_quant" not in res


def test_dequantize_cli_args():
    parser = get_parser()
    args = parser.parse_args(["-i", "dummy.safetensors", "--dequantize"])
    assert args.dequantize is True
    assert args.dequant_dtype == "bf16"

    args_alias = parser.parse_args(["-i", "dummy.safetensors", "--dequantize-to-bf16", "--dequant-dtype", "fp16"])
    assert args_alias.dequantize is True
    assert args_alias.dequant_dtype == "fp16"
