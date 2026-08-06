"""
Unit tests for scan_and_replace_comfy_quant_metadata (--replace-quant-metadata).
"""

import json
import os
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from convert_to_quant.cli.main import get_parser
from convert_to_quant.formats.format_migration import scan_and_replace_comfy_quant_metadata
from convert_to_quant.utils.comfy_quant import tensor_to_dict


def test_scan_and_replace_metadata_fp8_legacy():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input_model.safetensors")
        output_path = os.path.join(tmpdir, "output_model.safetensors")

        # Create dummy quantized FP8 layer with legacy scale names and non-comfy metadata header
        w = torch.randn((64, 64), dtype=torch.float32).to(torch.float8_e4m3fn)
        sw = torch.tensor(0.01, dtype=torch.float32)
        si = torch.tensor(1.0, dtype=torch.float32)

        tensors = {
            "model.layer1.weight": w,
            "model.layer1.scale_weight": sw,
            "model.layer1.scale_input": si,
            "scaled_fp8": torch.empty((0,)),
            "model.bias": torch.randn(64, dtype=torch.float32),
        }

        metadata = {
            "quantization_config": '{"quant_method": "bitsandbytes"}',
            "quant_method": "fp8",
            "scaled_fp8": "true",
            "format": "legacy_fp8",
            "user_note": "custom model description",
        }

        save_file(tensors, input_path, metadata=metadata)

        # Run scan and replace
        scan_and_replace_comfy_quant_metadata(input_path, output_path)

        assert os.path.exists(output_path)

        with safe_open(output_path, framework="pt", device="cpu") as f:
            res_meta = f.metadata() or {}
            res_keys = set(f.keys())

            # Check header metadata
            assert "quantization_config" not in res_meta
            assert "quant_method" not in res_meta
            assert "scaled_fp8" not in res_meta
            assert "format" not in res_meta
            assert res_meta.get("user_note") == "custom model description"

            assert "_quantization_metadata" in res_meta
            qmeta = json.loads(res_meta["_quantization_metadata"])
            assert qmeta.get("format_version") == "1.0"
            assert "model.layer1" in qmeta.get("layers", {})
            assert qmeta["layers"]["model.layer1"]["format"] == "float8_e4m3fn"

            # Check tensor renaming & comfy_quant creation
            assert "model.layer1.weight" in res_keys
            assert "model.layer1.weight_scale" in res_keys
            assert "model.layer1.input_scale" in res_keys
            assert "model.layer1.scale_weight" not in res_keys
            assert "model.layer1.scale_input" not in res_keys
            assert "scaled_fp8" not in res_keys
            assert "model.layer1.comfy_quant" in res_keys

            cq_tensor = f.get_tensor("model.layer1.comfy_quant")
            cq_dict = tensor_to_dict(cq_tensor)
            assert cq_dict.get("format") == "float8_e4m3fn"


def test_scan_and_replace_metadata_int8_blockwise():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input_int8.safetensors")
        output_path = os.path.join(tmpdir, "output_int8.safetensors")

        M, N = 128, 128
        block_size = 64
        qweight = torch.randint(-127, 127, (M, N), dtype=torch.int8)
        wscale = torch.ones((M // block_size, N // block_size), dtype=torch.float32) * 0.02

        tensors = {
            "transformer.blocks.0.attn.weight": qweight,
            "transformer.blocks.0.attn.scale_weight": wscale,
        }

        save_file(tensors, input_path)

        scan_and_replace_comfy_quant_metadata(input_path, output_path, default_block_size=64)

        with safe_open(output_path, framework="pt", device="cpu") as f:
            res_meta = f.metadata() or {}
            assert "_quantization_metadata" in res_meta

            qmeta = json.loads(res_meta["_quantization_metadata"])
            layer_meta = qmeta.get("layers", {}).get("transformer.blocks.0.attn")
            assert layer_meta is not None
            assert layer_meta.get("format") == "int8_blockwise"
            assert layer_meta.get("group_size") == 64

            cq_tensor = f.get_tensor("transformer.blocks.0.attn.comfy_quant")
            cq_dict = tensor_to_dict(cq_tensor)
            assert cq_dict.get("format") == "int8_blockwise"
            assert cq_dict.get("group_size") == 64


def test_scan_and_replace_metadata_int4_convrot():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input_int4.safetensors")
        output_path = os.path.join(tmpdir, "output_int4.safetensors")

        out_f, in_f = 64, 128
        qweight = torch.randint(-127, 127, (out_f, in_f // 2), dtype=torch.int8)
        wscale = torch.ones((out_f,), dtype=torch.float32) * 0.05

        tensors = {
            "model.layer_conv.weight": qweight,
            "model.layer_conv.scale_weight": wscale,
        }

        metadata = {
            "quant_method": "INT4_CONVROT",
        }

        save_file(tensors, input_path, metadata=metadata)

        scan_and_replace_comfy_quant_metadata(input_path, output_path, int4=True)

        with safe_open(output_path, framework="pt", device="cpu") as f:
            res_meta = f.metadata() or {}
            assert "_quantization_metadata" in res_meta

            qmeta = json.loads(res_meta["_quantization_metadata"])
            layer_meta = qmeta.get("layers", {}).get("model.layer_conv")
            assert layer_meta is not None
            assert layer_meta.get("format") == "convrot_w4a4"
            assert layer_meta.get("group_size") == 64
            assert layer_meta.get("convrot") is True
            assert layer_meta.get("convrot_groupsize") == 256

            cq_tensor = f.get_tensor("model.layer_conv.comfy_quant")
            cq_dict = tensor_to_dict(cq_tensor)
            assert cq_dict.get("format") == "convrot_w4a4"
            assert cq_dict.get("group_size") == 64
            assert cq_dict.get("convrot") is True
            assert cq_dict.get("convrot_groupsize") == 256


def test_scan_and_replace_cli_parser():
    parser = get_parser()
    args = parser.parse_args(["-i", "input.safetensors", "--replace-quant-metadata"])
    assert args.replace_quant_metadata is True

    args_alias = parser.parse_args(["-i", "input.safetensors", "--replace_quant_metadata"])
    assert args_alias.replace_quant_metadata is True

