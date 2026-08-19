"""
Unit tests for custom layers argument options (--custom-fpmm, --custom-convrot) and has_bias optimization for INT4 ConvRot W4A4.
"""

import os
import unittest

import torch
from safetensors.torch import load_file, save_file

from convert_to_quant.formats.int4_convrot_conversion import convert_to_int4_convrot
from convert_to_quant.utils.comfy_quant import tensor_to_dict


class TestCustomLayersAndBiasOpt(unittest.TestCase):
    def setUp(self):
        self.input_path = "test_custom_opt_input.safetensors"
        self.output_path = "test_custom_opt_output.safetensors"

        self.tensors = {
            "blocks.0.attn.wq.weight": torch.randn(256, 256, dtype=torch.float16),
            "blocks.0.attn.wq.bias": torch.randn(256, dtype=torch.float16),
            "blocks.0.attn.wk.weight": torch.randn(256, 256, dtype=torch.float16),
            "blocks.0.mlp.down.weight": torch.randn(256, 256, dtype=torch.float16),
        }
        save_file(self.tensors, self.input_path)

    def tearDown(self):
        for path in [self.input_path, self.output_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_custom_fpmm_and_has_bias(self):
        convert_to_int4_convrot(
            input_file=self.input_path,
            output_file=self.output_path,
            comfy_quant=True,
            filter_flags={},
            calib_samples=100,
            seed=42,
            no_learned_rounding=True,
            custom_layers="(blocks.0.mlp.down|blocks.0.attn.wk)",
            custom_full_precision_mm=True,
            device="cpu",
        )

        out_tensors = load_file(self.output_path)

        self.assertIn("blocks.0.mlp.down.comfy_quant", out_tensors)
        comfy_quant_down = tensor_to_dict(out_tensors["blocks.0.mlp.down.comfy_quant"])
        self.assertTrue(comfy_quant_down.get("full_precision_matrix_mult", False))

        self.assertIn("blocks.0.attn.wk.comfy_quant", out_tensors)
        comfy_quant_wk = tensor_to_dict(out_tensors["blocks.0.attn.wk.comfy_quant"])
        self.assertTrue(comfy_quant_wk.get("full_precision_matrix_mult", False))

        self.assertIn("blocks.0.attn.wq.comfy_quant", out_tensors)
        comfy_quant_wq = tensor_to_dict(out_tensors["blocks.0.attn.wq.comfy_quant"])
        self.assertFalse(comfy_quant_wq.get("full_precision_matrix_mult", False))


if __name__ == "__main__":
    unittest.main()
