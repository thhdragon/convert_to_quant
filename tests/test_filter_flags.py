"""
Filter flag regression tests for INT4 ConvRot W4A4.
"""

import os
import unittest

import torch
from safetensors.torch import load_file, save_file

from convert_to_quant.cli.main import extract_filter_flags
from convert_to_quant.formats.int4_convrot_conversion import convert_to_int4_convrot


def _is_quantized(tensors: dict, base: str) -> bool:
    return f"{base}.weight_scale" in tensors


def _has_comfy_quant(tensors: dict, base: str) -> bool:
    return f"{base}.comfy_quant" in tensors


def _build_model() -> dict:
    t = {}
    t["transformer.blocks.2.attn.qkv.weight"] = torch.randn(256, 256)
    t["transformer.blocks.2.attn.proj.weight"] = torch.randn(256, 256)
    t["transformer.blocks.2.mlp.fc1.weight"] = torch.randn(256, 256)

    t["net.blocks.0.attn.weight"] = torch.randn(256, 256)
    t["net.blocks.1.adaln_modulation.weight"] = torch.randn(256, 256)
    t["final_layer.linear.weight"] = torch.randn(256, 256)

    t["decoder.block.0.attn.weight"] = torch.randn(256, 256)
    t["lm_head.proj.weight"] = torch.randn(256, 256)

    t["transformer.blocks.2.attn.qkv.bias"] = torch.randn(256)
    t["transformer.norm.weight"] = torch.randn(256)

    t["conv1d_layer.weight"] = torch.randn(16, 8, 3)
    t["conv2d_layer.weight"] = torch.randn(16, 8, 3, 3)
    return t


_INT4_KWARGS = dict(
    comfy_quant=True,
    calib_samples=4,
    seed=0,
    no_learned_rounding=True,
    save_quant_metadata=False,
    low_memory=False,
    device="cpu",
    block_size=64,
    convrot_group_size=256,
)


class TestFilterFlags(unittest.TestCase):
    def setUp(self):
        self.input_file = "_test_filter_input.safetensors"
        self.output_file = "_test_filter_output.safetensors"
        save_file(_build_model(), self.input_file)

    def tearDown(self):
        for f in [self.input_file, self.output_file]:
            if os.path.exists(f):
                os.remove(f)

    def _run_convrot_int4(self, filter_flags, **extra):
        kw = {**_INT4_KWARGS, **extra}
        convert_to_int4_convrot(self.input_file, self.output_file, filter_flags=filter_flags, **kw)
        return load_file(self.output_file)

    def test_non2d_never_quantized(self):
        out = self._run_convrot_int4({})
        for base in ("conv1d_layer", "conv2d_layer"):
            self.assertIn(f"{base}.weight", out)
            self.assertFalse(_is_quantized(out, base))

    def test_normal_layers_quantized(self):
        out = self._run_convrot_int4({})
        for base in ("transformer.blocks.2.attn.qkv", "transformer.blocks.2.mlp.fc1"):
            self.assertTrue(_is_quantized(out, base))
            self.assertTrue(_has_comfy_quant(out, base))

    def test_anima_flag_skips_highprec_layers(self):
        out = self._run_convrot_int4({"anima": True})
        for base in ("net.blocks.0.attn", "net.blocks.1.adaln_modulation", "final_layer.linear"):
            self.assertFalse(_is_quantized(out, base))

    def test_t5xxl_remove_deletes_decoder_tensors(self):
        out = self._run_convrot_int4({"t5xxl": True})
        for key in ("decoder.block.0.attn.weight", "lm_head.proj.weight"):
            self.assertNotIn(key, out)


if __name__ == "__main__":
    unittest.main()
