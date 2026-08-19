import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from convert_to_quant.converters.learned_int4 import LearnedINT4Converter
from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
from convert_to_quant.formats.int4_convrot_conversion import convert_to_int4_convrot


class TestLoraExtraction(unittest.TestCase):
    def test_heuristics(self):
        converter = LearnedRoundingConverter(extract_lora=True, lora_depth=1, lora_rank=16)

        self.assertTrue(converter._should_extract_lora("double_blocks.0.img_attn.qkv.weight", torch.Size([4096, 4096]), depth=0))
        self.assertFalse(converter._should_extract_lora("double_blocks.1.img_attn.qkv.weight", torch.Size([4096, 4096]), depth=1))
        self.assertFalse(converter._should_extract_lora("double_blocks.1.img_mlp.0.weight", torch.Size([16384, 4096]), depth=1))

        converter.lora_target_regex = __import__("re").compile("mlp")
        self.assertTrue(converter._should_extract_lora("double_blocks.1.img_mlp.0.weight", torch.Size([16384, 4096]), depth=1))

    def test_list_lora_target(self):
        converter = LearnedRoundingConverter(
            extract_lora=True,
            lora_target=["img_attn", "img_mlp"],
            lora_depth=1,
            lora_rank=16,
        )
        self.assertTrue(converter._should_extract_lora("double_blocks.5.img_attn.qkv.weight", torch.Size([4096, 4096]), depth=5))
        self.assertTrue(converter._should_extract_lora("double_blocks.5.img_mlp.0.weight", torch.Size([16384, 4096]), depth=5))

    def test_unlimited_depth(self):
        converter = LearnedRoundingConverter(extract_lora=True, lora_depth=-1, lora_rank=16)
        self.assertTrue(converter._should_extract_lora("double_blocks.5.img_attn.qkv.weight", torch.Size([128, 128]), depth=5))

    def test_extraction_learned_rounding(self):
        converter = LearnedINT4Converter(
            extract_lora=True,
            lora_rank=4,
            no_learned_rounding=True,
            convrot=False,
        )
        W = torch.randn(256, 256)
        q, s, dq, extra = converter.convert(W, key="double_blocks.0.img_attn.qkv.weight")

        self.assertIn("lora_up", extra)
        self.assertIn("lora_down", extra)
        self.assertEqual(extra["lora_up"].shape, (256, 4))
        self.assertEqual(extra["lora_down"].shape, (4, 256))

        error_approx = extra["lora_up"].float() @ extra["lora_down"].float()
        self.assertGreater(torch.norm(error_approx), 0)

    def test_conversion_lora_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "model.safetensors")
            output_int4 = os.path.join(tmpdir, "model_int4.safetensors")
            lora_out = os.path.join(tmpdir, "custom_lora.safetensors")

            tensors = {"double_blocks.0.weight": torch.randn(256, 256)}
            save_file(tensors, input_path)

            convert_to_int4_convrot(
                input_path,
                output_int4,
                comfy_quant=True,
                filter_flags={},
                calib_samples=3072,
                seed=42,
                no_learned_rounding=True,
                extract_lora=True,
                lora_rank=4,
                lora_depth=1,
                lora_output=lora_out,
            )
            self.assertTrue(os.path.exists(lora_out))


if __name__ == "__main__":
    unittest.main()
