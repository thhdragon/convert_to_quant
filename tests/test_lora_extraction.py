import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from convert_to_quant.converters.learned_mxfp8 import LearnedMXFP8Converter
from convert_to_quant.converters.learned_nvfp4 import LearnedNVFP4Converter
from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
from convert_to_quant.formats.fp8_conversion import convert_to_fp8_scaled
from convert_to_quant.formats.mxfp8_conversion import convert_to_mxfp8
from convert_to_quant.formats.nvfp4_conversion import convert_to_nvfp4


class TestLoraExtraction(unittest.TestCase):
    def test_heuristics(self):
        # Setup converter with LoRA enabled
        converter = LearnedRoundingConverter(extract_lora=True, lora_depth=1, lora_rank=16)

        # 1. Block 0 should be targeted
        self.assertTrue(converter._should_extract_lora("double_blocks.0.img_attn.qkv.weight", torch.Size([4096, 4096]), depth=0))

        # 2. Block 1 Attention (Square) should NOT be targeted (Depth 1 means only Block 0)
        self.assertFalse(converter._should_extract_lora("double_blocks.1.img_attn.qkv.weight", torch.Size([4096, 4096]), depth=1))

        # 3. Block 1 MLP (Elongated) should NOT be targeted
        self.assertFalse(converter._should_extract_lora("double_blocks.1.img_mlp.0.weight", torch.Size([16384, 4096]), depth=1))

        # 4. Explicit Regex Target
        converter.lora_target_regex = __import__("re").compile("mlp")
        self.assertTrue(converter._should_extract_lora("double_blocks.1.img_mlp.0.weight", torch.Size([16384, 4096]), depth=1))

    def test_list_lora_target(self):
        # Test lora_target as list[str]
        converter = LearnedRoundingConverter(
            extract_lora=True,
            lora_target=["img_attn", "img_mlp"],
            lora_depth=1,
            lora_rank=16,
        )
        self.assertTrue(converter._should_extract_lora("double_blocks.5.img_attn.qkv.weight", torch.Size([4096, 4096]), depth=5))
        self.assertTrue(converter._should_extract_lora("double_blocks.5.img_mlp.0.weight", torch.Size([16384, 4096]), depth=5))

    def test_unlimited_depth(self):
        # Test lora_depth=-1 allows extraction for depth > 0
        converter = LearnedRoundingConverter(extract_lora=True, lora_depth=-1, lora_rank=16)
        self.assertTrue(converter._should_extract_lora("double_blocks.5.img_attn.qkv.weight", torch.Size([128, 128]), depth=5))

    def test_padded_mxfp8_lora_extraction(self):
        # Test LoRA extraction when pad_to_32x is used on non-divisible shape
        converter = LearnedMXFP8Converter(
            extract_lora=True,
            lora_rank=4,
            pad_to_32x=True,
            no_learned_rounding=True,
        )
        W = torch.randn(50, 50)
        q, bs, dq, extra = converter.convert(W, key="double_blocks.0.weight")
        self.assertIn("lora_up", extra)
        self.assertEqual(extra["lora_up"].shape, (50, 4))
        self.assertEqual(extra["lora_down"].shape, (4, 50))

    def test_extraction_learned_rounding(self):
        converter = LearnedRoundingConverter(
            extract_lora=True,
            lora_rank=4,
            no_learned_rounding=True,  # Use simple quant for fast test
        )

        # Create dummy weight
        W = torch.randn(128, 128)

        # Run conversion
        q, s, dq, extra = converter.convert(W, key="double_blocks.0.img_attn.qkv.weight")

        self.assertIn("lora_up", extra)
        self.assertIn("lora_down", extra)
        self.assertEqual(extra["lora_up"].shape, (128, 4))
        self.assertEqual(extra["lora_down"].shape, (4, 128))

        # Verify reconstruction
        error_approx = extra["lora_up"].float() @ extra["lora_down"].float()

        # The approximation should capture some variance
        self.assertGreater(torch.norm(error_approx), 0)

    def test_extraction_nvfp4(self):
        converter = LearnedNVFP4Converter(extract_lora=True, lora_rank=4, no_learned_rounding=True)

        W = torch.randn(64, 64)
        q, s, ps, dq, extra = converter.convert(W, key="double_blocks.0.weight")

        self.assertIn("lora_up", extra)
        self.assertEqual(extra["lora_up"].shape, (64, 4))

    def test_contiguous_output(self):
        converter = LearnedRoundingConverter(extract_lora=True, lora_rank=4, no_learned_rounding=True)
        W = torch.randn(64, 64)
        _, _, _, extra = converter.convert(W, key="double_blocks.0.weight")

        self.assertTrue(extra["lora_up"].is_contiguous(), "lora_up should be contiguous")
        self.assertTrue(extra["lora_down"].is_contiguous(), "lora_down should be contiguous")

    def test_conversion_lora_output_all_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "model.safetensors")
            output_nvfp4 = os.path.join(tmpdir, "model_nvfp4.safetensors")
            output_mxfp8 = os.path.join(tmpdir, "model_mxfp8.safetensors")
            output_fp8 = os.path.join(tmpdir, "model_fp8.safetensors")
            lora_out = os.path.join(tmpdir, "custom_lora.safetensors")

            tensors = {"double_blocks.0.weight": torch.randn(64, 64)}
            save_file(tensors, input_path)

            convert_to_nvfp4(
                input_path,
                output_nvfp4,
                simple=False,
                num_iter=1,
                extract_lora=True,
                lora_rank=4,
                lora_depth=1,
                lora_output=lora_out,
            )
            self.assertTrue(os.path.exists(lora_out))
            os.remove(lora_out)

            convert_to_mxfp8(
                input_path,
                output_mxfp8,
                simple=False,
                num_iter=1,
                extract_lora=True,
                lora_rank=4,
                lora_depth=1,
                lora_output=lora_out,
            )
            self.assertTrue(os.path.exists(lora_out))
            os.remove(lora_out)

            convert_to_fp8_scaled(
                input_path,
                output_fp8,
                comfy_quant=True,
                filter_flags={},
                calib_samples=3072,
                seed=42,
                simple=True,
                extract_lora=True,
                lora_rank=4,
                lora_depth=1,
                lora_output=lora_out,
            )
            self.assertTrue(os.path.exists(lora_out))

    def test_device_mismatch_lora_extraction(self):
        from convert_to_quant.converters.base_converter import BaseLearnedConverter

        class DummyConverter(BaseLearnedConverter):
            def convert(self, W_orig, key=None, depth=-1, **kwargs):
                pass

        converter = DummyConverter(extract_lora=True, lora_rank=4)
        W_orig = torch.randn(64, 64, device="cpu")
        device_cuda = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        W_dequant = torch.randn(64, 64, device=device_cuda)

        lora_data = converter._extract_error_lora(W_orig, W_dequant)
        self.assertIsNotNone(lora_data)
        self.assertIn("lora_up", lora_data)
        self.assertIn("lora_down", lora_data)
        self.assertEqual(lora_data["lora_up"].device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()


