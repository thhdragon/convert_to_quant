"""
Unit tests for per-layer sidecar checkpointing, stop & resume, and output sharding.
"""

import json
import os
import shutil
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from convert_to_quant.utils.checkpoint import QuantCheckpointManager, parse_shard_size


class TestQuantCheckpointManager(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="ctq_test_ckpt_")
        self.input_file = os.path.join(self.test_dir, "input_model.safetensors")
        self.output_file = os.path.join(self.test_dir, "output_model.safetensors")

        # Create dummy input safetensors model with 4 weight layers
        self.tensors = {
            "layer0.weight": torch.randn(64, 64, dtype=torch.float32),
            "layer0.bias": torch.randn(64, dtype=torch.float32),
            "layer1.weight": torch.randn(64, 64, dtype=torch.float32),
            "layer1.bias": torch.randn(64, dtype=torch.float32),
            "layer2.weight": torch.randn(64, 64, dtype=torch.float32),
            "layer2.bias": torch.randn(64, dtype=torch.float32),
            "layer3.weight": torch.randn(64, 64, dtype=torch.float32),
            "layer3.bias": torch.randn(64, dtype=torch.float32),
        }
        save_file(self.tensors, self.input_file)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_parse_shard_size(self):
        self.assertIsNone(parse_shard_size(None))
        self.assertIsNone(parse_shard_size(""))
        self.assertEqual(parse_shard_size("5GB"), 5 * 1024**3)
        self.assertEqual(parse_shard_size("500MB"), 500 * 1024**2)
        self.assertEqual(parse_shard_size(1024), 1024)

    def test_checkpoint_save_and_resume(self):
        # 1. Initialize checkpoint manager
        mgr = QuantCheckpointManager(
            output_file=self.output_file,
            input_file=self.input_file,
            primary_format="w4a8_int8",
        )

        # Layer 0 finishes
        layer0_result = {
            "key": "layer0.weight",
            "base_name": "layer0",
            "tensors": {
                "layer0.weight": torch.ones(64, 32, dtype=torch.uint8),
                "layer0.weight_scale": torch.ones(64, 1, dtype=torch.float32),
            },
            "meta_entry": {"format": "w4a8_int8", "group_size": 16},
            "skipped": False,
        }
        mgr.save_layer_checkpoint(layer0_result)

        # Layer 1 finishes
        layer1_result = {
            "key": "layer1.weight",
            "base_name": "layer1",
            "tensors": {
                "layer1.weight": torch.ones(64, 32, dtype=torch.uint8) * 2,
                "layer1.weight_scale": torch.ones(64, 1, dtype=torch.float32),
            },
            "meta_entry": {"format": "w4a8_int8", "group_size": 16},
            "skipped": False,
        }
        mgr.save_layer_checkpoint(layer1_result)

        # Sidecar file should exist
        sidecar_file = f"{self.output_file}.progress.json"
        self.assertTrue(os.path.exists(sidecar_file))

        with open(sidecar_file, "r") as f:
            sidecar_data = json.load(f)
        self.assertIn("layer0.weight", sidecar_data["completed_layers"])
        self.assertIn("layer1.weight", sidecar_data["completed_layers"])

        # 2. Simulate interruption and new manager with resume=True
        resume_mgr = QuantCheckpointManager(
            output_file=self.output_file,
            input_file=self.input_file,
            primary_format="w4a8_int8",
            resume=True,
        )

        self.assertTrue(resume_mgr.is_layer_completed("layer0.weight"))
        self.assertTrue(resume_mgr.is_layer_completed("layer1.weight"))
        self.assertFalse(resume_mgr.is_layer_completed("layer2.weight"))

        loaded_l0 = resume_mgr.load_completed_layer("layer0.weight")
        self.assertIsNotNone(loaded_l0)
        self.assertIn("layer0.weight", loaded_l0["tensors"])

        # Complete remaining layers
        layer2_result = {
            "key": "layer2.weight",
            "base_name": "layer2",
            "tensors": {
                "layer2.weight": torch.ones(64, 32, dtype=torch.uint8) * 3,
                "layer2.weight_scale": torch.ones(64, 1, dtype=torch.float32),
            },
            "meta_entry": {"format": "w4a8_int8", "group_size": 16},
            "skipped": False,
        }
        resume_mgr.save_layer_checkpoint(layer2_result)

        layer3_result = {
            "key": "layer3.weight",
            "base_name": "layer3",
            "tensors": {
                "layer3.weight": torch.ones(64, 32, dtype=torch.uint8) * 4,
                "layer3.weight_scale": torch.ones(64, 1, dtype=torch.float32),
            },
            "meta_entry": {"format": "w4a8_int8", "group_size": 16},
            "skipped": False,
        }
        resume_mgr.save_layer_checkpoint(layer3_result)

        # Assemble final output
        passthrough = {
            "layer0.bias": self.tensors["layer0.bias"],
            "layer1.bias": self.tensors["layer1.bias"],
            "layer2.bias": self.tensors["layer2.bias"],
            "layer3.bias": self.tensors["layer3.bias"],
        }
        success = resume_mgr.assemble_final_output(passthrough, original_metadata={"source": "test"})
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.output_file))

        # Check final output content
        final_tensors = load_file(self.output_file)
        self.assertIn("layer0.weight", final_tensors)
        self.assertIn("layer1.weight", final_tensors)
        self.assertIn("layer2.weight", final_tensors)
        self.assertIn("layer3.weight", final_tensors)
        self.assertIn("layer0.bias", final_tensors)

    def test_sharded_output_assembly(self):
        sharded_output_file = os.path.join(self.test_dir, "sharded_model.safetensors")
        # Use small max_shard_size to force multiple shards
        mgr = QuantCheckpointManager(
            output_file=sharded_output_file,
            input_file=self.input_file,
            primary_format="fp8",
            max_shard_size="1KB",  # force small shard limit
        )

        for i in range(4):
            key = f"layer{i}.weight"
            mgr.save_layer_checkpoint({
                "key": key,
                "base_name": f"layer{i}",
                "tensors": {key: torch.randn(64, 64, dtype=torch.float32)},
                "meta_entry": {"format": "fp8"},
            })

        success = mgr.assemble_final_output({})
        self.assertTrue(success)

        # Verify index file exists
        index_file = os.path.join(self.test_dir, "sharded_model.safetensors.index.json")
        self.assertTrue(os.path.exists(index_file))

        with open(index_file, "r") as f:
            idx_data = json.load(f)
        self.assertIn("weight_map", idx_data)
        self.assertGreater(len(idx_data["weight_map"]), 0)


if __name__ == "__main__":
    unittest.main()
