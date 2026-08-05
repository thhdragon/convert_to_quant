"""
Unit tests for multi-GPU layer parallelism utilities and device parsing.
"""

import unittest
from unittest.mock import MagicMock, patch
import torch

from convert_to_quant.cli.main import get_parser
from convert_to_quant.utils.parallel_utils import parse_devices, run_parallel_layer_processing


class TestMultiGPUUtils(unittest.TestCase):
    """Test suite for multi-GPU parsing and parallel dispatch."""

    def test_parse_devices_explicit_string(self):
        # Comma-separated strings
        devs = parse_devices(devices="cuda:0,cuda:1")
        self.assertEqual(devs, ["cuda:0", "cuda:1"])

        # Space/index shorthand
        devs = parse_devices(devices="0, 1")
        self.assertEqual(devs, ["cuda:0", "cuda:1"])

    def test_parse_devices_explicit_list(self):
        devs = parse_devices(devices=["cuda:0", "cuda:1"])
        self.assertEqual(devs, ["cuda:0", "cuda:1"])

    def test_parse_devices_single_device(self):
        devs = parse_devices(device="cuda:0")
        self.assertEqual(devs, ["cuda:0"])

        devs = parse_devices(device="cpu")
        self.assertEqual(devs, ["cpu"])

    def test_parse_devices_num_gpus(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=4):
            devs = parse_devices(num_gpus=2)
            self.assertEqual(devs, ["cuda:0", "cuda:1"])

    def test_cli_parser_multi_gpu_args(self):
        parser = get_parser()
        args = parser.parse_args(["input.safetensors", "--devices", "cuda:0,cuda:1", "--num-gpus", "2"])
        self.assertEqual(args.devices, "cuda:0,cuda:1")
        self.assertEqual(args.num_gpus, 2)

    def test_run_parallel_layer_processing_single_device(self):
        items = [(0, "layer1"), (1, "layer2"), (2, "layer3")]

        def dummy_process(item, dev):
            idx, name = item
            return {"name": name, "dev": dev, "processed": True}

        results = run_parallel_layer_processing(items, dummy_process, ["cpu"])
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r["dev"], "cpu")
            self.assertTrue(r["processed"])

    def test_run_parallel_layer_processing_multi_device(self):
        items = [(0, "layer1"), (1, "layer2"), (2, "layer3"), (3, "layer4")]

        def dummy_process(item, dev):
            idx, name = item
            return {"name": name, "dev": dev, "processed": True}

        results = run_parallel_layer_processing(items, dummy_process, ["cpu:0", "cpu:1"])
        self.assertEqual(len(results), 4)
        processed_names = {r["name"] for r in results}
        self.assertEqual(processed_names, {"layer1", "layer2", "layer3", "layer4"})


if __name__ == "__main__":
    unittest.main()
