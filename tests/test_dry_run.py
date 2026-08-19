from __future__ import annotations

import subprocess
import sys

import torch
from safetensors.torch import save_file


def test_dry_run_analyzes_without_writing(tmp_path):
    input_path = tmp_path / "input.safetensors"
    output_path = tmp_path / "must_not_exist.safetensors"
    save_file(
        {
            "transformer.block.weight": torch.ones((256, 256), dtype=torch.float32),
            "transformer.block.bias": torch.ones((256,), dtype=torch.float32),
        },
        str(input_path),
    )

    analyze = subprocess.run(
        [
            sys.executable,
            "-m",
            "convert_to_quant.cli.main",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert analyze.returncode == 0, analyze.stderr
    assert "Dry-run analysis (no conversion will be performed)" in analyze.stdout
    assert "transformer.block.weight [256, 256] -> primary:convrot_w4a4" in analyze.stdout
    assert "passthrough tensors: 1" in analyze.stdout
    assert "No output file was written." in analyze.stdout
    assert not output_path.exists()
