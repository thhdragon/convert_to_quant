import os
import shutil
import subprocess
import sys
import tempfile
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from convert_to_quant.consolidate_calibration import consolidate_calibration_data, parse_dtype
from convert_to_quant.utils.calibration_loader import CalibrationDataLoader


@pytest.fixture
def temp_calib_env():
    """Create a temporary directory with simulated multi-step diffusion activation safetensors files."""
    temp_dir = tempfile.mkdtemp(prefix="test_calib_")
    raw_dir = os.path.join(temp_dir, "raw_steps")
    os.makedirs(raw_dir, exist_ok=True)

    # Simulate 5 diffusion sampling steps
    num_steps = 5
    in_dim_proj = 128
    in_dim_mlp = 256
    tokens_per_step = 200

    created_files = []
    for step in range(num_steps):
        # Generate 3D shape (batch=1, tokens, dim) or 2D (tokens, dim)
        step_tensors = {
            "diffusion_model.double_blocks.0.img_attn.proj.weight": torch.randn(
                1, tokens_per_step, in_dim_proj, dtype=torch.float32
            ) + float(step),
            "diffusion_model.double_blocks.0.img_mlp.0.weight": torch.randn(
                tokens_per_step, in_dim_mlp, dtype=torch.float32
            ) + float(step),
            "diffusion_model.single_blocks.0.linear.weight": torch.randn(
                tokens_per_step, in_dim_proj, dtype=torch.float32
            ) + float(step),
        }
        fpath = os.path.join(raw_dir, f"step_{step:02d}.safetensors")
        save_file(step_tensors, fpath)
        created_files.append(fpath)

    yield {
        "temp_dir": temp_dir,
        "raw_dir": raw_dir,
        "files": created_files,
        "num_steps": num_steps,
        "in_dim_proj": in_dim_proj,
        "in_dim_mlp": in_dim_mlp,
    }

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_parse_dtype():
    """Verify parse_dtype handles string aliases and torch dtypes."""
    assert parse_dtype("bf16") == torch.bfloat16
    assert parse_dtype("bfloat16") == torch.bfloat16
    assert parse_dtype(torch.bfloat16) == torch.bfloat16
    assert parse_dtype("fp16") == torch.float16
    assert parse_dtype("float16") == torch.float16
    assert parse_dtype("fp32") == torch.float32
    assert parse_dtype("float32") == torch.float32

    with pytest.raises(ValueError):
        parse_dtype("invalid_dtype")


def test_consolidate_calibration_balanced_bf16(temp_calib_env):
    """Test consolidating multi-step files into BF16 with balanced strategy."""
    raw_dir = temp_calib_env["raw_dir"]
    temp_dir = temp_calib_env["temp_dir"]
    out_file = os.path.join(temp_dir, "calib_stack_bf16.safetensors")

    samples_target = 250  # 5 steps * 50 tokens each = 250
    result = consolidate_calibration_data(
        input_path=raw_dir,
        output_path=out_file,
        samples_per_layer=samples_target,
        dtype="bfloat16",
        strategy="balanced",
        seed=42,
    )

    assert os.path.exists(out_file)
    assert len(result) == 3

    # Check tensors in returned dict and file
    proj_key = "double_blocks.0.img_attn.proj"
    mlp_key = "double_blocks.0.img_mlp.0"
    single_key = "single_blocks.0.linear"

    assert proj_key in result
    assert mlp_key in result
    assert single_key in result

    assert result[proj_key].shape == (samples_target, temp_calib_env["in_dim_proj"])
    assert result[proj_key].dtype == torch.bfloat16

    assert result[mlp_key].shape == (samples_target, temp_calib_env["in_dim_mlp"])
    assert result[mlp_key].dtype == torch.bfloat16

    # Verify saved safetensors file metadata and keys
    with safe_open(out_file, framework="pt") as f:
        meta = f.metadata()
        assert meta["format"] == "consolidated_ptq_calibration"
        assert meta["dtype"] == "torch.bfloat16"
        assert meta["samples_per_layer"] == str(samples_target)
        assert meta["source_files_count"] == "5"

        loaded_t = f.get_tensor(proj_key)
        assert loaded_t.shape == (samples_target, temp_calib_env["in_dim_proj"])
        assert loaded_t.dtype == torch.bfloat16


def test_consolidate_calibration_random_strategy(temp_calib_env):
    """Test consolidation with random pooling strategy."""
    raw_dir = temp_calib_env["raw_dir"]
    temp_dir = temp_calib_env["temp_dir"]
    out_file = os.path.join(temp_dir, "calib_stack_random.safetensors")

    samples_target = 100
    result = consolidate_calibration_data(
        input_path=raw_dir,
        output_path=out_file,
        samples_per_layer=samples_target,
        dtype="bfloat16",
        strategy="random",
        seed=123,
    )

    for k, tensor in result.items():
        assert tensor.shape[0] == samples_target
        assert tensor.dtype == torch.bfloat16


def test_consolidate_calibration_layer_filter(temp_calib_env):
    """Test filtering specific layers during consolidation."""
    raw_dir = temp_calib_env["raw_dir"]
    temp_dir = temp_calib_env["temp_dir"]
    out_file = os.path.join(temp_dir, "calib_stack_filtered.safetensors")

    result = consolidate_calibration_data(
        input_path=raw_dir,
        output_path=out_file,
        samples_per_layer=128,
        dtype="bfloat16",
        layer_filter="img_attn",
    )

    assert len(result) == 1
    assert "double_blocks.0.img_attn.proj" in result


def test_calibration_loader_with_consolidated_bf16(temp_calib_env):
    """Verify that CalibrationDataLoader seamlessly consumes the consolidated BF16 file."""
    raw_dir = temp_calib_env["raw_dir"]
    temp_dir = temp_calib_env["temp_dir"]
    out_file = os.path.join(temp_dir, "calib_stack_bf16.safetensors")

    samples_target = 300
    consolidate_calibration_data(
        input_path=raw_dir,
        output_path=out_file,
        samples_per_layer=samples_target,
        dtype="bfloat16",
        strategy="balanced",
    )

    # Initialize loader pointing directly to the consolidated safetensors file
    loader = CalibrationDataLoader(out_file, max_tokens=samples_target)

    assert loader.has_layer("double_blocks.0.img_attn.proj.weight")
    assert loader.has_layer("model.diffusion_model.double_blocks.0.img_mlp.0.weight")
    assert loader.metadata.get("dtype") == "torch.bfloat16"

    # Fetch tensor
    tensor = loader.get_calibration_tensor(
        "double_blocks.0.img_attn.proj.weight",
        max_tokens=samples_target,
        dtype=torch.float32,
    )
    assert tensor is not None
    assert tensor.shape == (samples_target, temp_calib_env["in_dim_proj"])
    assert tensor.dtype == torch.float32


def test_consolidate_cli(temp_calib_env):
    """Test CLI command execution for consolidation."""
    raw_dir = temp_calib_env["raw_dir"]
    temp_dir = temp_calib_env["temp_dir"]
    out_file = os.path.join(temp_dir, "cli_calib_stack.safetensors")

    cmd = [
        sys.executable,
        "-m",
        "convert_to_quant.consolidate_calibration",
        "-i",
        raw_dir,
        "-o",
        out_file,
        "-n",
        "150",
        "--dtype",
        "bf16",
        "--strategy",
        "balanced",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed with error:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert os.path.exists(out_file)
