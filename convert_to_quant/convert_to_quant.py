"""
convert_to_quant - Dedicated INT4 ConvRot W4A4 Quantization toolkit.

This module provides tools for converting model weights to INT4 ConvRot W4A4 format
with optional learned rounding optimization and group-wise Hadamard rotation.
"""

from .cli import main
from .config import generate_config_template, get_layer_settings, load_layer_config, pattern_specificity
from .constants import (
    AVOID_KEY_NAMES,
    COMPUTE_DTYPE,
    DISTILL_LAYER_KEYNAMES_LARGE,
    DISTILL_LAYER_KEYNAMES_SMALL,
    FLUX2_LAYER_KEYNAMES,
    HUNYUAN_AVOID_KEY_NAMES,
    INT4_MAX,
    INT4_MIN,
    MODEL_FILTERS,
    NERF_LAYER_KEYNAMES_LARGE,
    NERF_LAYER_KEYNAMES_SMALL,
    NORMALIZE_SCALES_ENABLED,
    QWEN_AVOID_KEY_NAMES,
    QWEN_LAYER_KEYNAMES,
    RADIANCE_LAYER_KEYNAMES,
    SCALE_DTYPE,
    T5XXL_REMOVE_KEY_NAMES,
    VALID_QUANT_FORMATS,
    VISUAL_AVOID_KEY_NAMES,
    WAN_LAYER_KEYNAMES,
    ZIMAGE_AVOID_KEY_NAMES,
    ZIMAGE_LAYER_KEYNAMES,
    ZIMAGE_REFINER_LAYER_KEYNAMES,
    build_exclusion_patterns,
)
from .converters import BaseLearnedConverter, LearnedINT4Converter, LearnedRoundingConverter
from .formats import convert_int4_to_comfy_quant, convert_to_int4_convrot
from .utils import (
    create_comfy_quant_tensor,
    dict_to_tensor,
    edit_comfy_quant,
    fix_comfy_quant_params_structure,
    normalize_tensorwise_scales,
    parse_add_keys_string,
    should_skip_layer_for_performance,
    tensor_to_dict,
)


def quantize(input: str, output: str = None, **kwargs):
    """
    Programmatic entry point for quantizing a model to INT4 ConvRot W4A4 format.

    Args:
        input (str): Path to input safetensors file.
        output (str, optional): Path to output safetensors file.
        **kwargs: Additional arguments matching CLI flags.
    """
    import argparse
    from .cli.main import get_parser, run_conversion

    parser = get_parser()

    defaults = {}
    for action in parser._actions:
        if action.dest != "help":
            defaults[action.dest] = action.default

    defaults["input"] = input
    defaults["output"] = output

    valid_keys = set(defaults.keys())
    for k, v in kwargs.items():
        if k not in valid_keys:
            raise ValueError(f"Unknown parameter: '{k}'. Valid parameters match CLI arguments: {list(valid_keys)}")
        defaults[k] = v

    args = argparse.Namespace(**defaults)
    run_conversion(args)


__all__ = [
    "quantize",
    "AVOID_KEY_NAMES",
    "T5XXL_REMOVE_KEY_NAMES",
    "VISUAL_AVOID_KEY_NAMES",
    "QWEN_AVOID_KEY_NAMES",
    "HUNYUAN_AVOID_KEY_NAMES",
    "ZIMAGE_AVOID_KEY_NAMES",
    "FLUX2_LAYER_KEYNAMES",
    "DISTILL_LAYER_KEYNAMES_LARGE",
    "DISTILL_LAYER_KEYNAMES_SMALL",
    "NERF_LAYER_KEYNAMES_LARGE",
    "NERF_LAYER_KEYNAMES_SMALL",
    "RADIANCE_LAYER_KEYNAMES",
    "WAN_LAYER_KEYNAMES",
    "QWEN_LAYER_KEYNAMES",
    "ZIMAGE_LAYER_KEYNAMES",
    "ZIMAGE_REFINER_LAYER_KEYNAMES",
    "COMPUTE_DTYPE",
    "SCALE_DTYPE",
    "INT4_MIN",
    "INT4_MAX",
    "VALID_QUANT_FORMATS",
    "NORMALIZE_SCALES_ENABLED",
    "MODEL_FILTERS",
    "build_exclusion_patterns",
    "dict_to_tensor",
    "tensor_to_dict",
    "normalize_tensorwise_scales",
    "create_comfy_quant_tensor",
    "fix_comfy_quant_params_structure",
    "parse_add_keys_string",
    "edit_comfy_quant",
    "should_skip_layer_for_performance",
    "pattern_specificity",
    "load_layer_config",
    "get_layer_settings",
    "generate_config_template",
    "BaseLearnedConverter",
    "LearnedRoundingConverter",
    "LearnedINT4Converter",
    "convert_to_int4_convrot",
    "convert_int4_to_comfy_quant",
    "main",
]

if __name__ == "__main__":
    main()
