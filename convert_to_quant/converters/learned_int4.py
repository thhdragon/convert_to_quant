"""Learned Rounding INT4 Quantization Converter.

Implements INT4 W4A4 quantization with ConvRot group-wise Hadamard rotation
and optional SVD-based learned rounding optimization.
"""

from .learned_rounding import LearnedRoundingConverter


class LearnedINT4Converter(LearnedRoundingConverter):
    """Convert to INT4 W4A4 ConvRot quantization.

    Sets target_format="int4", convrot=True, and scaling_mode="row" by default.
    """

    def __init__(
        self,
        scaling_mode: str = "row",
        block_size: int = 64,
        target_format: str = "int4",
        convrot: bool = True,
        convrot_group_size: int = 256,
        dynamic_convrot: bool = False,
        **kwargs,
    ):
        super().__init__(
            target_format=target_format,
            scaling_mode=scaling_mode,
            block_size=block_size,
            convrot=convrot,
            convrot_group_size=convrot_group_size,
            dynamic_convrot=dynamic_convrot,
            **kwargs,
        )
