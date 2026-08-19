import logging
from typing import Dict, Tuple

import torch

from . import (
    float as comfy_float,  # Aliased to match ComfyUI's comfy.float usage
)

_LAYOUT_REGISTRY = {}
_GENERIC_UTILS = {}


def register_layout_op(torch_op, layout_type):
    """
    Decorator to register a layout-specific operation handler.
    """
    def decorator(handler_func):
        if torch_op not in _LAYOUT_REGISTRY:
            _LAYOUT_REGISTRY[torch_op] = {}
        _LAYOUT_REGISTRY[torch_op][layout_type] = handler_func
        return handler_func

    return decorator


def register_generic_util(torch_op):
    """
    Decorator to register a generic utility that works for all layouts.
    """
    def decorator(handler_func):
        _GENERIC_UTILS[torch_op] = handler_func
        return handler_func

    return decorator


def _get_layout_from_args(args):
    for arg in args:
        if isinstance(arg, QuantizedTensor):
            return arg._layout_type
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, QuantizedTensor):
                    return item._layout_type
    return None


def _move_layout_params_to_device(params, device):
    new_params = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            new_params[k] = v.to(device=device)
        else:
            new_params[k] = v
    return new_params


def _copy_layout_params(params):
    new_params = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            new_params[k] = v.clone()
        else:
            new_params[k] = v
    return new_params


def _copy_layout_params_inplace(src, dst, non_blocking=False):
    for k, v in src.items():
        if isinstance(v, torch.Tensor):
            dst[k].copy_(v, non_blocking=non_blocking)
        else:
            dst[k] = v


class QuantizedLayout:
    """Base class for quantization layouts."""

    @classmethod
    def quantize(cls, tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError(f"{cls.__name__} must implement quantize()")

    @staticmethod
    def dequantize(qdata, **layout_params) -> torch.Tensor:
        raise NotImplementedError("TensorLayout must implement dequantize()")

    @classmethod
    def get_plain_tensors(cls, qtensor) -> torch.Tensor:
        raise NotImplementedError(f"{cls.__name__} must implement get_plain_tensors()")


class QuantizedTensor(torch.Tensor):
    """
    Universal quantized tensor wrapper for INT4 ConvRot W4A4 layout.
    """

    @staticmethod
    def __new__(cls, qdata, layout_type, layout_params):
        return torch.Tensor._make_wrapper_subclass(cls, qdata.shape, device=qdata.device, dtype=qdata.dtype, requires_grad=False)

    def __init__(self, qdata, layout_type, layout_params):
        self._qdata = qdata
        self._layout_type = layout_type
        self._layout_params = layout_params

    def __repr__(self):
        param_str = ", ".join(f"{k}={v}" for k, v in list(self._layout_params.items())[:2])
        return f"QuantizedTensor(shape={self.shape}, layout={self._layout_type}, {param_str})"

    @property
    def layout_type(self):
        return self._layout_type

    def __tensor_flatten__(self):
        inner_tensors = ["_qdata"]
        ctx = {"layout_type": self._layout_type}

        tensor_params = {}
        non_tensor_params = {}
        for k, v in self._layout_params.items():
            if isinstance(v, torch.Tensor):
                tensor_params[k] = v
            else:
                non_tensor_params[k] = v

        ctx["tensor_param_keys"] = list(tensor_params.keys())
        ctx["non_tensor_params"] = non_tensor_params

        for k, v in tensor_params.items():
            attr_name = f"_layout_param_{k}"
            object.__setattr__(self, attr_name, v)
            inner_tensors.append(attr_name)

        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        layout_type = ctx["layout_type"]
        layout_params = dict(ctx["non_tensor_params"])

        for key in ctx["tensor_param_keys"]:
            attr_name = f"_layout_param_{key}"
            layout_params[key] = inner_tensors[attr_name]

        return QuantizedTensor(inner_tensors["_qdata"], layout_type, layout_params)

    @classmethod
    def from_float(cls, tensor, layout_type="TensorCoreConvRotW4A4Layout", **quantize_kwargs) -> "QuantizedTensor":
        qdata, layout_params = LAYOUTS[layout_type].quantize(tensor, **quantize_kwargs)
        return cls(qdata, layout_type, layout_params)

    def dequantize(self) -> torch.Tensor:
        return LAYOUTS[self._layout_type].dequantize(self._qdata, **self._layout_params)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if func in _GENERIC_UTILS:
            return _GENERIC_UTILS[func](func, args, kwargs)

        layout_type = _get_layout_from_args(args)
        if layout_type and func in _LAYOUT_REGISTRY:
            handler = _LAYOUT_REGISTRY[func].get(layout_type)
            if handler:
                return handler(func, args, kwargs)

        return cls._dequant_and_fallback(func, args, kwargs)

    @classmethod
    def _dequant_and_fallback(cls, func, args, kwargs):
        def dequant_arg(arg):
            if isinstance(arg, QuantizedTensor):
                return arg.dequantize()
            elif isinstance(arg, (list, tuple)):
                return type(arg)(dequant_arg(a) for a in arg)
            return arg

        new_args = dequant_arg(args)
        new_kwargs = dequant_arg(kwargs)
        return func(*new_args, **new_kwargs)

    def data_ptr(self):
        return self._qdata.data_ptr()

    def is_pinned(self):
        return self._qdata.is_pinned()

    def is_contiguous(self, *arg, **kwargs):
        return self._qdata.is_contiguous(*arg, **kwargs)

    def storage(self):
        return self._qdata.storage()


# ==============================================================================
# Generic Utilities (Layout-Agnostic Operations)
# ==============================================================================

def _create_transformed_qtensor(qt, transform_fn):
    new_data = transform_fn(qt._qdata)
    new_params = _copy_layout_params(qt._layout_params)
    return QuantizedTensor(new_data, qt._layout_type, new_params)


def _handle_device_transfer(qt, target_device, target_dtype=None, target_layout=None, op_name="to"):
    current_device = qt._qdata.device
    if target_device is not None:
        if isinstance(target_device, str):
            target_device = torch.device(target_device)
        if isinstance(current_device, str):
            current_device = torch.device(current_device)

        if target_device != current_device:
            new_q_data = qt._qdata.to(device=target_device)
            new_params = _move_layout_params_to_device(qt._layout_params, target_device)
            if target_dtype is not None:
                new_params["orig_dtype"] = target_dtype
            return QuantizedTensor(new_q_data, qt._layout_type, new_params)

    return qt


@register_generic_util(torch.ops.aten.detach.default)
def generic_detach(func, args, kwargs):
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        return _create_transformed_qtensor(qt, lambda x: x.detach())
    return func(*args, **kwargs)


@register_generic_util(torch.ops.aten.clone.default)
def generic_clone(func, args, kwargs):
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        return _create_transformed_qtensor(qt, lambda x: x.clone())
    return func(*args, **kwargs)


@register_generic_util(torch.ops.aten._to_copy.default)
def generic_to_copy(func, args, kwargs):
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        return _handle_device_transfer(qt, target_device=kwargs.get("device", None), target_dtype=kwargs.get("dtype", None), op_name="_to_copy")
    return func(*args, **kwargs)


@register_generic_util(torch.ops.aten.to.dtype_layout)
def generic_to_dtype_layout(func, args, kwargs):
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        return _handle_device_transfer(qt, target_device=kwargs.get("device", None), target_dtype=kwargs.get("dtype", None), target_layout=kwargs.get("layout", None), op_name="to")
    return func(*args, **kwargs)


@register_generic_util(torch.ops.aten.to.dtype)
def generic_to_dtype(func, args, kwargs):
    src = args[0]
    if isinstance(src, QuantizedTensor):
        target_dtype = args[1] if len(args) > 1 else kwargs.get("dtype")
        src._layout_params["orig_dtype"] = target_dtype
        return src
    return func(*args, **kwargs)


@register_generic_util(torch.ops.aten.copy_.default)
def generic_copy_(func, args, kwargs):
    qt_dest = args[0]
    src = args[1]
    non_blocking = args[2] if len(args) > 2 else False
    if isinstance(qt_dest, QuantizedTensor):
        if isinstance(src, QuantizedTensor):
            qt_dest._qdata.copy_(src._qdata, non_blocking=non_blocking)
            qt_dest._layout_type = src._layout_type
            orig_dtype = qt_dest._layout_params.get("orig_dtype")
            _copy_layout_params_inplace(src._layout_params, qt_dest._layout_params, non_blocking=non_blocking)
            if orig_dtype is not None:
                qt_dest._layout_params["orig_dtype"] = orig_dtype
        else:
            qt_dest._qdata.copy_(src)
        return qt_dest
    return func(*args, **kwargs)


@register_generic_util(torch.ops.aten._has_compatible_shallow_copy_type.default)
def generic_has_compatible_shallow_copy_type(func, args, kwargs):
    return True


@register_generic_util(torch.ops.aten.empty_like.default)
def generic_empty_like(func, args, kwargs):
    qt = args[0]
    if isinstance(qt, QuantizedTensor):
        hp_dtype = kwargs.pop("dtype", qt._layout_params.get("orig_dtype", torch.float32))
        new_qdata = torch.empty_like(qt._qdata, **kwargs)
        target_device = kwargs.get("device", new_qdata.device)
        new_params = _move_layout_params_to_device(qt._layout_params, target_device)
        new_params["orig_dtype"] = hp_dtype
        return QuantizedTensor(new_qdata, qt._layout_type, new_params)
    return func(*args, **kwargs)


# ==============================================================================
# INT4 ConvRot W4A4 Layout Definition
# ==============================================================================
class TensorCoreConvRotW4A4Layout(QuantizedLayout):
    """
    ConvRot W4A4 Quantization Layout (group-wise Hadamard rotation + INT4).

    Storage format:
    - qdata: Packed int8 tensor (2 x 4-bit signed values [-7, 7] per byte)
    - scale: Weight scale tensor (float32, 1 per row)
    - convrot_groupsize: Group size for Hadamard rotation (default 256)
    - quant_group_size: INT4 quantization group size (default 64)
    """

    @classmethod
    def quantize(cls, tensor, scale=None, convrot_groupsize=256, quant_group_size=64, **kwargs):
        from .int4_kernels import quantize_convrot_w4a4_weight

        qdata, scales = quantize_convrot_w4a4_weight(tensor, convrot_groupsize=convrot_groupsize, quant_group_size=quant_group_size)

        layout_params = {
            "scale": scales.to(torch.float32),
            "convrot_groupsize": convrot_groupsize,
            "quant_group_size": quant_group_size,
            "orig_dtype": tensor.dtype,
        }
        return qdata, layout_params

    @staticmethod
    def dequantize(qdata, scale, orig_dtype=None, convrot_groupsize=256, quant_group_size=64, **kwargs):
        from .int4_kernels import dequantize_convrot_w4a4_weight

        out_dtype = orig_dtype if orig_dtype is not None else torch.float32
        return dequantize_convrot_w4a4_weight(
            qdata,
            scale,
            convrot_groupsize=convrot_groupsize,
            quant_group_size=quant_group_size,
            output_dtype=out_dtype,
        )

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return qtensor._qdata, qtensor._layout_params["scale"]


# Layout Registry
LAYOUTS = {
    "TensorCoreConvRotW4A4Layout": TensorCoreConvRotW4A4Layout,
}

QUANT_ALGOS = {
    "convrot_w4a4": {
        "layout_params": ["scale"],
        "comfy_tensor_layout": "TensorCoreConvRotW4A4Layout",
    },
}


# ==============================================================================
# INT4 ConvRot W4A4 Layout Operation Handlers
# ==============================================================================
@register_layout_op(torch.ops.aten.linear.default, "TensorCoreConvRotW4A4Layout")
def convrot_w4a4_linear_handler(func, args, kwargs):
    from .int4_kernels import convrot_w4a4_linear

    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias", None)

    qweight, wscales = TensorCoreConvRotW4A4Layout.get_plain_tensors(weight)
    convrot_groupsize = weight._layout_params.get("convrot_groupsize", 256)
    quant_group_size = weight._layout_params.get("quant_group_size", 64)

    return convrot_w4a4_linear(
        input_tensor,
        qweight,
        wscales,
        bias=bias,
        convrot_groupsize=convrot_groupsize,
        quant_group_size=quant_group_size,
        linear_dtype="int4",
    )


@register_layout_op(torch.ops.aten.mm.default, "TensorCoreConvRotW4A4Layout")
def convrot_w4a4_mm_handler(func, args, kwargs):
    from .int4_kernels import convrot_w4a4_linear

    input_tensor = args[0]
    weight = args[1]

    qweight, wscales = TensorCoreConvRotW4A4Layout.get_plain_tensors(weight)
    convrot_groupsize = weight._layout_params.get("convrot_groupsize", 256)
    quant_group_size = weight._layout_params.get("quant_group_size", 64)

    return convrot_w4a4_linear(
        input_tensor,
        qweight,
        wscales,
        bias=None,
        convrot_groupsize=convrot_groupsize,
        quant_group_size=quant_group_size,
        linear_dtype="int4",
    )


@register_layout_op(torch.ops.aten.addmm.default, "TensorCoreConvRotW4A4Layout")
def convrot_w4a4_addmm_handler(func, args, kwargs):
    from .int4_kernels import convrot_w4a4_linear

    bias = args[0]
    input_tensor = args[1]
    weight = args[2]

    qweight, wscales = TensorCoreConvRotW4A4Layout.get_plain_tensors(weight)
    convrot_groupsize = weight._layout_params.get("convrot_groupsize", 256)
    quant_group_size = weight._layout_params.get("quant_group_size", 64)

    return convrot_w4a4_linear(
        input_tensor,
        qweight,
        wscales,
        bias=bias,
        convrot_groupsize=convrot_groupsize,
        quant_group_size=quant_group_size,
        linear_dtype="int4",
    )


@register_layout_op(torch.ops.aten.view.default, "TensorCoreConvRotW4A4Layout")
@register_layout_op(torch.ops.aten.t.default, "TensorCoreConvRotW4A4Layout")
def convrot_w4a4_func(func, args, kwargs):
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        plain_input, scale = TensorCoreConvRotW4A4Layout.get_plain_tensors(input_tensor)
        ar = list(args)
        ar[0] = plain_input
        return QuantizedTensor(func(*ar, **kwargs), "TensorCoreConvRotW4A4Layout", input_tensor._layout_params)
    return func(*args, **kwargs)
