from __future__ import annotations

from ..language.dtypes import DType, DTypeCategory
from .interface import ScalarParameter, ViewParameter


def torch_dtype(element: DType):
    import torch

    names = {"index": "int64", "bool": "bool", "i8": "int8", "i16": "int16",
             "i32": "int32", "i64": "int64", "u8": "uint8", "u16": "uint16",
             "u32": "uint32", "u64": "uint64", "f16": "float16", "bf16": "bfloat16",
             "f32": "float32", "f64": "float64", "f8e4m3fn": "float8_e4m3fn", "f8e5m2": "float8_e5m2"}
    return getattr(torch, names[element.name])


def register_operator(artifact, name: str):
    """Register a functional allocating invocation with PyTorch's dispatcher.

    Device execution is opaque to PyTorch. The fake implementation consumes the
    same exported interface as ordinary calls and never runs provider code.
    """
    import torch
    from .gpu.program import GPUProgram

    interface = artifact.interface
    if not isinstance(artifact.runtime, GPUProgram):
        raise NotImplementedError("PyTorch registration currently requires the GPU tensor interface")
    if any(isinstance(parameter, ViewParameter) and parameter.writable for parameter in interface.inputs):
        raise NotImplementedError("PyTorch registration currently requires read-only inputs and fresh Out tensors")
    if not interface.outputs:
        raise NotImplementedError("a functional PyTorch registration requires at least one Out tensor")
    if not any(isinstance(parameter, ViewParameter) for parameter in interface.inputs):
        raise NotImplementedError("PyTorch registration requires an input tensor to determine device dispatch")
    pieces = name.split("::")
    if len(pieces) != 2 or not all(piece.isidentifier() for piece in pieces):
        raise ValueError("operator name must be 'your_namespace::your_operator'")

    annotations = []
    for parameter in interface.inputs:
        if isinstance(parameter, ViewParameter):
            annotations.append("torch.Tensor")
        elif isinstance(parameter, ScalarParameter):
            if parameter.dtype.category is DTypeCategory.BOOL:
                annotations.append("bool")
            elif parameter.dtype.category in {DTypeCategory.SIGNED_INTEGER, DTypeCategory.UNSIGNED_INTEGER, DTypeCategory.INDEX}:
                annotations.append("int")
            elif parameter.dtype.category in {DTypeCategory.FLOAT, DTypeCategory.BFLOAT}:
                annotations.append("float")
            else:
                raise NotImplementedError(f"PyTorch scalar schema for {parameter.dtype}")
    parameters = ", ".join(f"arg{index}: {annotation}" for index, annotation in enumerate(annotations))
    arguments = ", ".join(f"arg{index}" for index in range(len(annotations)))
    returns = "torch.Tensor" if len(interface.outputs) == 1 else (
        "tuple[" + ", ".join("torch.Tensor" for _ in interface.outputs) + "]")

    def function(invoke):
        namespace = {"torch": torch, "_invoke": invoke}
        source = f"def operator({parameters}) -> {returns}:\n    return _invoke({arguments})\n"
        exec(compile(source, f"<intent PyTorch {name}>", "exec", dont_inherit=True), namespace)
        return namespace["operator"]

    operator = torch.library.custom_op(name, function(artifact.run), mutates_args=(), device_types="cuda")
    operator.register_fake(function(lambda *arguments: artifact.runtime.binding.bind(
        arguments, device=artifact.runtime.device, abstract=True).result()))
    return operator
