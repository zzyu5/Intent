from __future__ import annotations

from inspect import Parameter, Signature
from typing import Protocol, runtime_checkable

from ..language.dtypes import DType, DTypeCategory
from .interface import ScalarParameter, ViewParameter


@runtime_checkable
class TorchOutputInference(Protocol):
    """Infer Torch outputs from the public interface without native execution."""

    def infer_outputs(self, arguments: tuple) -> object: ...


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

    interface = artifact.interface
    if artifact.device_type not in {"cpu", "cuda"} or not isinstance(artifact.runtime, TorchOutputInference):
        raise NotImplementedError("PyTorch registration requires a CPU or CUDA tensor runtime with output inference")
    if any(isinstance(parameter, ViewParameter) and parameter.writable for parameter in interface.inputs):
        raise NotImplementedError("PyTorch registration currently requires read-only inputs and fresh Out tensors")
    if not interface.outputs:
        raise NotImplementedError("a functional PyTorch registration requires at least one Out tensor")
    if not any(isinstance(parameter, ViewParameter) for parameter in interface.inputs):
        raise NotImplementedError("PyTorch registration requires an input tensor to determine device dispatch")
    pieces = name.split("::")
    if len(pieces) != 2 or not all(piece.isidentifier() for piece in pieces):
        raise ValueError("operator name must be 'your_namespace::your_operator'")

    parameters = []
    for parameter in interface.inputs:
        if isinstance(parameter, ViewParameter):
            scalar = "Tensor"
        elif isinstance(parameter, ScalarParameter):
            if parameter.dtype.category is DTypeCategory.BOOL:
                scalar = "bool"
            elif parameter.dtype.category in {DTypeCategory.SIGNED_INTEGER, DTypeCategory.UNSIGNED_INTEGER, DTypeCategory.INDEX}:
                scalar = "SymInt"
            elif parameter.dtype.category in {DTypeCategory.FLOAT, DTypeCategory.BFLOAT}:
                scalar = "float"
            else:
                raise NotImplementedError(f"PyTorch scalar schema for {parameter.dtype}")
        parameters.append(f"{scalar} arg{len(parameters)}")
    returns = "Tensor" if len(interface.outputs) == 1 else (
        "(" + ", ".join("Tensor" for _ in interface.outputs) + ")")
    schema = f"({', '.join(parameters)}) -> {returns}"
    signature = Signature(Parameter(f"arg{index}", Parameter.POSITIONAL_OR_KEYWORD)
                          for index in range(len(parameters)))

    def function(invoke):
        def operator(*arguments, **keywords):
            return invoke(*signature.bind(*arguments, **keywords).args)

        return operator

    operator = torch.library.custom_op(name, function(artifact.run), mutates_args=(),
                                       device_types=artifact.device_type, schema=schema)
    operator.register_fake(function(lambda *arguments: artifact.runtime.infer_outputs(arguments)))
    return operator
