from __future__ import annotations

from .gpu.interface import GPUInterface, Scalar, View


def register_operator(artifact, name: str):
    """Register a functional allocating invocation with PyTorch's dispatcher.

    Device execution is opaque to PyTorch. The fake implementation consumes the
    same exported interface as ordinary calls and never runs provider code.
    """
    import torch

    interface = artifact.interface
    if not isinstance(interface, GPUInterface):
        raise NotImplementedError("PyTorch registration currently requires the GPU tensor interface")
    if any(isinstance(parameter, View) and parameter.writable for parameter in interface.inputs):
        raise NotImplementedError("PyTorch registration currently requires read-only inputs and fresh Out tensors")
    if not interface.outputs:
        raise NotImplementedError("a functional PyTorch registration requires at least one Out tensor")
    if not any(isinstance(parameter, View) for parameter in interface.inputs):
        raise NotImplementedError("PyTorch registration requires an input tensor to determine device dispatch")
    pieces = name.split("::")
    if len(pieces) != 2 or not all(piece.isidentifier() for piece in pieces):
        raise ValueError("operator name must be 'your_namespace::your_operator'")

    annotations = []
    for parameter in interface.inputs:
        if isinstance(parameter, View):
            annotations.append("torch.Tensor")
        elif isinstance(parameter, Scalar):
            if parameter.dtype in {"i1", "bool"}:
                annotations.append("bool")
            elif parameter.dtype.startswith(("i", "u")):
                annotations.append("int")
            elif parameter.dtype.startswith(("f", "bf")):
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
    operator.register_fake(function(lambda *arguments: interface.bind(arguments, abstract=True).result()))
    return operator
