from __future__ import annotations

from .interface import ViewParameter
from .invocation import ViewFacts
from .torch import torch_dtype


def view_byte_span(view) -> tuple[int, int]:
    pointer = view.data_ptr()
    elements = view.numel()
    if elements == 0:
        return pointer, pointer
    if view.is_contiguous():
        return pointer, pointer + elements * view.element_size()
    low = high = 0
    for extent, stride in zip(view.shape, view.stride()):
        displacement = (extent - 1) * stride
        low += min(0, displacement)
        high += max(0, displacement)
    size = view.element_size()
    return pointer + low * size, pointer + (high + 1) * size


def observe_view(parameter: ViewParameter, value, *, device, abstract: bool = False) -> ViewFacts:
    """Observe a tensor's public geometry; concrete binding also observes storage.

    Layout, alignment and writable overlap requirements belong to the compiled
    entry, independently of these observations and the author's interface.
    """
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{parameter.name} must be a torch.Tensor")
    expected = torch_dtype(parameter.dtype)
    if value.dtype != expected:
        raise TypeError(f"{parameter.name} must have dtype {expected}, got {value.dtype}")
    if value.ndim != len(parameter.shape):
        raise ValueError(f"{parameter.name} must have rank {len(parameter.shape)}, got {value.ndim}")
    if value.device != device:
        raise ValueError(f"{parameter.name} must be on {device}, got {value.device}")
    shape, strides = tuple(value.shape), tuple(value.stride())
    if abstract:
        return ViewFacts(shape, strides, None, None, None, None, value.dtype, None, None, None)
    storage = value.untyped_storage()
    allocation = storage.data_ptr()
    # Independently wrapped external storage can have distinct StorageImpls
    # over the same allocation. Empty allocations instead need a real handle:
    # their zero pointer does not identify shared ownership.
    identity = (value.device, allocation) if storage.nbytes() else (value.device, "empty", storage._cdata)
    begin, end = view_byte_span(value)
    return ViewFacts(shape, strides, value.data_ptr(), allocation, allocation + storage.nbytes(),
                     value.storage_offset(), value.dtype, begin, end, identity)


def allocate_output(parameter: ViewParameter, shape: tuple, *, device,
                    abstract: bool = False) -> tuple[object, ViewFacts]:
    import torch

    value = torch.empty(shape, dtype=torch_dtype(parameter.dtype), device=device)
    return value, observe_view(parameter, value, device=device, abstract=abstract)


def check_abstract_relation(actual, expected, message: str) -> None:
    import torch

    torch._check(actual == expected, lambda: message)
