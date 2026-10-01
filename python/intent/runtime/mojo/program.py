from __future__ import annotations

import ctypes
from dataclasses import dataclass
from math import prod
import statistics

import torch

from .compilation import compile_library
from ..cpu import check_alias
from ..interface import ViewParameter
from ..native import NativeABI, NativePreparedRuntime, ViewFacts
from ..torch import torch_dtype


_winners: dict[tuple[object, ...], dict[tuple[object, ...], int]] = {}
_candidate_timings: dict[tuple[object, ...], dict[tuple[object, ...], tuple[float, ...]]] = {}

def _timing_samples(measure, arguments: tuple[object, ...], *, samples: int) -> float:
    first = measure(*arguments, 1)
    if first <= 0:
        raise RuntimeError("native monotonic timing returned a non-positive duration")
    repetitions = min(50, max(1, int(10.0 / first)))
    return statistics.median(measure(*arguments, repetitions) for _ in range(samples))


def _measure_candidates(measurements, arguments: tuple[object, ...]) -> tuple[float, ...]:
    probes = [measure(*arguments, 1) for measure in measurements]
    if any(elapsed <= 0 for elapsed in probes):
        raise RuntimeError("native monotonic timing returned a non-positive duration")
    repetitions = [min(50, max(1, int(10.0 / elapsed))) for elapsed in probes]
    samples = [[] for _ in measurements]
    for round_index in range(3):
        order = list(range(len(measurements)))
        if round_index == 1:
            order.reverse()
        elif round_index == 2:
            order = order[1:] + order[:1]
        for candidate in order:
            samples[candidate].append(measurements[candidate](*arguments, repetitions[candidate]))
    return tuple(statistics.median(values) for values in samples)


@dataclass
class NativeCall:
    program: NativeProgram
    arguments: tuple[object, ...]
    native_arguments: tuple[object, ...]
    outputs: tuple[torch.Tensor, ...]
    key: tuple[object, ...]
    winner: int | None = None

    def choose(self) -> int:
        if self.winner is not None:
            return self.winner
        if self.key not in self.program.winners:
            timings = _measure_candidates(self.program.measurements, self.native_arguments)
            self.program.winners[self.key] = min(range(len(timings)), key=timings.__getitem__)
            self.program.timings[self.key] = timings
        self.winner = self.program.winners[self.key]
        return self.winner

    def launch(self) -> None:
        self.program.functions[self.choose()](*self.native_arguments)

    def benchmark(self) -> float:
        return _timing_samples(self.program.measurements[self.choose()], self.native_arguments, samples=7)

    def result(self):
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs


class NativeProgram(NativePreparedRuntime):
    def __init__(self, source: str, metadata: dict[str, object], target) -> None:
        abi = NativeABI.read(metadata)
        self.interface = abi.interface
        self._view_dtypes = tuple(
            torch_dtype(parameter.dtype)
            if isinstance(parameter, ViewParameter) else None
            for parameter in self.interface.parameters
        )
        self.candidates = metadata["candidates"]
        self.contiguous_views = metadata["native"]["contiguous_views"]
        if not metadata["native"]["disjoint_outputs"]:
            raise NotImplementedError("Mojo runtime requires declared disjoint-output entry legality")
        self._binders = abi.binders(
            observe_view=type(self)._view, allocate_output=type(self)._allocate_output,
            check_alias=check_alias,
            view_key=lambda facts, group: (facts.shape, facts.strides, facts.offset, facts.dtype, group),
            scalar_key=lambda parameter, value: parameter.dtype.name,
        )
        self.compilation = compile_library(source, metadata, target, abi=abi)
        argument_types = abi.argument_types()
        self.functions = []
        self.measurements = []
        for candidate, compilation in zip(self.candidates, self.compilation.libraries):
            function = getattr(compilation.library, candidate["entry"])
            function.argtypes = argument_types
            function.restype = None
            self.functions.append(function)
            measure = getattr(compilation.library, candidate["entry"] + "_benchmark")
            measure.argtypes = [*argument_types, ctypes.c_int64]
            measure.restype = ctypes.c_double
            self.measurements.append(measure)
        self.winners = _winners.setdefault(self.compilation.identity, {})
        self.timings = _candidate_timings.setdefault(self.compilation.identity, {})

    def _view(self, parameter: ViewParameter, tensor) -> ViewFacts:
        dtype = self._view_dtypes[parameter.position]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != dtype:
            raise ValueError(f"{parameter.name} must be a CPU {parameter.dtype} tensor")
        if tensor.numel() == 0:
            raise NotImplementedError("Mojo CPU empty-storage pointer ABI is not implemented")
        shape, strides = tuple(tensor.shape), tuple(tensor.stride())
        if len(shape) != len(parameter.shape):
            raise ValueError(f"{parameter.name} has an incompatible rank")
        if self.contiguous_views or parameter.access != 0:
            expected_stride = 1
            for extent, stride in zip(reversed(shape), reversed(strides)):
                if stride != expected_stride:
                    raise NotImplementedError("Mojo CPU writable views require canonical contiguous strides")
                expected_stride *= extent
        pointer = tensor.data_ptr()
        element_size = tensor.element_size()
        lower, upper = 0, 1
        for extent, stride in zip(shape, strides):
            offset = (extent - 1) * stride
            lower += min(0, offset)
            upper += max(0, offset)
        storage = tensor.untyped_storage()
        return ViewFacts(shape, strides, pointer, storage.data_ptr(), storage.data_ptr() + storage.nbytes(),
                         tensor.storage_offset(), dtype, pointer + lower * element_size,
                         pointer + upper * element_size)

    def _allocate_output(self, parameter: ViewParameter, shape: tuple[int, ...]) -> tuple[torch.Tensor, ViewFacts]:
        dtype = self._view_dtypes[parameter.position]
        tensor = torch.empty(shape, dtype=dtype, device="cpu")
        elements = prod(shape)
        if elements == 0:
            raise NotImplementedError("Mojo CPU empty-storage pointer ABI is not implemented")
        strides = tensor.stride()
        pointer = tensor.data_ptr()
        # This invocation owns the new contiguous allocation. Shape, dtype,
        # zero offset and storage identity follow from the factory contract.
        facts = ViewFacts(shape, strides, pointer, pointer, pointer + tensor.untyped_storage().nbytes(), 0, dtype,
                          pointer, pointer + elements * tensor.element_size())
        return tensor, facts

    def prepare(self, arguments: tuple[object, ...], *, explicit_outputs: bool = False) -> NativeCall:
        bound = self._binders[bool(explicit_outputs)](self, arguments)
        return NativeCall(self, bound.arguments, bound.native_arguments, bound.outputs, bound.key)

    def run(self, *arguments):
        call = self.prepare(arguments)
        call.launch()
        return call.result()

    def launch(self, *arguments):
        self.prepare(arguments, explicit_outputs=True).launch()
