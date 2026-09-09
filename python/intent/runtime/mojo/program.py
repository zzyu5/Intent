from __future__ import annotations

import ctypes
from dataclasses import dataclass
import statistics

import torch

from .compilation import compile_library


def _timing_samples(measure, arguments: tuple[object, ...], *, samples: int) -> float:
    first = measure(*arguments, 1)
    if first <= 0:
        raise RuntimeError("native monotonic timing returned a non-positive duration")
    repetitions = min(50, max(1, int(10.0 / first)))
    return statistics.median(measure(*arguments, repetitions) for _ in range(samples))


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
            timings = tuple(_timing_samples(measure, self.native_arguments, samples=3)
                            for measure in self.program.measurements)
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


class NativeProgram:
    def __init__(self, source: str, metadata: dict[str, object], target) -> None:
        self.parameters = metadata["parameters"]
        self.candidates = metadata["candidates"]
        if not metadata["contiguous_views"] or not metadata["disjoint_outputs"]:
            raise NotImplementedError("Mojo runtime requires declared contiguous/disjoint-output entry legality")
        self.compilation = compile_library(source, metadata, target)
        argument_types = []
        for parameter in self.parameters:
            if parameter["kind"] == "view":
                argument_types.extend([ctypes.c_void_p, *([ctypes.c_int64] * (2 * len(parameter["shape"])))])
            else:
                argument_types.append(ctypes.c_float if parameter["dtype"] == "f32" else ctypes.c_int64)
        self.functions = []
        self.measurements = []
        for candidate in self.candidates:
            function = getattr(self.compilation.library, candidate["entry"])
            function.argtypes = argument_types
            function.restype = None
            self.functions.append(function)
            measure = getattr(self.compilation.library, candidate["entry"] + "_benchmark")
            measure.argtypes = [*argument_types, ctypes.c_int64]
            measure.restype = ctypes.c_double
            self.measurements.append(measure)
        self.winners: dict[tuple[object, ...], int] = {}
        self.timings: dict[tuple[object, ...], tuple[float, ...]] = {}

    def _view(self, parameter, tensor, dimensions: dict[int, int]) -> None:
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError(f"{parameter['name']} must be a CPU f32 tensor")
        if not tensor.is_contiguous():
            raise NotImplementedError("Mojo CPU non-contiguous views are not implemented")
        if tensor.numel() == 0:
            raise NotImplementedError("Mojo CPU empty-storage pointer ABI is not implemented")
        if tensor.ndim != len(parameter["shape"]):
            raise ValueError(f"{parameter['name']} has an incompatible rank")
        for axis, (static, identity) in enumerate(zip(parameter["shape"], parameter["dimensions"])):
            extent = tensor.shape[axis]
            if static >= 0 and static != extent:
                raise ValueError(f"{parameter['name']} has an incompatible static extent")
            if identity > 0 and identity in dimensions and dimensions[identity] != extent:
                raise ValueError("CPU views disagree on a canonical dimension identity")
            if identity > 0:
                dimensions[identity] = extent

    def prepare(self, arguments: tuple[object, ...], *, explicit_outputs: bool = False) -> NativeCall:
        expected = len(self.parameters) if explicit_outputs else sum(
            parameter["kind"] != "view" or parameter["access"] != 1 for parameter in self.parameters)
        if len(arguments) != expected:
            raise TypeError(f"expected {expected} CPU artifact arguments, got {len(arguments)}")
        dimensions: dict[int, int] = {}
        supplied = iter(arguments)
        all_arguments: list[object] = []
        for parameter in self.parameters:
            if parameter["kind"] == "view" and parameter["access"] == 1 and not explicit_outputs:
                all_arguments.append(None)
                continue
            value = next(supplied)
            if parameter["kind"] == "view":
                self._view(parameter, value, dimensions)
            all_arguments.append(value)
        outputs = []
        for index, parameter in enumerate(self.parameters):
            if parameter["kind"] != "view" or parameter["access"] != 1:
                continue
            if not explicit_outputs:
                shape = tuple(static if static >= 0 else dimensions[identity]
                              for static, identity in zip(parameter["shape"], parameter["dimensions"]))
                all_arguments[index] = torch.empty(shape, dtype=torch.float32, device="cpu")
                self._view(parameter, all_arguments[index], dimensions)
            outputs.append(all_arguments[index])
        views = [(parameter, value) for parameter, value in zip(self.parameters, all_arguments)
                 if parameter["kind"] == "view"]
        for index, (parameter, value) in enumerate(views):
            begin, end = value.data_ptr(), value.data_ptr() + value.numel() * value.element_size()
            for other_parameter, other in views[index + 1:]:
                same_allocation = value.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()
                overlap = begin < other.data_ptr() + other.numel() * other.element_size() and other.data_ptr() < end
                if overlap and (parameter["access"] == 1 or other_parameter["access"] == 1):
                    raise NotImplementedError("Mojo CPU overlapping writable views are not implemented")
                if same_allocation and (parameter["noalias"] or other_parameter["noalias"]):
                    raise ValueError("CPU invocation violates an author noalias constraint")
                if parameter["alias"] and parameter["alias"] == other_parameter["alias"] and not same_allocation:
                    raise ValueError("CPU invocation violates an author allocation-alias constraint")
        native: list[object] = []
        key: list[object] = []
        allocation_groups: dict[int, int] = {}
        for parameter, value in zip(self.parameters, all_arguments):
            if parameter["kind"] == "view":
                native.extend([value.data_ptr(), *value.shape, *value.stride()])
                allocation = value.untyped_storage().data_ptr()
                group = allocation_groups.setdefault(allocation, len(allocation_groups))
                key.append((tuple(value.shape), tuple(value.stride()), value.storage_offset(), value.dtype, group))
            else:
                native.append(value)
                key.append(parameter["dtype"])
        return NativeCall(self, tuple(all_arguments), tuple(native), tuple(outputs), tuple(key))

    def run(self, *arguments):
        call = self.prepare(arguments)
        call.launch()
        return call.result()

    def launch(self, *arguments):
        self.prepare(arguments, explicit_outputs=True).launch()
