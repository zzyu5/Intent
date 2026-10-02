from __future__ import annotations

import ctypes
from dataclasses import dataclass
import statistics

from .buffer import DeviceBuffer, DeviceView, runtime
from .compilation import compile_library
from ..interface import ViewParameter
from ..invocation import ViewFacts, invocation_result
from ..native import NativePreparedRuntime


@dataclass
class NativeCall:
    program: NativeProgram
    arguments: tuple[object, ...]
    native_arguments: tuple[object, ...]
    outputs: tuple[DeviceBuffer | DeviceView, ...]

    def _validate(self) -> None:
        if not self.program.queue.value:
            raise ValueError("BANG C program is closed")
        if any(isinstance(value, (DeviceBuffer, DeviceView)) and not value.pointer for value in self.arguments):
            raise ValueError("BANG C call refers to a closed device allocation")
        self.program.runtime.select(self.program.target.device)

    def _submit(self, queue: ctypes.c_void_p) -> None:
        status = self.program.function(queue, *self.native_arguments)
        if status:
            raise RuntimeError(f"BANG C kernel submission failed with CNRT status {status}")

    def enqueue(self, queue: ctypes.c_void_p | None = None) -> None:
        self._validate()
        self._submit(self.program.queue if queue is None else queue)

    def launch(self) -> None:
        self.enqueue()
        self.program.runtime.invoke("cnrtQueueSync", self.program.queue)

    def result(self):
        return invocation_result(self.outputs)

    def benchmark(self) -> float:
        return benchmark_calls((self,))


def _queue_owner(calls: tuple[NativeCall, ...]):
    if not calls:
        raise ValueError("a BANG C launch sequence must contain at least one call")
    owner = calls[0].program
    if any(call.program.runtime is not owner.runtime or call.program.target.device != owner.target.device for call in calls):
        raise ValueError("a BANG C launch sequence must use one runtime and device")
    if not owner.queue.value:
        raise ValueError("BANG C launch sequence queue is closed")
    return owner


def launch_calls(calls: tuple[NativeCall, ...]) -> None:
    """Execute an explicitly supplied host sequence on one CNRT queue."""
    owner = _queue_owner(calls)
    for call in calls:
        call.enqueue(owner.queue)
    owner.runtime.invoke("cnrtQueueSync", owner.queue)


def benchmark_calls(calls: tuple[NativeCall, ...], *, prepare=None, repetitions: int = 10) -> float:
    owner = _queue_owner(calls)
    if repetitions <= 0:
        raise ValueError("BANG C measurement repetitions must be positive")
    owner.runtime.select(owner.target.device)
    start, end = ctypes.c_void_p(), ctypes.c_void_p()
    owner.runtime.invoke("cnrtNotifierCreate", ctypes.byref(start))
    try:
        owner.runtime.invoke("cnrtNotifierCreate", ctypes.byref(end))
        samples = []
        for _ in range(repetitions):
            if prepare is not None:
                prepare()
            for call in calls:
                call._validate()
            owner.runtime.invoke("cnrtPlaceNotifier", start, owner.queue)
            for call in calls:
                call._submit(owner.queue)
            owner.runtime.invoke("cnrtPlaceNotifier", end, owner.queue)
            owner.runtime.invoke("cnrtQueueSync", owner.queue)
            microseconds = ctypes.c_float()
            owner.runtime.invoke("cnrtNotifierDuration", start, end, ctypes.byref(microseconds))
            samples.append(microseconds.value * 1e-3)
        return statistics.median(samples)
    finally:
        owner.runtime.invoke("cnrtQueueSync", owner.queue)
        if end.value:
            owner.runtime.invoke("cnrtNotifierDestroy", end)
        owner.runtime.invoke("cnrtNotifierDestroy", start)


class NativeProgram(NativePreparedRuntime):
    def __init__(self, source: str, contract, target) -> None:
        from intent.targets.specification import require_matching_target

        require_matching_target(contract.target, target.compilation)
        self.target = target
        self.facts = contract.facts
        abi = contract.abi
        self.interface = abi.interface
        self.requirements = requirements = abi.requirements
        converters = {"bool": bool, "i8": int, "i16": int, "i32": int, "i64": int,
                      "f32": float, "f64": float}
        scalar_converters = {slot.parameter.position: converters[slot.carrier]
                             for slot in abi.slots if slot.role == "scalar"}
        self._binders = abi.binders(
            observe_view=type(self)._view, allocate_output=type(self)._allocate_output,
            check_alias=requirements.check_pair,
            scalar_argument=lambda parameter, value: scalar_converters[parameter.position](value),
            check_view_requirements=lambda owner, parameter, facts: requirements.check_view(parameter, facts),
            check_dimensions=type(self)._check_dimensions,
        )
        parameters_by_name = {parameter.name: parameter for parameter in self.interface.parameters}
        for name, shape in target.compilation.shapes:
            parameter = parameters_by_name.get(name)
            if not isinstance(parameter, ViewParameter) or len(shape) != len(parameter.shape):
                raise ValueError("BANG C artifact and target disagree on bound parameter shapes")
            if any(extent >= 0 and extent != declared for extent, declared in zip(shape, parameter.shape)):
                raise ValueError("BANG C artifact and target disagree on bound extents")
        self.compilation = compile_library(source, target)
        self.runtime = runtime(target.neuware)
        self.runtime.select(target.device)
        self.queue = ctypes.c_void_p()
        self.runtime.invoke("cnrtQueueCreate", ctypes.byref(self.queue))
        self.function = getattr(self.compilation.library, self.facts.entry)
        self.function.argtypes = (ctypes.c_void_p, *abi.argument_types())
        self.function.restype = ctypes.c_int

    def prepare(self, arguments: tuple[object, ...], *, explicit_outputs: bool = False) -> NativeCall:
        bound = self._binders[bool(explicit_outputs)](self, arguments)
        return NativeCall(self, bound.arguments, bound.native_arguments, bound.outputs)

    def _view(self, parameter: ViewParameter, value) -> ViewFacts:
        if not isinstance(value, (DeviceBuffer, DeviceView)) or not value.pointer:
            raise TypeError(f"{parameter.name} requires a live MLU buffer or view")
        if value.device != self.target.device or value.runtime is not self.runtime:
            raise ValueError("BANG C arguments must belong to the artifact's MLU device and runtime")
        if value.dtype != parameter.dtype.name or len(value.shape) != len(parameter.shape):
            raise ValueError(f"{parameter.name} has incompatible shape or dtype")
        lower, upper = value.byte_bounds
        owner = value.owner if isinstance(value, DeviceView) else value
        return ViewFacts(value.shape, value.strides, value.pointer, value.allocation_pointer,
                         owner.pointer + owner.nbytes,
                         value.pointer - value.allocation_pointer, value.dtype, lower, upper, value.allocation_pointer)

    def _allocate_output(self, parameter: ViewParameter, shape: tuple[int, ...]) -> tuple[DeviceBuffer, ViewFacts]:
        value = DeviceBuffer(shape, parameter.dtype.name, device=self.target.device, neuware=self.target.neuware)
        lower, upper = value.byte_bounds
        return value, ViewFacts(value.shape, value.strides, value.pointer, value.allocation_pointer,
                                value.pointer + value.nbytes,
                                0, value.dtype, lower, upper, value.allocation_pointer)

    def _check_dimensions(self, dimensions: dict[int, int]) -> None:
        for identity in self.facts.full_extent_dimensions:
            if dimensions[identity] > self.target.compilation.tile:
                raise NotImplementedError("this row reduction requires a larger DSA tile binding")

    def close(self) -> None:
        if self.queue.value:
            self.runtime.invoke("cnrtSetDevice", self.target.device)
            self.runtime.invoke("cnrtQueueSync", self.queue)
            self.runtime.invoke("cnrtQueueDestroy", self.queue)
            self.queue = ctypes.c_void_p()

    def __del__(self):
        queue = getattr(self, "queue", None)
        if queue and queue.value:
            self.runtime.library.cnrtSetDevice(self.target.device)
            self.runtime.library.cnrtQueueSync(queue)
            self.runtime.library.cnrtQueueDestroy(queue)
