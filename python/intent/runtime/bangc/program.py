from __future__ import annotations

import ctypes
from dataclasses import dataclass
import statistics

from .buffer import DeviceBuffer, DeviceView, runtime
from .compilation import compile_library
from ..native import AliasCheck, NativeInterface, NativePreparedRuntime, ScalarParameter, ViewFacts, ViewParameter


_SCALAR_CTYPES = {"f32": ctypes.c_float, "i64": ctypes.c_int64,
                  "i32": ctypes.c_int32, "bool": ctypes.c_bool}


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
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs

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
    def __init__(self, source: str, metadata: dict[str, object], target) -> None:
        if metadata.get("provider") != "bangc" or metadata.get("architecture") != target.architecture:
            raise ValueError("BANG C artifact and target disagree")
        for binding in ("tile", "tile_m", "tile_n", "tile_k", "region_tile", "tasks", "local_bytes"):
            if metadata[binding] != getattr(target, binding):
                raise ValueError(f"BANG C artifact and target disagree on {binding}")
        self.target = target
        self.metadata = metadata
        self.parameters = metadata["parameters"]
        self.interface = NativeInterface.read(self.parameters)
        self._binders = self.interface.binders(
            observe_view=type(self)._view, allocate_output=type(self)._allocate_output,
            check_alias=type(self)._check_alias, scalar_argument=type(self)._scalar,
            check_dimensions=type(self)._check_dimensions,
        )
        parameters_by_name = {parameter["name"]: parameter for parameter in self.parameters}
        for name, shape in target.shapes:
            parameter = parameters_by_name.get(name)
            if parameter is None or parameter["kind"] != "view" or len(shape) != len(parameter["shape"]):
                raise ValueError("BANG C artifact and target disagree on bound parameter shapes")
            if any(extent >= 0 and extent != declared for extent, declared in zip(shape, parameter["shape"])):
                raise ValueError("BANG C artifact and target disagree on bound extents")
        self.compilation = compile_library(source, target)
        self.runtime = runtime(target.neuware)
        self.runtime.select(target.device)
        self.queue = ctypes.c_void_p()
        self.runtime.invoke("cnrtQueueCreate", ctypes.byref(self.queue))
        self.function = getattr(self.compilation.library, metadata["entry"])
        self.function.argtypes = (ctypes.c_void_p, *self.interface.argument_types(_SCALAR_CTYPES.__getitem__))
        self.function.restype = ctypes.c_int

    def run(self, *arguments):
        call = self.prepare(arguments)
        call.launch()
        return call.result()

    def launch(self, *arguments):
        self.prepare(arguments, explicit_outputs=True).launch()

    def prepare(self, arguments: tuple[object, ...], *, explicit_outputs: bool = False) -> NativeCall:
        bound = self._binders[bool(explicit_outputs)](self, arguments)
        # BANG C run returns only declared Out buffers. CPU runtimes also return
        # their InOut state; that difference is not part of the flattened ABI.
        outputs = tuple(bound.arguments[parameter.position] for parameter in self.interface.allocated_outputs)
        return NativeCall(self, bound.arguments, bound.native_arguments, outputs)

    def _view(self, parameter: ViewParameter, value) -> ViewFacts:
        if not isinstance(value, (DeviceBuffer, DeviceView)) or not value.pointer:
            raise TypeError(f"{parameter.name} requires a live MLU buffer or view")
        if value.device != self.target.device or value.runtime is not self.runtime:
            raise ValueError("BANG C arguments must belong to the artifact's MLU device and runtime")
        if value.dtype != parameter.dtype or len(value.shape) != len(parameter.shape):
            raise ValueError(f"{parameter.name} has incompatible shape or dtype")
        lower, upper = value.byte_bounds
        return ViewFacts(value.shape, value.strides, value.pointer, value.allocation_pointer,
                         value.pointer - value.allocation_pointer, value.dtype, lower, upper)

    def _allocate_output(self, parameter: ViewParameter, shape: tuple[int, ...]) -> tuple[DeviceBuffer, ViewFacts]:
        value = DeviceBuffer(shape, parameter.dtype, device=self.target.device, neuware=self.target.neuware)
        if parameter.stride_mismatch(value.strides):
            value.close()
            raise NotImplementedError("automatic BANG C outputs require contiguous declared strides; launch accepts explicit strided output views")
        lower, upper = value.byte_bounds
        return value, ViewFacts(value.shape, value.strides, value.pointer, value.allocation_pointer,
                                0, value.dtype, lower, upper)

    def _check_dimensions(self, dimensions: dict[int, int]) -> None:
        for identity in self.metadata["full_extent_dimensions"]:
            if dimensions[identity] > self.metadata["tile"]:
                raise NotImplementedError("this row reduction requires a larger DSA tile binding")

    @staticmethod
    def _scalar(parameter: ScalarParameter, value):
        return float(value) if parameter.dtype == "f32" else int(value)

    @staticmethod
    def _check_alias(check: AliasCheck, left: ViewFacts, right: ViewFacts) -> None:
        if check.noalias_violation(left, right):
            raise ValueError("MLU views violate a declared noalias contract")
        if check.writable_overlap(left, right):
            raise NotImplementedError("BANG C currently requires nonoverlapping writable views")
        if check.allocation_violation(left, right):
            raise ValueError("MLU views violate a declared allocation alias relation")

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
