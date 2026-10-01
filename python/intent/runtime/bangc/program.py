from __future__ import annotations

import ctypes
from dataclasses import dataclass
import statistics

from .buffer import DeviceBuffer, DeviceView, runtime
from .compilation import compile_library


@dataclass
class NativeCall:
    program: NativeProgram
    arguments: tuple[object, ...]
    native_arguments: tuple[object, ...]
    outputs: tuple[DeviceBuffer, ...]

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


class NativeProgram:
    def __init__(self, source: str, metadata: dict[str, object], target) -> None:
        if metadata.get("provider") != "bangc" or metadata.get("architecture") != target.architecture:
            raise ValueError("BANG C artifact and target disagree")
        for binding in ("tile", "tile_m", "tile_n", "tile_k", "region_tile", "tasks", "local_bytes"):
            if metadata[binding] != getattr(target, binding):
                raise ValueError(f"BANG C artifact and target disagree on {binding}")
        self.target = target
        self.metadata = metadata
        self.parameters = metadata["parameters"]
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
        argument_types = [ctypes.c_void_p]
        for parameter in self.parameters:
            if parameter["kind"] == "view":
                argument_types.extend([ctypes.c_void_p, *([ctypes.c_int64] * (2 * len(parameter["shape"])))])
            else:
                argument_types.append({"f32": ctypes.c_float, "i64": ctypes.c_int64,
                                       "i32": ctypes.c_int32, "bool": ctypes.c_bool}[parameter["dtype"]])
        self.function.argtypes = argument_types
        self.function.restype = ctypes.c_int

    def run(self, *arguments):
        call = self.prepare(arguments)
        call.launch()
        return call.result()

    def launch(self, *arguments):
        self.prepare(arguments, explicit_outputs=True).launch()

    def prepare(self, arguments: tuple[object, ...], *, explicit_outputs: bool = False) -> NativeCall:
        expected = len(self.parameters) if explicit_outputs else sum(
            p["kind"] != "view" or p["access"] != 1 for p in self.parameters)
        if len(arguments) != expected:
            raise TypeError(f"expected {expected} BANG C arguments, got {len(arguments)}")
        supplied = iter(arguments)
        bound = []
        dimensions: dict[int, int] = {}
        for parameter in self.parameters:
            output = parameter["kind"] == "view" and parameter["access"] == 1
            value = None if output and not explicit_outputs else next(supplied)
            if parameter["kind"] == "view" and value is not None:
                if not isinstance(value, (DeviceBuffer, DeviceView)) or not value.pointer:
                    raise TypeError(f"{parameter['name']} requires a live MLU buffer or view")
                if value.device != self.target.device or value.runtime is not self.runtime:
                    raise ValueError("BANG C arguments must belong to the artifact's MLU device and runtime")
                if value.dtype != parameter["dtype"] or len(value.shape) != len(parameter["shape"]):
                    raise ValueError(f"{parameter['name']} has incompatible shape or dtype")
                for size, fixed, identity in zip(value.shape, parameter["shape"], parameter["dimensions"]):
                    if fixed >= 0 and size != fixed:
                        raise ValueError("MLU argument violates a static extent")
                    if identity > 0:
                        if identity in dimensions and dimensions[identity] != size:
                            raise ValueError("MLU arguments disagree on a logical dimension")
                        dimensions[identity] = size
                for fixed, stride in zip(parameter["strides"], value.strides):
                    if fixed is not None and fixed != stride:
                        raise ValueError("MLU argument violates its declared strides")
            bound.append(value)
        for identity in self.metadata["full_extent_dimensions"]:
            if dimensions[identity] > self.metadata["tile"]:
                raise NotImplementedError("this row reduction requires a larger DSA tile binding")
        outputs = []
        native = []
        views = []
        for index, (parameter, value) in enumerate(zip(self.parameters, bound)):
            if parameter["kind"] != "view":
                native.append(float(value) if parameter["dtype"] == "f32" else int(value))
                continue
            if value is None:
                missing = [identity for fixed, identity in zip(parameter["shape"], parameter["dimensions"])
                           if fixed < 0 and identity not in dimensions]
                if missing:
                    raise ValueError(f"cannot infer {parameter['name']} output dimensions {missing} from supplied inputs")
                shape = tuple(fixed if fixed >= 0 else dimensions[identity]
                              for fixed, identity in zip(parameter["shape"], parameter["dimensions"]))
                value = DeviceBuffer(shape, parameter["dtype"], device=self.target.device, neuware=self.target.neuware)
                bound[index] = value
                if any(fixed is not None and fixed != stride for fixed, stride in zip(parameter["strides"], value.strides)):
                    value.close()
                    raise NotImplementedError("automatic BANG C outputs require contiguous declared strides; launch accepts explicit strided output views")
            for previous, previous_parameter in views:
                lower, upper = value.byte_bounds
                previous_lower, previous_upper = previous.byte_bounds
                overlap = lower < previous_upper and previous_lower < upper
                if value.allocation_pointer == previous.allocation_pointer and (parameter["noalias"] or previous_parameter["noalias"]):
                    raise ValueError("MLU views violate a declared noalias contract")
                if overlap and (parameter["access"] != 0 or previous_parameter["access"] != 0):
                    raise NotImplementedError("BANG C currently requires nonoverlapping writable views")
                if parameter["alias"] and parameter["alias"] == previous_parameter["alias"] and value.allocation_pointer != previous.allocation_pointer:
                    raise ValueError("MLU views violate a declared allocation alias relation")
            views.append((value, parameter))
            native.extend((value.pointer, *value.shape, *value.strides))
            if parameter["access"] == 1:
                outputs.append(value)
        return NativeCall(self, tuple(bound), tuple(native), tuple(outputs))

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
