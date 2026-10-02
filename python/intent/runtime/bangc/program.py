from __future__ import annotations

import ctypes
from dataclasses import dataclass
import statistics
from threading import RLock

from .buffer import DeviceBuffer, DeviceView, runtime
from .compilation import compile_library
from ..interface import ViewParameter
from ..invocation import ViewFacts, invocation_result
from ..native import NativePreparedRuntime
from ..native_artifact import load_native_library
from ..diagnostics import CacheObservation, ObservedCall, bindings, observation
from intent.compiler.toolchain import CompilationStageError


@dataclass
class NativeCall(ObservedCall):
    program: NativeProgram
    arguments: tuple[object, ...]
    native_arguments: tuple[object, ...]
    outputs: tuple[DeviceBuffer | DeviceView, ...]
    description: tuple

    def compile(self) -> None:
        """Compile the fixed entry without loading it or creating a CNRT queue."""
        try:
            self.program.compile()
        except CompilationStageError as error:
            details = self._observed("failed")
            self._record_observation(details)
            error.observation = details
            raise
        self._record_observation(self._observed("compiled"))

    def inspect_configurations(self):
        """BANG C has a fixed generated entry, not a runtime tuning portfolio."""
        return ()

    def _observed(self, stage):
        return observation("bangc", self.program.target_facts, self.description, None, (), stage=stage,
            history_unavailable="The generated DSA entry has fixed bindings; there is no runtime candidate search",
            caches=(CacheObservation("tuning", "not_applicable", None, "BANG C fixed entry", "selection"),
                    *self.program.compilation_observations()))

    def _validate(self) -> None:
        self.program.check_open()
        if any(isinstance(value, (DeviceBuffer, DeviceView)) and not value.pointer for value in self.arguments):
            raise ValueError("BANG C call refers to a closed device allocation")
        self.program.runtime.select(self.program.target.device)

    def _submit(self, queue: ctypes.c_void_p) -> None:
        status = self.program.function(queue, *self.native_arguments)
        if status:
            raise RuntimeError(f"BANG C kernel submission failed with CNRT status {status}")

    def enqueue(self, queue: ctypes.c_void_p | None = None) -> None:
        try:
            self.program.ensure_loaded()
            self._validate()
            self._submit(self.program.ensure_queue() if queue is None else queue)
        except CompilationStageError as error:
            details = self._observed("failed")
            self._record_observation(details)
            error.observation = details
            raise
        except Exception as error:
            details = self._observed("failed")
            self._record_observation(details)
            raise CompilationStageError("provider_invocation", str(error), observation=details) from error
        self._record_observation(self.observation if self.observation is not None and
                                 self.observation.stage == "submitted" else self._observed("submitted"))

    def launch(self) -> None:
        self.enqueue()
        try:
            self.program.runtime.invoke("cnrtQueueSync", self.program.queue)
        except Exception as error:
            details = self._observed("failed")
            self._record_observation(details)
            raise CompilationStageError("provider_invocation", str(error), observation=details) from error
        self._record_observation(self._observed("launched"))

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
    for call in calls:
        call.program.check_open()
        call.program.ensure_loaded()
    owner.ensure_queue()
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
        self.source = source
        self._state_lock = RLock()
        self._closed = False
        self._runtime = None
        self.compilation = None
        self.library = None
        self.function = None
        self.queue = ctypes.c_void_p()
        self.target_facts = bindings(contract.metadata["target"])
        self.facts = contract.facts
        self.candidates = ()
        self.configuration_descriptions = ()
        abi = contract.abi
        self._argument_types = (ctypes.c_void_p, *abi.argument_types())
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
            check_view_geometry=lambda owner, parameter, facts: requirements.check_geometry(parameter, facts),
            check_view_storage=lambda owner, parameter, facts: requirements.check_storage(parameter, facts),
            check_dimensions=type(self)._check_dimensions,
        )
        parameters_by_name = {parameter.name: parameter for parameter in self.interface.parameters}
        for name, shape in target.compilation.shapes:
            parameter = parameters_by_name.get(name)
            if not isinstance(parameter, ViewParameter) or len(shape) != len(parameter.shape):
                raise ValueError("BANG C artifact and target disagree on bound parameter shapes")
            if any(extent >= 0 and extent != declared for extent, declared in zip(shape, parameter.shape)):
                raise ValueError("BANG C artifact and target disagree on bound extents")

    def check_open(self) -> None:
        if self._closed:
            raise ValueError("BANG C program is closed")

    @property
    def runtime(self):
        with self._state_lock:
            self.check_open()
            if self._runtime is None:
                self._runtime = runtime(self.target.neuware)
            return self._runtime

    def compile(self):
        with self._state_lock:
            self.check_open()
            if self.compilation is None:
                self.compilation = compile_library(self.source, self.target)
            return self.compilation

    def ensure_loaded(self) -> None:
        with self._state_lock:
            self.check_open()
            if self.function is not None:
                return
            compilation = self.compile()
            library = load_native_library(compilation)
            try:
                function = library.bind(self.facts.entry, self._argument_types, ctypes.c_int)
            except Exception:
                library.close()
                raise
            self.library, self.function = library, function

    def ensure_queue(self) -> ctypes.c_void_p:
        with self._state_lock:
            self.check_open()
            if not self.queue.value:
                self.runtime.select(self.target.device)
                self.runtime.invoke("cnrtQueueCreate", ctypes.byref(self.queue))
            return self.queue

    def compilation_observations(self):
        if self.compilation is None:
            return ()
        artifact = self.compilation
        return (CacheObservation("native_compilation", "persistent_artifact", artifact.cache_hit,
                                 "BANG C native artifact", "native_compilation",
                                 artifact.cache_reason, str(artifact.directory)),)

    def prepare(self, arguments: tuple[object, ...], *, explicit_outputs: bool = False) -> NativeCall:
        self.check_open()
        bound = self._binders[bool(explicit_outputs)](self, arguments)
        return NativeCall(self, bound.arguments, bound.native_arguments, bound.outputs,
                          self.describe_arguments(bound, f"mlu:{self.target.device}"))

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
        with self._state_lock:
            if self._closed:
                return
            if self.queue.value:
                self.runtime.invoke("cnrtSetDevice", self.target.device)
                self.runtime.invoke("cnrtQueueSync", self.queue)
                self.runtime.invoke("cnrtQueueDestroy", self.queue)
                self.queue = ctypes.c_void_p()
            if self.library is not None:
                self.library.close()
                self.library = None
                self.function = None
            self._closed = True

    def __del__(self):
        queue = getattr(self, "queue", None)
        if queue and queue.value:
            self._runtime.library.cnrtSetDevice(self.target.device)
            self._runtime.library.cnrtQueueSync(queue)
            self._runtime.library.cnrtQueueDestroy(queue)
        library = getattr(self, "library", None)
        if library is not None:
            library.close()
