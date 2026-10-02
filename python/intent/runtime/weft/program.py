from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys

from .buffer import Buffer
from .compilation import validate_artifact
from .target import TargetProfile, matrix_capability
from ..contract import ProgramContract
from ..interface import ViewParameter
from ..invocation import ViewFacts, invocation_result
from ..native import NativePreparedRuntime
from ..diagnostics import CacheObservation, CandidateObservation, ObservedCall, bindings, observation
from intent.compiler.toolchain import CompilationStageError


def _isa_extensions(isa: str) -> set[str]:
    base, *extensions = isa.split("_")
    if not base.startswith("rv64"):
        raise ValueError("native CPU does not report an RV64 ISA")
    result = set(base[4:]) | set(extensions)
    if "g" in result:
        result.remove("g")
        result |= {"i", "m", "a", "f", "d", "zicsr", "zifencei"}
    return result


def _cpu_facts() -> dict[int, dict[str, str]]:
    facts = {}
    for paragraph in Path("/proc/cpuinfo").read_text().split("\n\n"):
        fields = dict(line.split(":", 1) for line in paragraph.splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if "processor" in fields:
            facts[int(fields["processor"])] = fields
    return facts


class _ExecutionContract:
    """Artifact-local hardware facts, with calling-thread state checked on use."""

    def __init__(self, profile: TargetProfile, used_extensions: frozenset[str]) -> None:
        if sys.platform != "linux" or platform.machine() != "riscv64" or sys.byteorder != "little":
            raise NotImplementedError("Weft native loading requires little-endian RISC-V Linux")
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise ValueError("native pointer size disagrees with RV64/lp64d")
        self.profile = profile
        self.required = _isa_extensions(profile.march)
        self.capabilities = tuple((extension, matrix_capability(extension, profile.vlen_bits))
                                  for extension in used_extensions)
        self.facts = _cpu_facts()
        self.validated_cpus: set[int] = set()
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.prctl = self.libc.prctl
        self.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
        self.prctl.restype = ctypes.c_int
        self.vlen_probe = None

    def bind_library(self, library) -> None:
        self.vlen_probe = library.intent_weft_vlen_bits
        self.vlen_probe.argtypes = []
        self.vlen_probe.restype = ctypes.c_int64

    def check(self) -> None:
        stack, _ = resource.getrlimit(resource.RLIMIT_STACK)
        if stack != resource.RLIM_INFINITY and self.profile.private_stack_bytes >= stack:
            raise ValueError("artifact private stack budget exceeds the native thread's stack limit")
        cpus = os.sched_getaffinity(0)
        if not cpus.issubset(self.profile.cpus):
            raise ValueError("current CPU affinity exceeds the artifact execution set")
        for cpu in cpus - self.validated_cpus:
            # ISA and vendor identity belong to the CPU. A newly online CPU may
            # be absent from the artifact's initial /proc/cpuinfo snapshot.
            if cpu not in self.facts:
                self.facts = _cpu_facts()
            facts = self.facts[cpu]
            extensions = _isa_extensions(facts["isa"])
            missing = self.required - extensions
            if missing:
                raise ValueError(f"CPU {cpu} lacks requested ISA extensions: {sorted(missing)}")
            for extension, capability in self.capabilities:
                if (capability.isa_feature not in extensions or
                        int(facts["mvendorid"], 0) != capability.vendor_id or
                        int(facts["marchid"], 0) != capability.architecture_id):
                    raise ValueError(f"CPU {cpu} does not support the artifact's {extension} instructions")
            self.validated_cpus.add(cpu)
        control = self.prctl(70, 0, 0, 0, 0)
        if control < 0:
            raise OSError(ctypes.get_errno(), "cannot query the calling thread's RVV state")
        if control & 3 == 1:
            raise ValueError("RVV is disabled for the calling thread")
        # The calling thread may have migrated, so VLEN remains a live check.
        if self.vlen_probe is not None and self.vlen_probe() != self.profile.vlen_bits:
            raise ValueError("native VLEN disagrees with the artifact")


_winners: dict[tuple, int] = {}


def _measure(measure, arguments: tuple, repetitions: int = 1) -> float:
    elapsed = measure(*arguments, repetitions)
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("native benchmark returned no valid positive duration")
    return elapsed


@dataclass
class NativeCall(ObservedCall):
    program: NativeProgram
    arguments: tuple
    native_arguments: tuple
    outputs: tuple[Buffer, ...]
    key: tuple
    description: tuple
    trial_storage: tuple[memoryview, ...]
    winner: int | None = None

    def inspect_configurations(self):
        return self.program.inspect_configurations()

    @contextmanager
    def _trial_state(self):
        snapshots = tuple((storage, bytes(storage)) for storage in self.trial_storage)

        def restore() -> None:
            for storage, snapshot in snapshots:
                storage[:] = snapshot

        try:
            yield restore
        finally:
            restore()

    def choose(self, prepare=None) -> int:
        if self.winner is not None:
            return self.winner
        observed = []
        single = len(self.program.measurements) == 1
        reused = not single and self.key in _winners
        if single:
            self.winner = 0
        elif not reused:
            timings = []
            with self._trial_state() as restore:
                for candidate, measure, configuration in zip(self.program.candidates, self.program.measurements,
                                                             self.program.configuration_descriptions, strict=True):
                    samples = []
                    try:
                        for _ in range(3):
                            restore()
                            if prepare is not None:
                                prepare()
                            samples.append(_measure(measure, self.native_arguments))
                    except Exception as error:
                        observed.append(CandidateObservation(configuration, "failed", "provider_tuning",
                                                             type(error).__name__, str(error)))
                        details = observation("weft", self.program.target, self.description, None, (), observed,
                                              stage="failed", caches=self.program.compilation_cache)
                        self._record_observation(details)
                        raise CompilationStageError("provider_tuning", str(error), candidate=candidate.entry,
                                                    observation=details) from error
                    elapsed = statistics.median(samples)
                    timings.append(elapsed)
                    observed.append(CandidateObservation(configuration, "trial_completed", "provider_tuning",
                                                         elapsed_ms=elapsed))
                _winners[self.key] = min(range(len(timings)), key=timings.__getitem__)
        if not single:
            self.winner = _winners[self.key]
        self._record_observation(observation("weft", self.program.target, self.description,
            self.program.configuration_descriptions[self.winner], (), observed, stage="selected",
            history_unavailable="Only one native candidate; tuning was not required" if single else
                                "The process-local winner was reused; no candidates were retried" if reused else None,
            caches=(*self.program.compilation_cache,
                    CacheObservation("tuning", "process", None if single else reused, "Weft runtime winner cache",
                                     "selection",
                                     "Only one native candidate" if single else None))))
        return self.winner

    def launch(self) -> None:
        try:
            self.program.check_execution()
        except Exception as error:
            details = observation("weft", self.program.target, self.description, None, (),
                                  stage="failed", caches=self.program.compilation_cache,
                                  history_unavailable="Native execution requirements failed before candidate invocation")
            self._record_observation(details)
            raise CompilationStageError("provider_invocation", str(error), observation=details) from error
        selected = self.choose()
        try:
            self.program.functions[selected](*self.native_arguments)
        except Exception as error:
            details = replace(self.observation, stage="failed")
            self._record_observation(details)
            raise CompilationStageError("provider_invocation", str(error),
                                        candidate=self.program.candidates[selected].entry, observation=details) from error
        self._record_execution("launched")

    def benchmark(self, *, prepare=None, samples: int = 10) -> float:
        self.program.check_execution()
        measure = self.program.measurements[self.choose(prepare)]
        values = []
        with self._trial_state() as restore:
            for _ in range(samples):
                restore()
                if prepare is not None:
                    prepare()
                values.append(_measure(measure, self.native_arguments))
        elapsed = statistics.median(values)
        self._record_execution("benchmarked")
        return elapsed

    def result(self):
        return invocation_result(self.outputs)


class NativeProgram(NativePreparedRuntime):
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        manifest_text = (self.directory / "artifact.json").read_text()
        manifest = json.loads(manifest_text)
        contract = ProgramContract.read((self.directory / "canonical.mlir").read_text(encoding="utf-8"),
                                        manifest["program"])
        validate_artifact(manifest, contract)
        self.profile = TargetProfile(**manifest["profile"])
        self.facts = contract.facts
        self.target = bindings({"family": "cpu", **{name: value for name, value in asdict(contract.target).items()
                                                  if name != "provider"}})
        self.compilation_cache = (CacheObservation("native_compilation", "external_aot", None, "Weft kernel.so", "materialization",
            "The native runtime loads an existing external AOT artifact and does not observe its build cache"),)
        abi = contract.abi
        self.interface = abi.interface
        self.trial_regions = abi.trial_regions()
        self.requirements = requirements = abi.requirements
        self._binders = abi.binders(
            observe_view=type(self)._view, allocate_output=type(self)._allocate_output,
            check_alias=requirements.check_pair,
            check_view_geometry=lambda owner, parameter, facts: requirements.check_geometry(parameter, facts),
            check_view_storage=lambda owner, parameter, facts: requirements.check_storage(parameter, facts),
            view_key=lambda facts, group: (facts.shape, facts.strides, facts.dtype, facts.offset, group),
            scalar_key=lambda parameter, value: (parameter.dtype.name, value),
        )
        self.candidates = self.facts.candidates
        self.configuration_descriptions = tuple(bindings(candidate.metadata()) for candidate in self.candidates)
        kernels = {kernel["symbol"]: kernel for kernel in manifest["weft"]["kernels"]}
        self.candidate_extensions = tuple(frozenset(
            extension for task in self.facts.tasks if task.cpu_entry == candidate.entry
            for extension in kernels[task.abi.symbol]["used_extensions"])
            for candidate in self.candidates)
        self.used_extensions = frozenset(extension for kernel in manifest["weft"]["kernels"]
                                         for extension in kernel["used_extensions"])
        self._execution = _ExecutionContract(self.profile, self.used_extensions)
        self._execution.check()
        self.library = ctypes.CDLL(str(self.directory / "kernel.so"))
        self._execution.bind_library(self.library)
        self.check_execution()
        self.identity = (manifest_text, (self.directory / "kernel.so").stat().st_mtime_ns)
        types = abi.argument_types()
        self.functions, self.measurements = [], []
        for candidate in self.candidates:
            function = getattr(self.library, candidate.entry + "_invoke")
            function.argtypes, function.restype = types, None
            measure = getattr(self.library, candidate.entry + "_benchmark")
            measure.argtypes, measure.restype = [*types, ctypes.c_int64], ctypes.c_double
            self.functions.append(function)
            self.measurements.append(measure)

    def check_execution(self) -> None:
        if self.library is None:
            raise RuntimeError("native artifact is closed")
        self._execution.check()

    def _view(self, parameter: ViewParameter, value) -> ViewFacts:
        if not isinstance(value, Buffer) or value.dtype != parameter.dtype.name:
            raise TypeError(f"{parameter.name} requires a native {parameter.dtype} Buffer")
        if len(value.shape) != len(parameter.shape):
            raise ValueError("native view rank disagrees with the compiler ABI")
        return ViewFacts(value.shape, value.strides, value.pointer, value.allocation, value.allocation_end,
                         value.pointer - value.allocation, value.dtype,
                         value.pointer, value.pointer + value.nbytes, value.allocation)

    def _allocate_output(self, parameter: ViewParameter, shape: tuple[int, ...]) -> tuple[Buffer, ViewFacts]:
        value = Buffer.empty(shape, parameter.dtype.name)
        facts = ViewFacts(value.shape, value.strides, value.pointer, value.allocation, value.allocation_end,
                          value.pointer - value.allocation, value.dtype,
                          value.pointer, value.pointer + value.nbytes, value.allocation)
        return value, facts

    def prepare(self, arguments: tuple, *, explicit_outputs: bool = False) -> NativeCall:
        bound = self._binders[bool(explicit_outputs)](self, arguments)
        trial_storage = []
        for region in self.trial_regions:
            position = region.pointer.parameter.position
            storage = bound.arguments[position].storage
            if storage.nbytes != region.byte_count(bound.views[position].shape):
                raise ValueError("native trial byte range disagrees with the bound buffer storage")
            trial_storage.append(storage)
        return NativeCall(self, bound.arguments, bound.native_arguments, bound.outputs,
                          (self.identity, bound.key), self.describe_arguments(bound, "cpu"),
                          tuple(trial_storage))

    def close(self) -> None:
        import _ctypes
        if self.library is not None:
            self.functions.clear()
            self.measurements.clear()
            self._execution.vlen_probe = None
            _ctypes.dlclose(self.library._handle)
            self.library = None
