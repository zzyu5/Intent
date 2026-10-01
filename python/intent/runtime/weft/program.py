from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys

from .buffer import Buffer
from .compilation import validate_artifact
from .target import TargetProfile, matrix_capability
from ..cpu import check_alias
from ..native import NativeInterface, NativePreparedRuntime, ViewFacts, ViewParameter


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


@dataclass
class NativeCall:
    program: NativeProgram
    arguments: tuple
    native_arguments: tuple
    outputs: tuple[Buffer, ...]
    key: tuple
    trial_inputs: tuple[tuple[Buffer, bytes], ...] = ()
    winner: int | None = None

    def restore_trial_inputs(self) -> None:
        for buffer, snapshot in self.trial_inputs:
            buffer.storage[:] = snapshot

    def choose(self, prepare=None) -> int:
        if self.winner is not None:
            return self.winner
        if len(self.program.measurements) == 1:
            self.winner = 0
            return self.winner
        if self.key not in _winners:
            timings = []
            try:
                for measure in self.program.measurements:
                    samples = []
                    for _ in range(3):
                        self.restore_trial_inputs()
                        if prepare is not None:
                            prepare()
                        samples.append(measure(*self.native_arguments, 1))
                    timings.append(statistics.median(samples))
                _winners[self.key] = min(range(len(timings)), key=timings.__getitem__)
            finally:
                self.restore_trial_inputs()
        self.winner = _winners[self.key]
        return self.winner

    def launch(self) -> None:
        self.program.check_execution()
        self.program.functions[self.choose()](*self.native_arguments)

    def benchmark(self, *, prepare=None, samples: int = 10) -> float:
        self.program.check_execution()
        measure = self.program.measurements[self.choose(prepare)]
        values = []
        for _ in range(samples):
            self.restore_trial_inputs()
            if prepare is not None:
                prepare()
            values.append(measure(*self.native_arguments, 1))
        return statistics.median(values)

    def result(self):
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs


class NativeProgram(NativePreparedRuntime):
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        manifest_text = (self.directory / "artifact.json").read_text()
        manifest = json.loads(manifest_text)
        validate_artifact(manifest)
        self.profile = TargetProfile(**manifest["profile"])
        self.metadata = manifest["program"]
        self.parameters = self.metadata["parameters"]
        self.interface = NativeInterface.read(self.parameters)
        self._alignments = tuple(self.parameters[parameter.position]["alignment"]
                                 if isinstance(parameter, ViewParameter) else None
                                 for parameter in self.interface.parameters)
        self._binders = self.interface.binders(
            observe_view=type(self)._view, allocate_output=type(self)._allocate_output,
            check_alias=check_alias,
            view_key=lambda facts, group: (facts.shape, facts.strides, facts.dtype, facts.offset, group),
            scalar_key=lambda parameter, value: (parameter.dtype, value),
        )
        self.candidates = self.metadata["candidates"]
        kernels = {kernel["symbol"]: kernel for kernel in manifest["weft"]["kernels"]}
        self.candidate_extensions = tuple(frozenset(
            extension for task in self.metadata["tasks"] if task["cpu_entry"] == candidate["entry"]
            for extension in kernels[task["abi"]["symbol"]]["used_extensions"])
            for candidate in self.candidates)
        self.used_extensions = frozenset(extension for kernel in manifest["weft"]["kernels"]
                                         for extension in kernel["used_extensions"])
        self._execution = _ExecutionContract(self.profile, self.used_extensions)
        self._execution.check()
        self.library = ctypes.CDLL(str(self.directory / "kernel.so"))
        self._execution.bind_library(self.library)
        self.check_execution()
        self.identity = (manifest_text, (self.directory / "kernel.so").stat().st_mtime_ns)
        types = self.interface.argument_types(lambda dtype: ctypes.c_float if dtype == "f32" else ctypes.c_int64)
        self.functions, self.measurements = [], []
        for candidate in self.candidates:
            function = getattr(self.library, candidate["entry"] + "_invoke")
            function.argtypes, function.restype = types, None
            measure = getattr(self.library, candidate["entry"] + "_benchmark")
            measure.argtypes, measure.restype = [*types, ctypes.c_int64], ctypes.c_double
            self.functions.append(function)
            self.measurements.append(measure)

    def check_execution(self) -> None:
        if self.library is None:
            raise RuntimeError("native artifact is closed")
        self._execution.check()

    def _view(self, parameter: ViewParameter, value) -> ViewFacts:
        if not isinstance(value, Buffer) or value.dtype != parameter.dtype:
            raise TypeError(f"{parameter.name} requires a native {parameter.dtype} Buffer")
        if len(value.shape) != len(parameter.shape):
            raise ValueError("native view rank disagrees with the compiler ABI")
        if value.pointer % self._alignments[parameter.position]:
            raise ValueError("native view does not meet the compiler alignment requirement")
        return ViewFacts(value.shape, value.strides, value.pointer, value.allocation,
                         value.pointer - value.allocation, value.dtype,
                         value.pointer, value.pointer + value.nbytes)

    def _allocate_output(self, parameter: ViewParameter, shape: tuple[int, ...]) -> tuple[Buffer, ViewFacts]:
        value = Buffer.empty(shape, parameter.dtype)
        if value.pointer % self._alignments[parameter.position]:
            raise ValueError("native view does not meet the compiler alignment requirement")
        facts = ViewFacts(value.shape, value.strides, value.pointer, value.allocation,
                          value.pointer - value.allocation, value.dtype,
                          value.pointer, value.pointer + value.nbytes)
        return value, facts

    def prepare(self, arguments: tuple, *, explicit_outputs: bool = False) -> NativeCall:
        bound = self._binders[bool(explicit_outputs)](self, arguments)
        snapshots = tuple((bound.arguments[parameter.position],
                           bytes(bound.arguments[parameter.position].storage))
                          for parameter in self.interface.mutable_inputs)
        return NativeCall(self, bound.arguments, bound.native_arguments, bound.outputs,
                          (self.identity, bound.key), snapshots)

    def run(self, *arguments):
        call = self.prepare(arguments)
        call.launch()
        return call.result()

    def launch(self, *arguments) -> None:
        self.prepare(arguments, explicit_outputs=True).launch()

    def close(self) -> None:
        import _ctypes
        if self.library is not None:
            self.functions.clear()
            self.measurements.clear()
            self._execution.vlen_probe = None
            _ctypes.dlclose(self.library._handle)
            self.library = None
