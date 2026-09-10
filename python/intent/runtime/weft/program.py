from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import statistics
import sys

from .buffer import Buffer
from .compilation import TargetProfile, validate_artifact


def _isa_extensions(isa: str) -> set[str]:
    base, *extensions = isa.split("_")
    if not base.startswith("rv64"):
        raise ValueError("native CPU does not report an RV64 ISA")
    result = set(base[4:]) | set(extensions)
    if "g" in result:
        result.remove("g")
        result |= {"i", "m", "a", "f", "d", "zicsr", "zifencei"}
    return result


def _check_execution(profile: TargetProfile, library=None) -> None:
    if sys.platform != "linux" or platform.machine() != "riscv64" or sys.byteorder != "little":
        raise NotImplementedError("Weft native loading requires little-endian RISC-V Linux")
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise ValueError("native pointer size disagrees with RV64/lp64d")
    cpus = os.sched_getaffinity(0)
    if not cpus.issubset(profile.cpus):
        raise ValueError("current CPU affinity exceeds the artifact execution set")
    required = _isa_extensions(profile.march)
    facts = {}
    for paragraph in Path("/proc/cpuinfo").read_text().split("\n\n"):
        fields = dict(line.split(":", 1) for line in paragraph.splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if "processor" in fields:
            facts[int(fields["processor"])] = _isa_extensions(fields["isa"])
    for cpu in cpus:
        missing = required - facts[cpu]
        if missing:
            raise ValueError(f"CPU {cpu} lacks requested ISA extensions: {sorted(missing)}")
    libc = ctypes.CDLL(None, use_errno=True)
    control = libc.prctl(ctypes.c_int(70), *(ctypes.c_ulong(0) for _ in range(4)))
    if control < 0:
        raise OSError(ctypes.get_errno(), "cannot query the calling thread's RVV state")
    if control & 3 == 1:
        raise ValueError("RVV is disabled for the calling thread")
    if library is not None:
        probe = library.intent_weft_vlen_bits
        probe.restype = ctypes.c_int64
        if probe() != profile.vlen_bits:
            raise ValueError("native VLEN disagrees with the artifact")


_winners: dict[tuple, int] = {}


@dataclass
class NativeCall:
    program: NativeProgram
    arguments: tuple
    native_arguments: tuple
    outputs: tuple[Buffer, ...]
    key: tuple
    winner: int | None = None

    def choose(self, prepare=None) -> int:
        if self.winner is not None:
            return self.winner
        if self.key not in _winners:
            timings = []
            for measure in self.program.measurements:
                samples = []
                for _ in range(3):
                    if prepare is not None:
                        prepare()
                    samples.append(measure(*self.native_arguments, 1))
                timings.append(statistics.median(samples))
            _winners[self.key] = min(range(len(timings)), key=timings.__getitem__)
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
            if prepare is not None:
                prepare()
            values.append(measure(*self.native_arguments, 1))
        return statistics.median(values)

    def result(self):
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs


class NativeProgram:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        manifest_text = (self.directory / "artifact.json").read_text()
        manifest = json.loads(manifest_text)
        validate_artifact(manifest)
        self.profile = TargetProfile(**manifest["profile"])
        self.metadata = manifest["program"]
        self.parameters = self.metadata["parameters"]
        self.candidates = self.metadata["candidates"]
        _check_execution(self.profile)
        self.library = ctypes.CDLL(str(self.directory / "kernel.so"))
        self.check_execution()
        self.identity = (manifest_text, (self.directory / "kernel.so").stat().st_mtime_ns)
        types = []
        for parameter in self.parameters:
            if parameter["kind"] == "view":
                types.extend([ctypes.c_void_p, *([ctypes.c_int64] * (2 * len(parameter["shape"])))])
            else:
                types.append(ctypes.c_float if parameter["dtype"] == "f32" else ctypes.c_int64)
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
        _check_execution(self.profile, self.library)

    def _view(self, parameter, value, dimensions: dict[int, int]) -> None:
        if not isinstance(value, Buffer) or value.dtype != parameter["dtype"]:
            raise TypeError(f"{parameter['name']} requires a native {parameter['dtype']} Buffer")
        if len(value.shape) != len(parameter["shape"]):
            raise ValueError("native view rank disagrees with the compiler ABI")
        if value.pointer % parameter["alignment"]:
            raise ValueError("native view does not meet the compiler alignment requirement")
        for extent, static, identity in zip(value.shape, parameter["shape"], parameter["dimensions"]):
            if static >= 0 and static != extent:
                raise ValueError("native view has an incompatible static extent")
            if identity > 0:
                if identity in dimensions and dimensions[identity] != extent:
                    raise ValueError("native views disagree on a canonical dimension identity")
                dimensions[identity] = extent

    def prepare(self, arguments: tuple, *, explicit_outputs: bool = False) -> NativeCall:
        expected = len(self.parameters) if explicit_outputs else sum(
            p["kind"] != "view" or p["access"] != 1 for p in self.parameters)
        if len(arguments) != expected:
            raise TypeError(f"expected {expected} native arguments, got {len(arguments)}")
        supplied, dimensions, values = iter(arguments), {}, []
        for parameter in self.parameters:
            if not explicit_outputs and parameter["kind"] == "view" and parameter["access"] == 1:
                values.append(None)
                continue
            value = next(supplied)
            if parameter["kind"] == "view":
                self._view(parameter, value, dimensions)
            values.append(value)
        outputs = []
        for index, parameter in enumerate(self.parameters):
            if parameter["kind"] != "view" or parameter["access"] != 1:
                continue
            if not explicit_outputs:
                shape = tuple(static if static >= 0 else dimensions[identity]
                              for static, identity in zip(parameter["shape"], parameter["dimensions"]))
                values[index] = Buffer.empty(shape, parameter["dtype"])
                self._view(parameter, values[index], dimensions)
            outputs.append(values[index])
        views = [(p, v) for p, v in zip(self.parameters, values) if p["kind"] == "view"]
        for i, (parameter, value) in enumerate(views):
            for other_parameter, other in views[i + 1:]:
                overlap = value.pointer < other.pointer + other.nbytes and other.pointer < value.pointer + value.nbytes
                same_allocation = value.allocation == other.allocation
                if overlap and (parameter["access"] != 0 or other_parameter["access"] != 0):
                    raise NotImplementedError("overlapping writable native views are not implemented")
                if same_allocation and (parameter["noalias"] or other_parameter["noalias"]):
                    raise ValueError("native invocation violates an author noalias constraint")
                if parameter["alias"] and parameter["alias"] == other_parameter["alias"] and not same_allocation:
                    raise ValueError("native invocation violates an author allocation-alias constraint")
        native, facts, groups = [], [], {}
        for parameter, value in zip(self.parameters, values):
            if parameter["kind"] == "view":
                native.extend((value.pointer, *value.shape, *value.strides))
                group = groups.setdefault(value.allocation, len(groups))
                facts.append((value.shape, value.strides, value.dtype, value.pointer - value.allocation, group))
            else:
                native.append(value)
                facts.append((parameter["dtype"], value))
        return NativeCall(self, tuple(values), tuple(native), tuple(outputs), (self.identity, tuple(facts)))

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
            _ctypes.dlclose(self.library._handle)
            self.library = None
