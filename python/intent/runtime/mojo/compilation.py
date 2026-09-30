from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import ctypes
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from threading import Lock
import weakref

from intent.compiler.cache import cache_root


@dataclass
class NativeLibrary:
    directory: Path
    library: ctypes.CDLL
    _cleanup: weakref.finalize = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cleanup = weakref.finalize(self, shutil.rmtree, self.directory)

    def close(self) -> None:
        self._cleanup()


@dataclass
class NativeCompilation:
    libraries: tuple[NativeLibrary, ...]
    identity: tuple[object, ...]


_compilations: dict[tuple[object, ...], Future[NativeCompilation]] = {}
_compilation_lock = Lock()
_compilers = ThreadPoolExecutor(max_workers=2)

ELEMENT_TYPES = {
    "f16": "Float16", "bf16": "BFloat16", "f32": "Float32", "f64": "Float64",
    "i1": "SIMD[DType.bool, 1]", "i8": "Int8", "i16": "Int16", "i32": "Int32", "i64": "Int64",
    "ui8": "UInt8", "ui16": "UInt16", "ui32": "UInt32", "ui64": "UInt64",
    "f8e4m3fn": "SIMD[DType.float8_e4m3fn, 1]", "f8e5m2": "SIMD[DType.float8_e5m2, 1]",
}

SCALAR_CTYPES = {
    "f32": ctypes.c_float, "f64": ctypes.c_double, "i1": ctypes.c_bool,
    "i8": ctypes.c_int8, "i16": ctypes.c_int16, "i32": ctypes.c_int32, "i64": ctypes.c_int64,
    "ui8": ctypes.c_uint8, "ui16": ctypes.c_uint16, "ui32": ctypes.c_uint32, "ui64": ctypes.c_uint64,
}


def flattened_signature(parameters: list[dict[str, object]]) -> tuple[list[str], list[str]]:
    signature: list[str] = []
    arguments: list[str] = []
    for index, parameter in enumerate(parameters):
        name = f"a{index}"
        if parameter["kind"] == "view":
            signature.append(f"{name}: Pointer[{ELEMENT_TYPES[parameter['dtype']]}, MutUntrackedOrigin]")
            arguments.append(f"{name}.unsafe_bitcast[Int{parameter['dtype'][2:]}]()"
                             if parameter["dtype"].startswith("ui") else name)
            rank = len(parameter["shape"])
            for role in ("d", "s"):
                for axis in range(rank):
                    field = f"{name}_{role}{axis}"
                    signature.append(f"{field}: Int64")
                    arguments.append(field)
        else:
            signature.append(f"{name}: {'Bool' if parameter['dtype'] == 'i1' else ELEMENT_TYPES[parameter['dtype']]}")
            arguments.append(f"bitcast[DType.int{parameter['dtype'][2:]}]({name})"
                             if parameter["dtype"].startswith("ui") else name)
    return signature, arguments


def benchmark_exports(metadata: dict[str, object]) -> str:
    signature, arguments = flattened_signature(metadata["parameters"])
    sections = ["\nfrom std.time import monotonic\nfrom std.sys import size_of\n"]
    mutable = [(index, parameter) for index, parameter in enumerate(metadata["parameters"])
               if parameter["kind"] == "view" and parameter["access"] == 2]
    for candidate in metadata["candidates"]:
        entry = candidate["entry"]
        sections.append(
            f'\n@export("{entry}_benchmark")\n'
            f"def {entry}_benchmark({', '.join(signature)}, repetitions: Int64) abi(\"C\") -> Float64:\n"
        )
        if not mutable:
            sections.append(
                "    var begin = monotonic()\n"
                "    for iteration in range(Int(repetitions)):\n"
                f"        {entry}({', '.join(arguments)})\n"
                "    return Float64(monotonic() - begin) * 1.0e-6 / Float64(repetitions)\n"
            )
            continue
        for index, parameter in mutable:
            dimensions = " * ".join(f"Int(a{index}_d{axis})" for axis in range(len(parameter["shape"]))) or "1"
            element = ELEMENT_TYPES[parameter["dtype"]]
            sections.append(
                f"    var bytes_a{index} = ({dimensions}) * size_of[{element}]()\n"
                f"    var saved_a{index} = alloc(Layout[UInt8](count=bytes_a{index}))\n"
                f'    external_call["memcpy", NoneType](saved_a{index}.unsafe_ptr(), a{index}, UInt(bytes_a{index}))\n'
            )
        sections.append("    var elapsed = Float64(0)\n    for iteration in range(Int(repetitions)):\n")
        for index, _ in mutable:
            sections.append(f'        external_call["memcpy", NoneType](a{index}, saved_a{index}.unsafe_ptr(), UInt(bytes_a{index}))\n')
        sections.append(
            "        var begin = monotonic()\n"
            f"        {entry}({', '.join(arguments)})\n"
            "        elapsed += Float64(monotonic() - begin)\n"
        )
        for index, _ in mutable:
            sections.append(
                f'    external_call["memcpy", NoneType](a{index}, saved_a{index}.unsafe_ptr(), UInt(bytes_a{index}))\n'
                f"    dealloc(saved_a{index}^)\n"
            )
        sections.append("    return elapsed * 1.0e-6 / Float64(repetitions)\n")
    return "".join(sections)


def _invoke_compiler(command: list[str], directory: Path, stage: str) -> None:
    (directory / f"{stage}.command").write_text(shlex.join(command) + "\n", encoding="utf-8")
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        (directory / f"{stage}.stdout").write_text("", encoding="utf-8")
        (directory / f"{stage}.stderr").write_text(str(error) + "\n", encoding="utf-8")
        (directory / f"{stage}.status").write_text("process not started\n", encoding="utf-8")
        raise
    (directory / f"{stage}.stdout").write_text(completed.stdout, encoding="utf-8")
    (directory / f"{stage}.stderr").write_text(completed.stderr, encoding="utf-8")
    (directory / f"{stage}.status").write_text(str(completed.returncode) + "\n", encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"compiler exited with code {completed.returncode}:\n{completed.stderr}{completed.stdout}"
        )


def _compile_unit(source: str, metadata: dict[str, object], target) -> NativeLibrary:
    candidate, = metadata["candidates"]
    entry = candidate["entry"]
    root = cache_root() / "mojo"
    stage = "artifact_setup"
    try:
        root.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="native-", dir=root))
        (root / "artifact.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        stage = "source_materialization"
        source_path = root / "kernel.mojo"
        source_path.write_text(source, encoding="utf-8")
        with source_path.open("a", encoding="utf-8") as output:
            output.write(benchmark_exports(metadata))
        fp_source = root / "fp_environment.c"
        fp_source.write_text(Path(__file__).with_name("fp_environment.c").read_text(encoding="utf-8"),
                             encoding="utf-8")
        fp_object = root / "fp_environment.o"
        library_path = root / "kernel.so"
        stage = "fp_environment_compilation"
        _invoke_compiler(
            ["cc", "-O2", "-fPIC", "-c", str(fp_source), "-o", str(fp_object)], root, stage,
        )
        stage = "native_compilation"
        _invoke_compiler(
            [target.executable, "build", str(source_path), "--emit", "shared-lib", "-o", str(library_path),
             *target.native_options, "-Xlinker", str(fp_object)], root, stage,
        )
        stage = "native_loading"
        library = ctypes.CDLL(str(library_path))
    except Exception as error:
        raise RuntimeError(
            f"Mojo {stage} failed for candidate {entry}:\n{error}\nNative compiler artifacts: {root}"
        ) from error
    return NativeLibrary(root, library)


def compile_library(source: str, metadata: dict[str, object], target) -> NativeCompilation:
    fp_source = Path(__file__).with_name("fp_environment.c")
    key = (source, json.dumps(metadata, sort_keys=True), target.executable,
           target.native_options, fp_source.read_text(encoding="utf-8"))
    with _compilation_lock:
        pending = _compilations.get(key)
        owner = pending is None
        if owner:
            pending = Future()
            _compilations[key] = pending
    if not owner:
        return pending.result()

    futures = []
    try:
        encoded = source.encode("utf-8")
        prelude = encoded[:metadata["source_prelude_end"]]
        for candidate in metadata["candidates"]:
            begin, end = candidate["source_range"]
            unit = (prelude + encoded[begin:end]).decode("utf-8")
            unit_metadata = {**metadata, "candidates": [candidate]}
            futures.append(_compilers.submit(_compile_unit, unit, unit_metadata, target))
        result = NativeCompilation(tuple(future.result() for future in futures), key)
    except BaseException as error:
        for future in futures:
            future.cancel()
        wait(futures)
        for future in futures:
            if not future.cancelled() and future.exception() is None:
                future.result().close()
        with _compilation_lock:
            del _compilations[key]
        pending.set_exception(error)
        raise
    pending.set_result(result)
    return result
