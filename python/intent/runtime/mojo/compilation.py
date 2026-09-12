from __future__ import annotations

from dataclasses import dataclass
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile


@dataclass
class NativeLibrary:
    directory: tempfile.TemporaryDirectory
    library: ctypes.CDLL
    identity: tuple[object, ...]


_libraries: dict[tuple[object, ...], NativeLibrary] = {}

ELEMENT_TYPES = {
    "f16": "Float16", "bf16": "BFloat16", "f32": "Float32", "f64": "Float64",
    "i1": "SIMD[DType.bool, 1]", "i8": "Int8", "i16": "Int16", "i32": "Int32", "i64": "Int64",
}

SCALAR_CTYPES = {
    "f32": ctypes.c_float, "f64": ctypes.c_double, "i1": ctypes.c_bool,
    "i8": ctypes.c_int8, "i16": ctypes.c_int16, "i32": ctypes.c_int32, "i64": ctypes.c_int64,
}


def flattened_signature(parameters: list[dict[str, object]]) -> tuple[list[str], list[str]]:
    signature: list[str] = []
    arguments: list[str] = []
    for index, parameter in enumerate(parameters):
        name = f"a{index}"
        if parameter["kind"] == "view":
            signature.append(f"{name}: Pointer[{ELEMENT_TYPES[parameter['dtype']]}, MutUntrackedOrigin]")
            arguments.append(name)
            rank = len(parameter["shape"])
            for role in ("d", "s"):
                for axis in range(rank):
                    field = f"{name}_{role}{axis}"
                    signature.append(f"{field}: Int64")
                    arguments.append(field)
        else:
            signature.append(f"{name}: {'Bool' if parameter['dtype'] == 'i1' else ELEMENT_TYPES[parameter['dtype']]}")
            arguments.append(name)
    return signature, arguments


def benchmark_exports(metadata: dict[str, object]) -> str:
    signature, arguments = flattened_signature(metadata["parameters"])
    sections = ["\nfrom std.time import monotonic\n"]
    for candidate in metadata["candidates"]:
        entry = candidate["entry"]
        sections.append(
            f'\n@export("{entry}_benchmark")\n'
            f"def {entry}_benchmark({', '.join(signature)}, repetitions: Int64) abi(\"C\") -> Float64:\n"
            "    var begin = monotonic()\n"
            "    for iteration in range(Int(repetitions)):\n"
            f"        {entry}({', '.join(arguments)})\n"
            "    return Float64(monotonic() - begin) * 1.0e-6 / Float64(repetitions)\n"
        )
    return "".join(sections)


def compile_library(source: str, metadata: dict[str, object], target) -> NativeLibrary:
    complete_source = source + benchmark_exports(metadata)
    fp_source = Path(__file__).with_name("fp_environment.c")
    key = (complete_source, json.dumps(metadata, sort_keys=True), target.executable,
           target.native_options, fp_source.read_text(encoding="utf-8"))
    if key in _libraries:
        return _libraries[key]
    directory = tempfile.TemporaryDirectory(prefix="intentdsl-mojo-artifact-")
    root = Path(directory.name)
    source_path = root / "kernel.mojo"
    library_path = root / "kernel.so"
    source_path.write_text(complete_source, encoding="utf-8")
    fp_object = root / "fp_environment.o"
    subprocess.run(
        ["cc", "-O2", "-fPIC", "-c", str(fp_source), "-o", str(fp_object)],
        capture_output=True, text=True, check=True,
    )
    completed = subprocess.run(
        [target.executable, "build", str(source_path), "--emit", "shared-lib", "-o", str(library_path),
         *target.native_options, "-Xlinker", str(fp_object)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Mojo native compilation failed:\n{completed.stderr}{completed.stdout}")
    result = NativeLibrary(directory, ctypes.CDLL(str(library_path)), key)
    _libraries[key] = result
    return result
