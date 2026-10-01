from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import base64
import ctypes
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from threading import Lock
import time
from uuid import uuid4

from intent.compiler.cache import cache_root, file_identity, locked_cache_entry
from .toolchain import resolve_toolchain, runtime_dependencies


@dataclass
class NativeLibrary:
    directory: Path
    library: ctypes.CDLL
    cache_hit: bool
    cache_reason: str | None


@dataclass
class NativeCompilation:
    libraries: tuple[NativeLibrary, ...]
    identity: tuple[object, ...]


_compilations: dict[tuple[object, ...], Future[NativeCompilation]] = {}
_compilation_lock = Lock()
_compilers = ThreadPoolExecutor(max_workers=2)
_BUILD_OPTIONS = ("--emit", "shared-lib")
_loaded_libraries: dict[tuple[object, ...], ctypes.CDLL] = {}
_loading_lock = Lock()

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


def _invoke_compiler(command: list[str], directory: Path, stage: str,
                     environment: dict[str, str]) -> None:
    (directory / f"{stage}.command").write_text(shlex.join(command) + "\n", encoding="utf-8")
    started = time.monotonic()
    try:
        completed = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    except OSError as error:
        (directory / f"{stage}.stdout").write_text("", encoding="utf-8")
        (directory / f"{stage}.stderr").write_text(str(error) + "\n", encoding="utf-8")
        (directory / f"{stage}.status").write_text("process not started\n", encoding="utf-8")
        raise
    finally:
        (directory / f"{stage}.seconds").write_text(str(time.monotonic() - started) + "\n", encoding="utf-8")
    (directory / f"{stage}.stdout").write_text(completed.stdout, encoding="utf-8")
    (directory / f"{stage}.stderr").write_text(completed.stderr, encoding="utf-8")
    (directory / f"{stage}.status").write_text(str(completed.returncode) + "\n", encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"compiler exited with code {completed.returncode}:\n{completed.stderr}{completed.stdout}"
        )


def _write_source(path: Path, text: str) -> None:
    # Stable inputs retain their mtimes for the provider's own compilation cache.
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def _compile_fp_environment(environment: dict[str, str]) -> tuple[Path, bytes]:
    executable = shutil.which("cc", path=environment.get("PATH"))
    if executable is None:
        raise FileNotFoundError("Mojo floating-point environment compilation requires a C compiler (cc)")
    compiler = file_identity(Path(executable))
    source = Path(__file__).with_name("fp_environment.c").read_text(encoding="utf-8")
    options = ("-O2", "-fPIC", "-c")
    key = json.dumps((compiler, options, source))
    with locked_cache_entry("mojo-fp", key) as cache:
        source_path = cache.directory / "fp_environment.c"
        _write_source(source_path, source)
        attempt = cache.create_attempt()
        output = attempt / "fp_environment.o"
        # Resolve headers and the C toolchain on each materialization. The actual
        # object is a native link input, shared by all candidates in this group.
        try:
            _invoke_compiler([executable, *options, str(source_path), "-o", str(output)],
                             attempt, "fp_environment_compilation", environment)
            if file_identity(Path(executable)) != compiler:
                raise RuntimeError("C compiler changed during compilation")
            payload = output.read_bytes()
            if not payload:
                raise RuntimeError("C compiler produced an empty FP environment object")
        except Exception as error:
            raise RuntimeError(f"Mojo FP environment compilation failed: {error}\n"
                               f"Native compiler artifacts: {attempt}") from error
        cache.publish(attempt)
        return output, payload


def _link_inputs(path: Path) -> list[Path]:
    # LLD emits Make escaping, not shell quoting. Only the first rule describes
    # link inputs; any following phony rules are not additional dependencies.
    rule = path.read_text(encoding="utf-8").replace("\\\n", "").splitlines()[0]
    escaped = False
    for index, character in enumerate(rule):
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ":":
            rule = rule[index + 1:]
            break
    else:
        raise ValueError(f"linker dependency file has no target: {path}")
    words, word = [], []
    index = 0
    while index < len(rule):
        character = rule[index]
        if character == "\\":
            index += 1
            if index == len(rule):
                raise ValueError(f"incomplete linker dependency escape: {path}")
            word.append(rule[index])
        elif character == "$" and rule[index:index + 2] == "$$":
            word.append("$")
            index += 1
        elif character.isspace():
            if word:
                words.append(Path("".join(word)).absolute())
                word = []
        else:
            word.append(character)
        index += 1
    if word:
        words.append(Path("".join(word)).absolute())
    return words


def _dependency(path: Path):
    # Keep the lookup path as well as the resolved file identity: retargeting a
    # symlink must invalidate a dependency even if its previous target remains.
    return str(path.absolute()), file_identity(path)


def _unchanged(dependency) -> bool:
    try:
        return json.dumps(_dependency(Path(dependency[0]))) == json.dumps(dependency)
    except FileNotFoundError:
        return False


def _load_library(path: Path, entry: str):
    identity = file_identity(path)
    # Dependency checks precede this lookup. Keep one process-lifetime handle
    # per immutable artifact, not one dlopen reference per materialization.
    with _loading_lock:
        library = _loaded_libraries.get(identity)
        owner = library is None
        if owner:
            library = ctypes.CDLL(str(path))
        try:
            getattr(library, entry)
            getattr(library, entry + "_benchmark")
        except BaseException:
            if owner:
                import _ctypes
                _ctypes.dlclose(library._handle)
            raise
        if owner:
            _loaded_libraries[identity] = library
        return library


def _compile_unit(source: str, metadata: dict[str, object], key: str, target,
                  environment: dict[str, str], snapshot, fp_object: Path) -> NativeLibrary:
    candidate, = metadata["candidates"]
    entry = candidate["entry"]
    root = cache_root() / "mojo"
    stage = "artifact_lookup"
    try:
        with locked_cache_entry("mojo", key) as cache:
            root = cache.directory
            previous = cache.ready_attempt() if snapshot.identity is not None else None
            if previous is not None:
                manifest = json.loads((previous / "native.json").read_text(encoding="utf-8"))
                library_path = previous / "kernel.so"
                valid = all(_unchanged(dependency) for dependency in
                            [manifest["library"], *manifest["link_inputs"], *manifest["runtime_inputs"]])
                if valid:
                    current = [_dependency(path) for path in runtime_dependencies(library_path, environment)]
                    valid = json.dumps(current) == json.dumps(manifest["runtime_inputs"])
                if valid:
                    stage = "cached_native_loading"
                    library = _load_library(previous / "kernel.so", entry)
                    return NativeLibrary(previous, library, True, None)
                cache.invalidate_ready()
            source_path = cache.directory / "kernel.mojo"
            _write_source(source_path, source)
            root = cache.create_attempt()
            (root / "artifact.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            (root / "request.json").write_text(json.dumps({
                "source": str(source_path), "fp_object": str(fp_object),
                "cache_unavailable_reason": snapshot.reason,
            }, indent=2) + "\n", encoding="utf-8")
            library_path = root / "kernel.so"
            dependencies = root / "link.d"
            link_options = ["-Xlinker", str(fp_object)]
            build_environment = environment
            if snapshot.identity is not None:
                link_options += ["-Xlinker", "--dependency-file=" + str(dependencies)]
                # Compiler-created link objects belong to this attempt, not to
                # the external dependency set. Keep their scratch namespace
                # explicit even when the provider removes them before returning.
                temporary = root / "temporary"
                temporary.mkdir()
                build_environment = {**environment, "TMPDIR": str(temporary)}
            stage = "native_compilation"
            _invoke_compiler(
                [target.executable, "build", str(source_path), *_BUILD_OPTIONS, "-o", str(library_path),
                 *target.native_options, *link_options], root, stage, build_environment,
            )
            snapshot.check_unchanged()
            stage = "native_loading"
            library = _load_library(library_path, entry)
            if snapshot.identity is not None:
                inputs = [_dependency(path) for path in _link_inputs(dependencies)
                          if path.resolve() != fp_object.resolve() and
                          not path.resolve().is_relative_to(cache.directory)]
                (root / "native.json").write_text(json.dumps({
                    "library": _dependency(library_path), "link_inputs": inputs,
                    "runtime_inputs": [_dependency(path) for path in runtime_dependencies(library_path, environment)],
                }, indent=2) + "\n", encoding="utf-8")
                cache.publish(root)
            return NativeLibrary(root, library, False, snapshot.reason)
    except Exception as error:
        raise RuntimeError(
            f"Mojo {stage} failed for candidate {entry}:\n{error}\nNative compiler artifacts: {root}"
        ) from error


def compile_library(source: str, metadata: dict[str, object], target) -> NativeCompilation:
    environment = dict(os.environ)
    snapshot = resolve_toolchain(target.executable, tuple(metadata.get("native_dependencies", ())), environment)
    fp_object, fp_bytes = _compile_fp_environment(environment)
    encoded = source.encode("utf-8")
    prelude = encoded[:metadata["source_prelude_end"]]
    units = []
    for candidate in metadata["candidates"]:
        begin, end = candidate["source_range"]
        body = prelude + encoded[begin:end]
        binding = {**candidate, "source_range": [len(prelude), len(body)]}
        unit_metadata = {**metadata, "candidates": [binding]}
        complete_source = body.decode("utf-8") + benchmark_exports(unit_metadata)
        unit_key = json.dumps((complete_source, unit_metadata, target.executable,
                              _BUILD_OPTIONS, target.native_options, snapshot.identity,
                              base64.b64encode(fp_bytes).decode("ascii")), sort_keys=True)
        units.append((complete_source, unit_metadata, unit_key))
    # An unclosed toolchain is always delegated to Mojo, including repeated
    # requests in this process. It must never acquire a weak native cache key.
    key = (str(cache_root()), tuple(unit[2] for unit in units),
           None if snapshot.identity is not None else uuid4().hex)
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
        for unit, unit_metadata, unit_key in units:
            futures.append(_compilers.submit(_compile_unit, unit, unit_metadata, unit_key,
                                             target, environment, snapshot, fp_object))
        libraries = tuple(future.result() for future in futures)
        result = NativeCompilation(libraries, (key, tuple(
            file_identity(library.directory / "kernel.so") for library in libraries)))
        snapshot.check_unchanged()
    except BaseException as error:
        for future in futures:
            future.cancel()
        wait(futures)
        with _compilation_lock:
            del _compilations[key]
        pending.set_exception(error)
        raise
    # Coalesce only concurrent requests. Later materializations revalidate disk
    # dependencies instead of keeping an unchecked, permanent successful Future.
    with _compilation_lock:
        del _compilations[key]
    pending.set_result(result)
    return result
