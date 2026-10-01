from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


@dataclass
class NativeCompilation:
    directory: tempfile.TemporaryDirectory
    library: ctypes.CDLL
    command: tuple[str, ...]


_compilations: dict[tuple[object, ...], NativeCompilation] = {}


def compile_library(source: str, target) -> NativeCompilation:
    selected = target.compiler or os.environ.get("INTENT_BANGC_CNCC") or str(Path(target.neuware) / "bin/cncc")
    executable = shutil.which(selected)
    if executable is None:
        raise FileNotFoundError("CNCC was not found; set BangCTarget(compiler=...) or INTENT_BANGC_CNCC")
    library_dir = str(Path(target.neuware) / "lib64")
    options = (f"--neuware-path={target.neuware}", f"--bang-mlu-arch={target.compilation.architecture}", "-O2", "-fPIC", "-shared",
               "-std=c++14", "-ffp-contract=off", "-I" + str(Path(target.neuware) / "include"),
               "-L" + library_dir, "-Wl,-rpath," + library_dir, "-lcnrt")
    identity = (source, executable, Path(executable).stat().st_mtime_ns, options)
    if identity in _compilations:
        return _compilations[identity]
    directory = tempfile.TemporaryDirectory(prefix="intent-bangc-")
    root = Path(directory.name)
    path = root / "kernel.mlu"
    path.write_text(source)
    output = root / "kernel.so"
    command = (executable, *options, str(path), "-o", str(output))
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        directory.cleanup()
        raise RuntimeError(f"CNCC failed ({result.returncode}):\n{result.stdout}{result.stderr}")
    compilation = NativeCompilation(directory, ctypes.CDLL(str(output)), command)
    _compilations[identity] = compilation
    return compilation
