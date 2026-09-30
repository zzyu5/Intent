#!/usr/bin/env python3
"""Install the NVIDIA/Triton source distribution into a virtual environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def run(*command: str | Path, env: dict[str, str]) -> None:
    arguments = [str(value) for value in command]
    print("+ " + shlex.join(arguments), flush=True)
    subprocess.run(arguments, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    parser.add_argument("--mlir-build", type=Path,
                        help="LLVM build containing the completed MLIRPythonModules target")
    parser.add_argument("--mlir-dir", type=Path,
                        default=Path(os.environ.get("INTENT_MLIR_DIR", "/usr/lib/llvm-20/lib/cmake/mlir")))
    parser.add_argument("--llvm-dir", type=Path,
                        default=Path(os.environ.get("INTENT_LLVM_DIR", "/usr/lib/llvm-20/lib/cmake/llvm")))
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    parser.add_argument("--build-dir", type=Path,
                        default=cache / "intentdsl" / "install-build" / sys.implementation.cache_tag)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu130")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this installer supports Linux; see the backend documentation for other toolchains")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    for directory, filename in ((args.mlir_dir, "MLIRConfig.cmake"), (args.llvm_dir, "LLVMConfig.cmake")):
        if not (directory.expanduser() / filename).is_file():
            parser.error(f"{filename} not found in {directory}; install the LLVM/MLIR SDK or set its directory")
    repository = Path(__file__).resolve().parents[1]
    destination = args.venv.expanduser().resolve()
    if destination.exists() and not (destination / "pyvenv.cfg").is_file():
        parser.error(f"{destination} exists but is not a virtual environment")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
    if not destination.exists():
        run(sys.executable, "-m", "venv", destination, env=env)
    python = destination / "bin" / "python"
    interpreter = json.loads(subprocess.check_output(
        [str(python), "-c", "import json, sys, sysconfig; "
         "print(json.dumps({'version': list(sys.version_info[:2]), 'site': sysconfig.get_path('purelib')}))"],
        text=True, env=env,
    ))
    if not (3, 10) <= tuple(interpreter["version"]) <= (3, 12):
        parser.error("the pinned Triton dependency set requires Python 3.10–3.12; select a matching virtual environment")
    if args.mlir_build is not None:
        run("cmake", "--install", args.mlir_build.expanduser().resolve(),
            "--component", "MLIRPythonModules", "--prefix", destination, env=env)
        bindings = destination / "python_packages" / "mlir_core"
        if not (bindings / "mlir" / "ir.py").is_file():
            parser.error("MLIRPythonModules was not installed; build that target with Python bindings enabled first")
        (Path(interpreter["site"]) / "intentdsl-mlir.pth").write_text(str(bindings) + "\n", encoding="utf-8")
    print("Checking MLIR Python bindings (the Python ABI must match this virtual environment).", flush=True)
    run(python, "-c", "from mlir.dialects import func; import mlir.ir; print(mlir.ir.__file__)", env=env)
    run(python, "-m", "pip", "install", "--upgrade", "pip", env=env)
    run(python, "-m", "pip", "install", "torch==2.10.0", "--index-url", args.torch_index_url, env=env)
    run(python, "-m", "pip", "install", "-r", repository / "environment" / "triton.txt", env=env)
    run(python, "-m", "pip", "install", str(repository) + "[manual]",
        "--config-settings=cmake.define.MLIR_DIR=" + str(args.mlir_dir.expanduser().resolve()),
        "--config-settings=cmake.define.LLVM_DIR=" + str(args.llvm_dir.expanduser().resolve()),
        "--config-settings=build-dir=" + str(args.build_dir.expanduser().resolve()), env=env)
    run(python, "-c", "import intent; from intent.compiler.toolchain import _resolve_compiler; "
        "print('Intent:', intent.__file__); print('Compiler:', _resolve_compiler(None, 'Intent compiler'))", env=env)
    print(f"Installed. Activate with: source {shlex.quote(str(destination / 'bin' / 'activate'))}")
    print(f"Run the example: {shlex.quote(str(python))} {shlex.quote(str(repository / 'examples' / 'softmax.py'))}")
    print(f"Start the manual MCP: {shlex.quote(str(destination / 'bin' / 'intent-manual'))}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
