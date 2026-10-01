#!/usr/bin/env python3
"""Install Intent and one selected backend into an explicit virtual environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
# This declaration module uses only the standard library. Loading it does not
# import Intent or require any backend before the environment has been created.
DECLARATIONS = runpy.run_path(str(REPOSITORY / "python/intent/tools/backends.py"))
BACKENDS = DECLARATIONS["BACKENDS"]


def run(*command: str | Path, env: dict[str, str]) -> None:
    arguments = [str(value) for value in command]
    print("+ " + shlex.join(arguments), flush=True)
    subprocess.run(arguments, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--venv", type=Path, help="Default: .venv-BACKEND")
    parser.add_argument("--mlir-build", type=Path, help="Matching LLVM build with completed MLIRPythonModules")
    parser.add_argument("--mlir-dir", type=Path,
                        default=Path(os.environ.get("INTENT_MLIR_DIR", "/usr/lib/llvm-20/lib/cmake/mlir")))
    parser.add_argument("--llvm-dir", type=Path,
                        default=Path(os.environ.get("INTENT_LLVM_DIR", "/usr/lib/llvm-20/lib/cmake/llvm")))
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--torch-index-url", help="Override the selected backend's PyTorch wheel index")
    parser.add_argument("--provider-compiler", type=Path, help="Existing Mojo, Weft, or BANG C compiler")
    parser.add_argument("--weft-source-dir", type=Path)
    parser.add_argument("--weft-binary-dir", type=Path)
    parser.add_argument("--neuware", type=Path, help="Existing NeuWare SDK for BANG C")
    args = parser.parse_args()
    selected = BACKENDS[args.backend]
    if sys.platform != "linux":
        parser.error("the packaged backend setup currently supports Linux")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if (args.weft_source_dir or args.weft_binary_dir) and args.backend != "weft":
        parser.error("Weft source/build flags require --backend weft")
    if args.neuware and args.backend != "bangc":
        parser.error("--neuware requires --backend bangc")
    if args.torch_index_url and selected["torch"] is None:
        parser.error("this backend does not install PyTorch")
    for directory, filename in ((args.mlir_dir, "MLIRConfig.cmake"), (args.llvm_dir, "LLVMConfig.cmake")):
        if not (directory.expanduser() / filename).is_file():
            parser.error(f"{filename} not found in {directory}; select an LLVM/MLIR 20 SDK")
    options = {}
    extra_cmake = []
    if args.backend == "mojo":
        compiler = args.provider_compiler or os.environ.get("INTENT_MOJO")
        if compiler is None:
            parser.error("Mojo must already be installed; provide --provider-compiler or INTENT_MOJO")
        resolved = shutil.which(os.path.expanduser(str(compiler)))
        if resolved is None:
            parser.error(f"Mojo compiler is not executable: {compiler}")
        options["compiler"] = str(Path(resolved).resolve())
    elif args.backend == "weft":
        if args.weft_source_dir is None or args.weft_binary_dir is None:
            parser.error("Weft requires explicit --weft-source-dir and --weft-binary-dir matching the MLIR SDK")
        source, build = args.weft_source_dir.expanduser().resolve(), args.weft_binary_dir.expanduser().resolve()
        for path in (source / "include/Weft/Dialect/Kernel/IR/KernelDialect.h",
                     build / "include/Weft/Dialect/Kernel/IR/KernelOps.h.inc"):
            if not path.is_file():
                parser.error(f"Weft source/build is incomplete: {path}")
        compiler = args.provider_compiler or build / "tools/weft-compile/weft-compile"
        if shutil.which(os.path.expanduser(str(compiler))) is None:
            parser.error(f"Weft compiler is not executable: {compiler}")
        extra_cmake = [f"--config-settings=cmake.define.INTENT_WEFT_SOURCE_DIR={source}",
                       f"--config-settings=cmake.define.INTENT_WEFT_BINARY_DIR={build}"]
    elif args.backend == "bangc":
        if args.neuware is None:
            parser.error("BANG C requires an existing SDK selected by --neuware")
        sdk = args.neuware.expanduser().resolve()
        compiler = args.provider_compiler or sdk / "bin/cncc"
        resolved = shutil.which(os.path.expanduser(str(compiler)))
        if resolved is None or not (sdk / "lib64/libcnrt.so").is_file():
            parser.error("NeuWare must supply an executable cncc and lib64/libcnrt.so")
        options.update(neuware=str(sdk), compiler=str(Path(resolved).resolve()))
    elif args.provider_compiler or args.neuware or args.weft_source_dir or args.weft_binary_dir:
        parser.error("external compiler/SDK flags apply only to their selected CPU or MLU backend")
    destination = (args.venv or Path(f".venv-{args.backend}")).expanduser().resolve()
    if destination.exists() and not (destination / "pyvenv.cfg").is_file():
        parser.error(f"{destination} exists but is not a virtual environment")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
    if not destination.exists():
        run(sys.executable, "-m", "venv", destination, env=env)
    python = destination / "bin/python"
    interpreter = json.loads(subprocess.check_output(
        [str(python), "-c", "import json,sys,sysconfig; print(json.dumps({'version':list(sys.version_info[:2]),"
         "'site':sysconfig.get_path('purelib')}))"], text=True, env=env))
    if not (3, 10) <= tuple(interpreter["version"]) <= selected["python_max"]:
        parser.error("the public dependency routes require Python 3.10–3.12 with matching MLIR bindings")
    run(python, "-m", "pip", "install", "--upgrade", "pip", env=env)
    run(python, "-m", "pip", "install", *DECLARATIONS["FRONTEND_REQUIREMENTS"], env=env)
    if args.mlir_build is not None:
        run("cmake", "--install", args.mlir_build.expanduser().resolve(), "--component", "MLIRPythonModules",
            "--prefix", destination, env=env)
        bindings = destination / "python_packages/mlir_core"
        if not (bindings / "mlir/ir.py").is_file():
            parser.error("build MLIRPythonModules with this Python ABI before installation")
        (Path(interpreter["site"]) / "intentdsl-mlir.pth").write_text(str(bindings) + "\n", encoding="utf-8")
    run(python, "-c", "from mlir.dialects import func; import mlir.ir; print(mlir.ir.__file__)", env=env)
    if selected["torch"] is not None:
        run(python, "-m", "pip", "install", f"torch=={selected['torch']}", "--index-url",
            args.torch_index_url or selected["torch_index"], env=env)
    if selected["requirements"] is not None:
        run(python, "-m", "pip", "install", "-r", REPOSITORY / "environment" / selected["requirements"], env=env)
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    build = (args.build_dir or cache / "intentdsl/install-build" / args.backend / sys.implementation.cache_tag).resolve()
    run(python, "-m", "pip", "install", str(REPOSITORY) + "[manual]",
        "--config-settings=cmake.define.MLIR_DIR=" + str(args.mlir_dir.expanduser().resolve()),
        "--config-settings=cmake.define.LLVM_DIR=" + str(args.llvm_dir.expanduser().resolve()),
        "--config-settings=build-dir=" + str(build), *extra_cmake, env=env)
    print(f"Installed {args.backend}. Activate: source {shlex.quote(str(destination / 'bin/activate'))}")
    if args.backend == "weft":
        print("Weft is available for source generation. Supply your actual vector_bits/workers to intent doctor/compile;")
        print("native RISC-V execution additionally requires an explicit deployment and matching external toolchain.")
    else:
        command = [str(destination / "bin/intent"), "doctor", "--target", args.backend]
        for key, value in options.items():
            command.extend(("--target-option", f"{key}={json.dumps(value)}"))
        run(*command, env=env)
        if options:
            print("Use the same --target-option settings in subsequent CLI calls, or the corresponding Target constructor arguments.")
    print(f"Manual MCP: {destination / 'bin/intent-manual'}")
    print(f"Optional compiler MCP: {destination / 'bin/intent-compiler-mcp'}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
