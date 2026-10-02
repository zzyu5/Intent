# Installation

The public setup uses Linux and Python 3.10–3.12. One installer selects the backend dependencies and installs either an existing Intent wheel or a package built from source. Python MLIR bindings are not needed. GPU execution requires an appropriate NVIDIA driver; CPU and MLU toolchains are selected explicitly below. Installation, successful compilation, and numerical correctness are separate checks.

## Install an existing wheel

Set `INTENT_WHEEL` to the actual path of a built `.whl` file, preserving its filename. From the checkout:

```bash
python3 environment/install.py --backend triton \
  --venv .venv-triton --wheel "$INTENT_WHEEL"
source .venv-triton/bin/activate
intent doctor --target triton
python examples/softmax.py
```

This route does not inspect SDK paths, build C++, or install MLIR bindings. The wheel includes `intent-compile`, `intent-opt`, their shared profiles, the language manual, and the tools' non-system ELF dependencies. The binaries find their libraries through relative `$ORIGIN` paths. Third-party copyright/license files and a library-to-notice listing are installed under `intent/_bin/third-party/`.

These are local Linux platform wheels. They require a compatible host architecture, glibc, libstdc++, libgcc, and selected backend environment; bundling does not make a binary built against a newer system work on an older one. No manylinux compatibility or public release is implied. For an environment whose backend dependencies are already installed, `python -m pip install "${INTENT_WHEEL}[manual]"` installs the package directly without a checkout.

For compiler/KIR tools and the MCP servers alone, the same wheel installation needs no provider SDK or tensor framework. Run `intent doctor --json` to check the installed compiler and its KIR output stage. Add a selected backend's dependencies when you need its runtime; a base compiler check does not establish device or kernel support.

## Source build prerequisites

Install CMake, Ninja, a C++17 compiler, Python's venv support, and the LLVM/MLIR 20 C++ SDK: headers, libraries, TableGen tools, and CMake packages. With the [LLVM package repository](https://apt.llvm.org/) configured on Debian/Ubuntu, the relevant packages are `llvm-20-dev`, `libmlir-20-dev`, and `mlir-20-tools`. The SDK is a build dependency; the installed C++ compiler owns KIR parsing, normalization, and verification.

## Choose a backend

Use a separate virtual environment for each public GPU route. The dependency declarations are in [backends.py](../python/intent/tools/backends.py); the installer and installed environment checker consume the same backend descriptions.

| Backend | Python runtime route | External requirements and current scope |
|---|---|---|
| `triton` | CUDA PyTorch 2.10.0, [triton.txt](triton.txt) | NVIDIA driver; callable GPU artifacts |
| `cutile` | CUDA PyTorch 2.10.0, [cutile-runtime.txt](cutile-runtime.txt) | NVIDIA driver compatible with the selected CUDA toolkit; callable GPU artifacts |
| `mojo` | CPU PyTorch 2.10.0 | Existing Mojo compiler, Linux x86-64 with AVX2/AVX512; callable CPU artifacts |
| `weft` | No additional tensor framework for source generation | Intent compiler built with Weft; public `WeftTarget` generates canonical Weft source, native execution requires its deployment/toolchain |
| `bangc` | Native buffer interface uses the Python standard library | Existing NeuWare SDK and MLU370 implementation; execution requires an actual compatible MLU device |

For Triton, from the Intent checkout:

```bash
python3 environment/install.py --backend triton \
  --venv .venv-triton
source .venv-triton/bin/activate
intent doctor --target triton
python examples/softmax.py
```

For cuTile, using the same source SDK:

```bash
python3 environment/install.py --backend cutile \
  --venv .venv-cutile
source .venv-cutile/bin/activate
intent doctor --target cutile
python examples/softmax.py --target cutile
```

The public cuTile route uses cuda-tile 1.6 and CUDA toolkit 13.3.1. The older [cutile.txt](cutile.txt), including TileGym, remains the historical experiment environment; it is not installed into the public route. This does not change its recorded results. Backend/device support still depends on the selected SDK and the kernel's operations.

The common installer:

1. Creates or reuses the chosen virtual environment, without adding the caller's `PYTHONPATH` or user site packages.
2. Installs only the selected backend's Python dependencies.
3. Installs the selected wheel, or builds `intent-compile` and `intent-opt` and packages their runtime libraries and notices.
4. Queries the installed compiler and checks the selected backend. The public CLI and both MCP entry points are included.

For another build SDK location, supply `--mlir-dir` and `--llvm-dir`, or set `INTENT_MLIR_DIR` and `INTENT_LLVM_DIR`. Add `--wheel "$INTENT_WHEEL"` to install an existing wheel instead; SDK/build flags are then unnecessary and rejected. `--torch-index-url` overrides the selected route's wheel index for the same PyTorch release; check the [PyTorch installation combinations](https://pytorch.org/get-started/previous-versions/).

Intent's C++ build defaults to the user cache, outside the checkout. `--build-dir` selects a different build directory and `--jobs` limits build parallelism. Re-running the installer reuses the environment and build directory. Failures stop installation and preserve the underlying command's diagnostics.

## Select an external CPU or MLU toolchain

These commands install Intent around existing toolchains; they do not install a compiler, driver, remote machine, or NeuWare license:

```bash
python3 environment/install.py --backend mojo --venv .venv-mojo \
  --provider-compiler /path/to/mojo

python3 environment/install.py --backend weft --venv .venv-weft \
  --weft-source-dir /path/to/Weft --weft-binary-dir /path/to/weft-build

python3 environment/install.py --backend bangc --venv .venv-bangc \
  --neuware /path/to/neuware
```

For a wheel, add `--wheel "$INTENT_WHEEL"`. Omit Weft source/build flags on that route: the wheel must already contain the Weft provider, which the installed compiler reports. Use selected external compiler/SDK paths in later target constructors or CLI options. For example:

```bash
.venv-mojo/bin/intent doctor --target mojo --target-option 'compiler="/path/to/mojo"'
.venv-weft/bin/intent doctor --target weft --target-option vector_bits=256 --target-option workers=8
.venv-bangc/bin/intent doctor --target bangc --target-option 'neuware="/path/to/neuware"'
```

Weft's vector width and worker count above are an example of explicit construction budgets; supply the values for your deployment. Its CMake integration requires matching LLVM/MLIR and generated Weft headers/libraries. [CPU](../experiments/cpu/README.md) and [MLU](../experiments/mlu/README.md) retain the production deployment instructions and measured coverage. The public installer does not infer those settings from a private profile.

## Build the package directly

Once the SDK and selected backend dependencies are installed:

```bash
python -m pip install '.[manual]' \
  --config-settings=cmake.define.MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  --config-settings=cmake.define.LLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  --config-settings=build-dir=/path/to/intent-build
```

To build a wheel for later installation, use `python -m pip wheel . --no-deps --wheel-dir /path/to/wheelhouse` with the same CMake settings. PEP 517 installs build-only `patchelf` in its isolated build environment; provide it yourself if disabling build isolation. The package contains an independent compiler executable, so its Python ABI tag is `py3-none`; its platform tag remains Linux-specific. Package metadata comes from Git; the build does not create a release tag.

The installed package discovers its own compiler. Developers can override it with `compiler="/path/to/intent-compile"` or `INTENT_COMPILER`. Keep `profiles/`, private `lib/`, and the executable together. A normal CMake developer installation outside the Python wheel build continues to use the selected SDK's runtime libraries.

### Runtime libraries and notices

Wheel installation rules collect the actual compiler's ELF dependency closure. Unresolved dependencies, conflicting SONAMEs, or missing notices fail packaging. Libraries keep their SONAMEs; the staged copies receive relative RPATHs. The SDK's original files are not changed. glibc, its loader, libstdc++, and libgcc remain platform requirements.

For a Debian/Ubuntu SDK, notices come from each resolved library's actual package ownership and copyright file, including referenced `/usr/share/common-licenses/` texts. For custom SDKs or libraries without package ownership, provide a JSON object mapping exact library paths to their full notice/license files. Relative paths are resolved against the JSON file's directory:

```json
{
  "/path/to/sdk/lib/libMLIR.so.20.1": ["licenses/llvm-LICENSE.txt"],
  "/path/to/sdk/lib/libLLVM.so.20.1": ["licenses/llvm-LICENSE.txt"]
}
```

Use `--runtime-notices /path/to/notices.json` with the source installer, or `--config-settings=cmake.define.INTENT_RUNTIME_NOTICES=/path/to/notices.json` with `pip wheel`. The example covers only those two libraries; all other copied dependencies also need their corresponding notices. The map supplies attribution documents, not permission to redistribute third-party software or a license for Intent itself.

## Compile and diagnose

`intent describe --target BACKEND --json` reads the installed public API's signatures, compile option declarations, and target fields without probing an SDK or device. Omit `--target` to see all declared provider routes. `intent doctor --json` checks the base compiler independently of those routes.

`intent doctor --target BACKEND --json` launches the compiler's `--compiler-info` query and checks its actual built providers, then checks only the selected backend's dependencies and target facts. A built source provider is not a claim that a device or kernel is supported. `intent compile path/to/program.py:kernel --stage kir --json` captures, normalizes, and verifies KIR without a target/device. `--target BACKEND` instead selects full provider generation; `--stage shared --target BACKEND` stops after the execution-family program. These commands use the public pipeline and return artifact paths or the actual failure stage. An importable `package.module:kernel` is also accepted. `--constexpr NAME=JSON`, `--target-option NAME=JSON`, `--compiler`, and `--tuning-config` forward existing public settings. `--materialize` applies only to provider generation; the tool does not actively launch the selected kernel.

Use `intent read-artifact PATH --offset 0 --limit 16000 --json` on a returned source, IR, metadata, or log path. Offsets count Unicode characters; continue with `next_offset` until `eof`. Compilation failures retain their actual stage, exception causes and artifact paths; native compilation may report a different directory from the Intent compiler's output.

The `intent-manual` MCP stays read-only. Explicitly enable `intent-compiler-mcp` when an agent should compile an existing user-supplied `.py` file; its compilation tool shares the CLI implementation. Its `describe`, `environment`, and `read_artifact` tools provide discovery, prerequisite inspection, and paged artifact/log reading through that same implementation. Both servers use the `manual` dependency extra. Module loading executes ordinary top-level Python host code, while kernel execution and numerical validation remain separate.
