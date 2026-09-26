from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess


from .target import TargetProfile


def invoke_compiler(command: list[str], source: str | None = None) -> str:
    result = subprocess.run(command, input=source, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"compiler failed ({command[0]}):\n{result.stderr}{result.stdout}")
    return result.stdout


def lower_artifact(source: str, *, compiler: str, profile: TargetProfile,
                   source_bindings: tuple[tuple[str, int], ...] = ()) -> dict:
    artifact = json.loads(invoke_compiler(
        [compiler, "--emit=artifact", f"--march={profile.march}", f"--abi={profile.abi}",
         f"--vlen-bits={profile.vlen_bits}",
         f"--private-stack-bytes={profile.private_stack_bytes}",
         *([f"--matrix-extension={profile.matrix_extension}"] if profile.matrix_extension else []),
         *(f"--meta={name}={value}" for name, value in source_bindings)], source,
    ))
    _validate_target(artifact, profile)
    return artifact


def _validate_target(artifact: dict, profile: TargetProfile) -> None:
    if artifact["kind"] != "weft-riscv-artifact":
        raise ValueError("Weft compiler did not produce a native artifact")
    for kernel in artifact["kernels"]:
        if (kernel["march"], kernel["abi"], kernel["vlen_bits"]) != (
            profile.march, profile.abi, profile.vlen_bits,
        ) or set(kernel["matrix_extensions"]) != ({profile.matrix_extension} if profile.matrix_extension else set()):
            raise ValueError("Weft artifact does not match the selected target profile")
        if not set(kernel["required_extensions"]).issubset(kernel["used_extensions"]) or not set(
                kernel["used_extensions"]).issubset(kernel["matrix_extensions"]):
            raise ValueError("Weft artifact extension requirements disagree with its selected instructions")
    # An Intent invocation can also contain RVV-only preparation and copy tasks.
    # Its extension requirement applies to the complete executable candidate.
    used = {extension for kernel in artifact["kernels"] for extension in kernel["used_extensions"]}
    if not set(profile.required_extensions).issubset(used):
        raise ValueError("Weft program does not use the required matrix extension")


def validate_artifact(manifest: dict) -> None:
    artifact = manifest["weft"]
    _validate_target(artifact, TargetProfile(**manifest["profile"]))
    kernels = {kernel["symbol"]: kernel for kernel in artifact["kernels"]}
    expected = {task["abi"]["symbol"]: task["abi"] for task in manifest["program"]["tasks"]}
    if (len(kernels) != len(artifact["kernels"]) or
            len(expected) != len(manifest["program"]["tasks"]) or kernels.keys() != expected.keys()):
        raise ValueError("Weft artifact kernel symbols disagree with the CPU task calls")
    for symbol, abi in expected.items():
        for field in ("arguments", "shape_parameters"):
            if kernels[symbol][field] != abi[field]:
                raise ValueError(f"Weft artifact {symbol} {field} disagree with the CPU task ABI")
    profile = TargetProfile(**manifest["profile"])
    for candidate in manifest["program"]["candidates"]:
        used = {extension for task in manifest["program"]["tasks"]
                if task["cpu_entry"] == candidate["entry"]
                for extension in kernels[task["abi"]["symbol"]]["used_extensions"]}
        required = set(profile.required_extensions)
        if candidate["requires_matrix_i8_i32"]:
            if profile.matrix_extension is None:
                raise ValueError("Weft candidate requires an unavailable integer matrix capability")
            required.add(profile.matrix_extension)
        if not required.issubset(used):
            raise ValueError("Weft candidate does not use the required matrix extension")


def export_artifact(program, directory: Path, *, compiler: str, profile: TargetProfile) -> None:
    """AOT lowering; system compilation and native loading remain separate."""
    if program.metadata["matrix_i8_i32"] and not profile.matrix_extension:
        raise ValueError("CPU program matrix capability disagrees with native materialization")
    artifact = lower_artifact(program.source, compiler=compiler, profile=profile)
    manifest = {"profile": asdict(profile), "program": program.metadata, "weft": artifact}
    validate_artifact(manifest)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "canonical.mlir").write_text(program.source, encoding="utf-8")
    (directory / "cpu.mlir").write_text(program.ir, encoding="utf-8")
    (directory / "kernels.c").write_text(artifact["intrinsic_c"], encoding="utf-8")
    (directory / "host.c").write_text(program.metadata["host_source"], encoding="utf-8")
    (directory / "artifact.json").write_text(json.dumps(manifest), encoding="utf-8")


def flattened_signature(parameters: list[dict]) -> tuple[list[str], list[str]]:
    signature, arguments = [], []
    for index, parameter in enumerate(parameters):
        name = f"a{index}"
        if parameter["kind"] == "view":
            signature.append(f"void *{name}")
            arguments.append(name)
            for role in ("d", "s"):
                for axis in range(len(parameter["shape"])):
                    field = f"{name}_{role}{axis}"
                    signature.append(f"int64_t {field}")
                    arguments.append(field)
        else:
            signature.append(f"{'float' if parameter['dtype'] == 'f32' else 'int64_t'} {name}")
            arguments.append(name)
    return signature, arguments


def native_exports(metadata: dict) -> str:
    signature, arguments = flattened_signature(metadata["parameters"])
    sections = ["#include <stdint.h>\n#include <time.h>\n#include <fenv.h>\n"]
    for candidate in metadata["candidates"]:
        entry = candidate["entry"]
        sections.append(
            f"extern void {entry}({', '.join(signature)});\n"
            f"void {entry}_invoke({', '.join(signature)}) {{\n"
            "  fenv_t saved; fegetenv(&saved); fesetround(FE_TONEAREST);\n"
            f"  {entry}({', '.join(arguments)});\n"
            "  fesetenv(&saved);\n}\n"
            f"double {entry}_benchmark({', '.join(signature)}, int64_t repetitions) {{\n"
            "  struct timespec begin, end;\n"
            "  fenv_t saved; fegetenv(&saved); fesetround(FE_TONEAREST);\n"
            "  clock_gettime(CLOCK_MONOTONIC, &begin);\n"
            "  for (int64_t iteration = 0; iteration < repetitions; ++iteration)\n"
            f"    {entry}({', '.join(arguments)});\n"
            "  clock_gettime(CLOCK_MONOTONIC, &end); fesetenv(&saved);\n"
            "  return ((end.tv_sec - begin.tv_sec) * 1.e3 + (end.tv_nsec - begin.tv_nsec) * 1.e-6) / repetitions;\n}\n"
        )
    sections.append("int64_t intent_weft_vlen_bits(void) { unsigned long v; __asm__ volatile(\"csrr %0, vlenb\" : \"=r\"(v)); return v * 8; }\n")
    sections.append("void intent_weft_evict(void *storage, int64_t bytes) { volatile uint8_t *p = storage; for (int64_t i = 0; i < bytes; i += 64) p[i] = p[i] + 1; }\n")
    return "".join(sections)


def compile_artifact(directory: Path, *, cc: tuple[str, ...], cflags: tuple[str, ...] = ()) -> Path:
    manifest = json.loads((directory / "artifact.json").read_text())
    validate_artifact(manifest)
    profile = TargetProfile(**manifest["profile"])
    metadata = manifest["program"]
    exports = directory / "exports.c"
    exports.write_text(native_exports(metadata), encoding="utf-8")
    library = directory / "kernel.so"
    invoke_compiler([
        *cc, "-O3", "-shared", "-fPIC", "-std=c11", "-D_POSIX_C_SOURCE=200809L",
        f"-march={profile.march}", f"-mabi={profile.abi}", *cflags,
        "-fno-fast-math", "-ffp-contract=off",
        *(["-fopenmp"] if metadata["workers"] > 1 else []),
        str(directory / "host.c"), str(directory / "kernels.c"), str(exports),
        "-lm", "-o", str(library),
    ])
    return library
