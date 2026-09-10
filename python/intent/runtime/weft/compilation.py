from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class TargetProfile:
    march: str
    abi: str
    vlen_bits: int
    cpus: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.march.startswith("rv64") or self.abi != "lp64d":
            raise NotImplementedError("native Weft currently requires RV64/lp64d")
        if self.vlen_bits <= 0 or not self.cpus or any(cpu < 0 for cpu in self.cpus):
            raise ValueError("native Weft requires an explicit VLEN and CPU execution set")


def invoke_compiler(command: list[str], source: str | None = None) -> str:
    result = subprocess.run(command, input=source, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"compiler failed ({command[0]}):\n{result.stderr}{result.stdout}")
    return result.stdout


def lower_artifact(source: str, *, compiler: str, profile: TargetProfile) -> dict:
    artifact = json.loads(invoke_compiler(
        [compiler, "--emit=artifact", f"--march={profile.march}", f"--abi={profile.abi}",
         f"--vlen-bits={profile.vlen_bits}"], source,
    ))
    _validate_target(artifact, profile)
    return artifact


def _validate_target(artifact: dict, profile: TargetProfile) -> None:
    if artifact["kind"] != "weft-riscv-artifact":
        raise ValueError("Weft compiler did not produce a native artifact")
    for kernel in artifact["kernels"]:
        if (kernel["march"], kernel["abi"], kernel["vlen_bits"]) != (
            profile.march, profile.abi, profile.vlen_bits,
        ) or kernel["matrix_extensions"]:
            raise NotImplementedError("Weft artifact does not match the selected standard RVV profile")


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


def export_artifact(program, directory: Path, *, compiler: str, profile: TargetProfile) -> None:
    """AOT lowering; system compilation and native loading remain separate."""
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
