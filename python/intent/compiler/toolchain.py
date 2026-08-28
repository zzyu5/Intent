from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class CompilationStageError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


_STAGE_BY_EXIT_CODE = {
    1: "compiler_invocation",
    2: "kernel_ir_parse",
    3: "physical_program",
    4: "physical_program_verification",
    5: "provider_program",
    6: "provider_program_verification",
    7: "terminal_translation",
    8: "compiler_output",
}


def run_compiler(
    executable_path: str | Path,
    module_text: str,
    options: tuple[str, ...],
    role: str,
) -> tuple[str, str]:
    executable = Path(executable_path)
    if not executable.is_file():
        raise CompilationStageError(
            "compiler_invocation", f"{role} does not exist: {executable}"
        )
    with tempfile.TemporaryDirectory(prefix="intentdsl-compile-") as directory:
        output_directory = Path(directory)
        source_path = output_directory / "kernel.py"
        mlir_path = output_directory / "kernel.mlir"
        try:
            completed = subprocess.run(
                [
                    str(executable),
                    *options,
                    f"--source-output={source_path}",
                    f"--ir-output={mlir_path}",
                    "-",
                ],
                input=module_text,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as error:
            raise CompilationStageError("compiler_invocation", str(error)) from error
        if completed.returncode != 0:
            stage = _STAGE_BY_EXIT_CODE.get(
                completed.returncode, "compiler_process"
            )
            raise CompilationStageError(
                stage,
                f"{role} failed with exit code {completed.returncode}:\n"
                f"{completed.stderr}{completed.stdout}",
            )
        if not source_path.is_file() or not mlir_path.is_file():
            raise CompilationStageError(
                "compiler_output", f"{role} did not produce both compiler outputs"
            )
        source = source_path.read_text(encoding="utf-8")
        realized_mlir = mlir_path.read_text(encoding="utf-8")
        if not source or not realized_mlir:
            raise CompilationStageError(
                "compiler_output", f"{role} produced an empty compiler output"
            )
        return source, realized_mlir


def run_shared_compiler(
    executable_path: str | Path,
    module_text: str,
    options: tuple[str, ...],
    role: str,
) -> str:
    executable = Path(executable_path)
    if not executable.is_file():
        raise CompilationStageError(
            "compiler_invocation", f"{role} does not exist: {executable}"
        )
    with tempfile.TemporaryDirectory(prefix="intentdsl-shared-") as directory:
        mlir_path = Path(directory) / "shared-gpu.mlir"
        try:
            completed = subprocess.run(
                [
                    str(executable),
                    *options,
                    "--stop-after-shared",
                    f"--ir-output={mlir_path}",
                    "-",
                ],
                input=module_text,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as error:
            raise CompilationStageError("compiler_invocation", str(error)) from error
        if completed.returncode != 0:
            stage = _STAGE_BY_EXIT_CODE.get(
                completed.returncode, "compiler_process"
            )
            raise CompilationStageError(
                stage,
                f"{role} failed with exit code {completed.returncode}:\n"
                f"{completed.stderr}{completed.stdout}",
            )
        if not mlir_path.is_file():
            raise CompilationStageError(
                "compiler_output", f"{role} did not produce shared GPU IR"
            )
        realized_mlir = mlir_path.read_text(encoding="utf-8")
        if not realized_mlir:
            raise CompilationStageError(
                "compiler_output", f"{role} produced empty shared GPU IR"
            )
        return realized_mlir
