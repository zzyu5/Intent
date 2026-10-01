from __future__ import annotations

import subprocess
import json
import os
from pathlib import Path
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum

from .cache import _compilation_key, compilation_directory


class CompilationStageError(RuntimeError):
    def __init__(self, stage: str, message: str, *, cache_directory: Path | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.cache_directory = cache_directory


_STAGE_BY_EXIT_CODE = {
    1: "compiler_invocation",
    2: "kernel_ir",
    3: "physical_program",
    4: "physical_program_verification",
    5: "provider_program",
    6: "provider_program_verification",
    7: "terminal_translation",
    8: "compiler_output",
}


def _resolve_compiler(executable_path: str | Path | None, role: str) -> Path:
    selected = executable_path if executable_path is not None else os.environ.get("INTENT_COMPILER")
    if selected is None:
        bundled = Path(__file__).resolve().parents[1] / "_bin" / "intent-compile"
        selected = bundled if bundled.exists() else shutil.which("intent-compile")
    if selected is None:
        raise CompilationStageError(
            "compiler_invocation",
            "intent-compile was not found. Install the IntentDSL package with its compiler, "
            "or set INTENT_COMPILER / pass compiler= to a built intent-compile executable.",
        )
    executable = Path(selected).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise CompilationStageError(
            "compiler_invocation", f"{role} is not an executable file: {executable}"
        )
    return executable


class CompilerStage(str, Enum):
    KIR = "kir"
    SHARED = "shared"
    PROVIDER = "provider"

    @property
    def options(self) -> tuple[str, ...]:
        return () if self is CompilerStage.PROVIDER else (f"--stop-after-{self.value}",)

    @property
    def outputs(self) -> tuple[str, ...]:
        return ("kernel.mlir", "kernel.source", "artifact.json") if self is CompilerStage.PROVIDER else ("kernel.mlir",)


@dataclass(frozen=True)
class CompilerOutput:
    stage: CompilerStage
    directory: Path
    ir: str
    source: str | None = None
    metadata: dict[str, object] | None = None


def compiler_info(executable_path: str | Path | None = None) -> dict[str, object]:
    """Query the installed compiler process, without loading a target or device."""
    executable = _resolve_compiler(executable_path, "Intent compiler")
    try:
        completed = subprocess.run([str(executable), "--compiler-info"], text=True,
                                   capture_output=True, check=False)
    except OSError as error:
        raise CompilationStageError("compiler_invocation", str(error)) from error
    if completed.returncode:
        raise CompilationStageError("compiler_invocation", f"Cannot start {executable}:\n"
                                    f"{completed.stderr}{completed.stdout}")
    try:
        info = json.loads(completed.stdout)
        if not isinstance(info, dict):
            raise ValueError("compiler info must be a JSON object")
        for name in ("providers", "stages"):
            value = info.get(name)
            if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
                raise ValueError(f"compiler info {name} must be a nonempty list of names")
        outputs = info.get("outputs")
        if not isinstance(outputs, dict) or any(
            not isinstance(outputs.get(stage), list)
            or any(not isinstance(item, str) for item in outputs[stage])
            for stage in info["stages"]
        ):
            raise ValueError("compiler info must describe outputs for every stage")
    except (ValueError, TypeError) as error:
        raise CompilationStageError("compiler_output", f"Invalid compiler info from {executable}: {error}") from error
    return {**info, "executable": str(executable)}


def _outputs(directory: Path, *, stage: CompilerStage, role: str) -> CompilerOutput:
    paths = tuple(directory / name for name in stage.outputs)
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise CompilationStageError("compiler_output", f"{role} did not produce complete outputs in {directory}",
                                    cache_directory=directory)
    try:
        ir = paths[0].read_text(encoding="utf-8")
        if stage is not CompilerStage.PROVIDER:
            return CompilerOutput(stage, directory, ir)
        source = paths[1].read_text(encoding="utf-8")
        metadata = json.loads(paths[2].read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("compiler metadata must be a JSON object")
        return CompilerOutput(stage, directory, ir, source, metadata)
    except (UnicodeDecodeError, ValueError) as error:
        raise CompilationStageError("compiler_output", f"{role} produced invalid outputs in {directory}: {error}",
                                    cache_directory=directory) from error


def run_compiler(executable_path: str | Path | None, module_text: str,
                 options: tuple[str, ...], role: str, *,
                 stage: CompilerStage = CompilerStage.PROVIDER) -> CompilerOutput:
    if not isinstance(stage, CompilerStage):
        raise TypeError("compiler stage must be a CompilerStage")
    executable = _resolve_compiler(executable_path, role)
    options = (*options, *stage.options)
    with compilation_directory(executable, module_text, options) as (directory, key):
        complete = directory / "complete"
        if complete.is_file():
            try:
                return _outputs(directory, stage=stage, role=role)
            except CompilationStageError:
                # An incomplete/invalid cached group is a miss. Fresh compiler
                # output below still has to pass the same validation.
                complete.unlink()
        (directory / "input.mlir").write_text(module_text, encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix=".building-", dir=directory) as temporary:
            staging = Path(temporary)
            command = [str(executable), *options, f"--ir-output={staging / 'kernel.mlir'}"]
            if stage is CompilerStage.PROVIDER:
                command.extend((f"--source-output={staging / 'kernel.source'}",
                                f"--metadata-output={staging / 'artifact.json'}"))
            command.append("-")
            try:
                completed = subprocess.run(command, input=module_text, text=True,
                                           capture_output=True, check=False)
            except OSError as error:
                raise CompilationStageError("compiler_invocation", str(error),
                                            cache_directory=directory) from error
            (directory / "compiler.log").write_text(completed.stderr + completed.stdout, encoding="utf-8")
            if completed.returncode:
                raise CompilationStageError(
                    _STAGE_BY_EXIT_CODE.get(completed.returncode, "compiler_process"),
                    f"{role} failed with exit code {completed.returncode}:\n"
                    f"{completed.stderr}{completed.stdout}\nCompiler artifacts: {directory}",
                    cache_directory=directory,
                )
            for name in ("kernel.source", "kernel.mlir", "artifact.json"):
                (directory / name).unlink(missing_ok=True)
            for output in staging.iterdir():
                output.replace(directory / output.name)
        result = _outputs(directory, stage=stage, role=role)
        if _compilation_key(executable, module_text, options) != key:
            raise CompilationStageError(
                "compiler_invocation", "compiler or profiles changed during compilation; retry with stable inputs",
                cache_directory=directory,
            )
        complete.touch()
        return result
