from __future__ import annotations

from pathlib import Path

from intent.api import KernelDefinition
from intent.frontend import lower_to_mlir
from intent.runtime import CompiledArtifact
from intent.targets.base import Target

from .toolchain import CompilationStageError
from .toolchain import run_compiler
from .toolchain import run_shared_compiler


def compile(
    definition: KernelDefinition[object, object],
    *,
    target: Target,
    compiler: str | Path,
    constexprs: dict[str, object] | None = None,
    tuning_config: str | Path | None = None,
) -> CompiledArtifact:
    try:
        kernel_mlir = lower_to_mlir(definition, constexprs=constexprs)
    except Exception as error:
        raise CompilationStageError("frontend_kir", str(error)) from error
    try:
        resolved = target.resolve()
    except Exception as error:
        raise CompilationStageError("target_resolution", str(error)) from error
    source, realized_mlir, metadata = run_compiler(
        compiler,
        kernel_mlir,
        resolved.compiler_options + (
            (f"--tuning-config={Path(tuning_config).resolve()}",)
            if tuning_config is not None else ()
        ),
        resolved.compiler_role,
    )
    try:
        return resolved.materialize(source, realized_mlir, definition.__name__, metadata)
    except Exception as error:
        raise CompilationStageError(
            "generated_source_materialization", str(error)
        ) from error


def compile_shared_gpu(
    definition: KernelDefinition[object, object],
    *,
    target: Target,
    compiler: str | Path,
    constexprs: dict[str, object] | None = None,
    tuning_config: str | Path | None = None,
) -> str:
    try:
        kernel_mlir = lower_to_mlir(definition, constexprs=constexprs)
    except Exception as error:
        raise CompilationStageError("frontend_kir", str(error)) from error
    try:
        resolved = target.resolve()
    except Exception as error:
        raise CompilationStageError("target_resolution", str(error)) from error
    return run_shared_compiler(
        compiler,
        kernel_mlir,
        resolved.compiler_options + (
            (f"--tuning-config={Path(tuning_config).resolve()}",)
            if tuning_config is not None else ()
        ),
        resolved.compiler_role,
    )
