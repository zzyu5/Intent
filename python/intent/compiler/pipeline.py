from __future__ import annotations

from pathlib import Path

from intent.api import KernelDefinition
from intent.frontend import lower_to_mlir
from intent.runtime import CompiledArtifact
from intent.targets.base import Target, SourceTarget, ResolvedSourceTarget, ResolvedTarget

from .artifact import GeneratedProgram
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
    kernel_mlir, resolved = _inputs(definition, target, constexprs)
    if not isinstance(resolved, ResolvedTarget):
        raise NotImplementedError("This target only generates source; use intent.generate, not intent.compile")
    program = _generate_source(kernel_mlir, resolved, compiler, tuning_config)
    try:
        return resolved.materialize(program.source, program.ir, definition.__name__, program.metadata)
    except Exception as error:
        raise CompilationStageError("generated_source_materialization", str(error)) from error


def generate(
    definition: KernelDefinition[object, object],
    *,
    target: SourceTarget,
    compiler: str | Path,
    constexprs: dict[str, object] | None = None,
    tuning_config: str | Path | None = None,
) -> GeneratedProgram:
    kernel_mlir, resolved = _inputs(definition, target, constexprs)
    return _generate_source(kernel_mlir, resolved, compiler, tuning_config)


def _inputs(definition, target, constexprs) -> tuple[str, ResolvedSourceTarget]:
    try:
        kernel_mlir = lower_to_mlir(definition, constexprs=constexprs)
    except Exception as error:
        raise CompilationStageError("frontend_kir", str(error)) from error
    try:
        resolved = target.resolve()
    except Exception as error:
        raise CompilationStageError("target_resolution", str(error)) from error
    return kernel_mlir, resolved


def _generate_source(kernel_mlir, resolved, compiler, tuning_config) -> GeneratedProgram:
    source, realized_mlir, metadata = run_compiler(
        compiler,
        kernel_mlir,
        resolved.compiler_options + (
            (f"--tuning-config={Path(tuning_config).resolve()}",)
            if tuning_config is not None else ()
        ),
        resolved.compiler_role,
    )
    return GeneratedProgram(source, realized_mlir, metadata)


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
