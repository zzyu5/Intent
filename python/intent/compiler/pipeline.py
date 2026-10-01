from __future__ import annotations

from pathlib import Path

from intent.api import KernelDefinition
from intent.frontend import lower_to_mlir
from intent.runtime import CompiledArtifact
from intent.targets.base import Target, SourceTarget, ResolvedSourceTarget, ResolvedTarget

from .artifact import CompiledIR, GeneratedProgram
from .toolchain import CompilationStageError
from .toolchain import CompilerStage
from .toolchain import run_compiler


def compile(
    definition: KernelDefinition[object, object],
    *,
    target: Target,
    compiler: str | Path | None = None,
    constexprs: dict[str, object] | None = None,
    tuning_config: str | Path | None = None,
) -> CompiledArtifact:
    kernel_mlir = _capture(definition, constexprs)
    resolved = _resolve(target)
    if not isinstance(resolved, ResolvedTarget):
        raise NotImplementedError("This target only generates source; use intent.generate, not intent.compile")
    program = _generate_source(kernel_mlir, resolved, compiler, tuning_config, definition.__name__)
    return program.materialize()


def generate(
    definition: KernelDefinition[object, object],
    *,
    target: SourceTarget,
    compiler: str | Path | None = None,
    constexprs: dict[str, object] | None = None,
    tuning_config: str | Path | None = None,
) -> GeneratedProgram:
    kernel_mlir = _capture(definition, constexprs)
    resolved = _resolve(target)
    return _generate_source(kernel_mlir, resolved, compiler, tuning_config, definition.__name__)


def compile_ir(
    definition: KernelDefinition[object, object],
    *,
    stage: str = "kir",
    target: SourceTarget | None = None,
    compiler: str | Path | None = None,
    constexprs: dict[str, object] | None = None,
    tuning_config: str | Path | None = None,
) -> CompiledIR:
    """Compile a definition to verified KIR or shared physical IR.

    ``stage='kir'`` normalizes and verifies target-independent Kernel IR using
    the Intent compiler. It accepts neither a target nor a tuning configuration,
    and does not resolve a device or import a provider SDK. ``stage='shared'``
    requires a target and stops after that execution family's shared pipeline.
    Both return IR and its cache directory without materialization or launch;
    use ``generate`` for provider source and its invocation metadata.
    """
    selected = CompilerStage(stage)
    if selected is CompilerStage.PROVIDER:
        raise ValueError("compile_ir supports 'kir' and 'shared'; use generate for provider source")
    if selected is CompilerStage.KIR:
        if target is not None or tuning_config is not None:
            raise ValueError("KIR compilation does not accept a target or tuning configuration")
    elif target is None:
        raise ValueError("shared IR compilation requires a target")
    kernel_mlir = _capture(definition, constexprs)
    if selected is CompilerStage.KIR:
        options, role = (), "Intent compiler"
    else:
        resolved = _resolve(target)
        options, role = _options(resolved, tuning_config), resolved.compiler_role
    output = run_compiler(compiler, kernel_mlir, options, role, stage=selected)
    return CompiledIR(selected.value, output.ir, output.directory)


def _capture(definition, constexprs) -> str:
    try:
        return lower_to_mlir(definition, constexprs=constexprs)
    except Exception as error:
        raise CompilationStageError("frontend_kir", str(error)) from error


def _resolve(target: SourceTarget) -> ResolvedSourceTarget:
    try:
        return target.resolve()
    except Exception as error:
        raise CompilationStageError("target_resolution", str(error)) from error


def _options(resolved: ResolvedSourceTarget, tuning_config: str | Path | None) -> tuple[str, ...]:
    return resolved.compiler_options + (
        (f"--tuning-config={Path(tuning_config).resolve()}",)
        if tuning_config is not None else ()
    )


def _generate_source(kernel_mlir, resolved, compiler, tuning_config, entry_name) -> GeneratedProgram:
    output = run_compiler(
        compiler,
        kernel_mlir,
        _options(resolved, tuning_config),
        resolved.compiler_role,
    )
    return GeneratedProgram(output.source, output.ir, output.metadata, output.directory, entry_name, resolved)


def compile_shared_gpu(
    definition: KernelDefinition[object, object],
    *,
    target: Target,
    compiler: str | Path | None = None,
    constexprs: dict[str, object] | None = None,
    tuning_config: str | Path | None = None,
) -> str:
    return compile_ir(
        definition, stage="shared", target=target, compiler=compiler,
        constexprs=constexprs, tuning_config=tuning_config,
    ).ir
