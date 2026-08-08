from __future__ import annotations

from pathlib import Path

from intent.api import KernelDefinition
from intent.frontend import lower_to_mlir
from intent.runtime import CompiledArtifact
from intent.targets.base import Target

from .toolchain import run_compiler


def compile(
    definition: KernelDefinition[object, object],
    *,
    target: Target,
    compiler: str | Path,
    constexprs: dict[str, object] | None = None,
) -> CompiledArtifact:
    kernel_mlir = lower_to_mlir(definition, constexprs=constexprs)
    resolved = target.resolve()
    source, realized_mlir = run_compiler(
        compiler,
        kernel_mlir,
        resolved.compiler_options,
        resolved.compiler_role,
    )
    return resolved.materialize(source, realized_mlir, definition.__name__)
