from __future__ import annotations

from pathlib import Path

from intent.api import KernelDefinition
from intent.frontend import lower_to_mlir
from intent.runtime import CompiledArtifact
from intent.targets.base import Target

from .toolchain import run_tool


def compile(
    definition: KernelDefinition[object, object],
    *,
    target: Target,
    realizer: str | Path,
    translator: str | Path,
    constexprs: dict[str, object] | None = None,
) -> CompiledArtifact:
    kernel_mlir = lower_to_mlir(definition, constexprs=constexprs)
    resolved = target.resolve()
    realized_mlir = run_tool(
        realizer,
        kernel_mlir,
        resolved.realizer_options,
        resolved.realizer_role,
    )
    source = run_tool(translator, realized_mlir, (), resolved.translator_role)
    return resolved.materialize(source, realized_mlir, definition.__name__)
