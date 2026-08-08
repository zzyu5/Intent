from __future__ import annotations

from pathlib import Path

from intent.api import KernelDefinition
from intent.frontend import lower_to_kernel_ir
from intent.mlir import emit_mlir
from intent.runtime import CompiledArtifact
from intent.runtime.triton import materialize_triton_artifact
from intent.targets import TritonTarget

from .toolchain import realize_mlir
from .toolchain import translate_mlir


def compile(
    definition: KernelDefinition[object, object],
    *,
    target: TritonTarget,
    realizer: str | Path,
    translator: str | Path,
    constexprs: dict[str, object] | None = None,
) -> CompiledArtifact:
    if not isinstance(target, TritonTarget):
        raise NotImplementedError("only TritonTarget is implemented")
    kernel_ir = lower_to_kernel_ir(definition, constexprs=constexprs)
    kernel_mlir = emit_mlir(kernel_ir)
    realized_mlir = realize_mlir(kernel_mlir, realizer, target.resolve())
    source = translate_mlir(realized_mlir, translator)
    return materialize_triton_artifact(source, realized_mlir, definition.__name__)
