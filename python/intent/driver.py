from __future__ import annotations

from pathlib import Path

from intent.backend.artifact import CompiledArtifact
from intent.backend.triton.runtime import materialize_triton_artifact
from intent.backend.triton.target import TritonTarget
from intent.backend.triton.translator import translate_mlir
from intent.definitions import KernelDefinition
from intent.frontend import lower_to_kernel_ir
from intent.mlir import emit_mlir
from intent.realizer.triton import realize_stable_softmax


def compile(
    definition: KernelDefinition[object, object],
    *,
    target: TritonTarget,
    translator: str | Path,
    constexprs: dict[str, object] | None = None,
) -> CompiledArtifact:
    if not isinstance(target, TritonTarget):
        raise NotImplementedError("only TritonTarget is implemented")
    module = lower_to_kernel_ir(definition, constexprs=constexprs)
    plan = realize_stable_softmax(module, target)
    module_text = emit_mlir(module, plan=plan)
    source = translate_mlir(module_text, translator)
    return materialize_triton_artifact(source, module_text, plan)
