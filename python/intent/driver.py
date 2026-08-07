from __future__ import annotations

from intent.backend.artifact import CompiledArtifact
from intent.backend.triton.emitter import emit_triton
from intent.backend.triton.target import TritonTarget
from intent.definitions import KernelDefinition
from intent.frontend import lower_to_kernel_ir
from intent.realizer.triton import realize_triton


def compile(
    definition: KernelDefinition[object, object],
    *,
    target: TritonTarget,
    constexprs: dict[str, object] | None = None,
    options: dict[str, object] | None = None,
) -> CompiledArtifact:
    if not isinstance(target, TritonTarget):
        raise NotImplementedError("only TritonTarget is implemented")
    if options:
        raise NotImplementedError("initial Triton backend exposes no compile policies")
    module = lower_to_kernel_ir(definition, constexprs=constexprs)
    plan = realize_triton(module, target)
    return emit_triton(module, plan)
