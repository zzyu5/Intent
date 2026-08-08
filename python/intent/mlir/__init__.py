from __future__ import annotations

from intent.definitions import KernelDefinition
from intent.frontend import lower_to_kernel_ir
from intent.realizer import PhysicalPlan

from .emitter import MlirEmitter
from .emitter import emit_mlir


def lower_to_mlir(
    definition: KernelDefinition[object, object],
    *,
    constexprs: dict[str, object] | None = None,
    plan: PhysicalPlan | None = None,
) -> str:
    return emit_mlir(
        lower_to_kernel_ir(definition, constexprs=constexprs),
        plan=plan,
    )


__all__ = ["MlirEmitter", "emit_mlir", "lower_to_mlir"]
