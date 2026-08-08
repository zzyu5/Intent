from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TypeAlias

from intent.frontend.semantics import AutoExtent
from intent.frontend.semantics import DimExpr
from intent.frontend.semantics import OperationKind
from intent.frontend.mlir import MlirValue


@dataclass(frozen=True, slots=True)
class Literal:
    value: bool | int | float


@dataclass(frozen=True, slots=True)
class ConstexprBinding:
    python_value: object
    ir_value: MlirValue


@dataclass(frozen=True, slots=True)
class ShapeDimension:
    dimension: DimExpr
    source: MlirValue
    axis: int


@dataclass(frozen=True, slots=True)
class ShapeValue:
    dimensions: tuple[ShapeDimension, ...]


@dataclass(frozen=True, slots=True)
class IterationSpec:
    opcode: OperationKind
    source: MlirValue


@dataclass(slots=True)
class StreamSpec:
    axis: MlirValue
    initial_state: tuple[MlirValue, ...]
    extent: AutoExtent | MlirValue
    ast_node: ast.AST
    results: tuple[MlirValue, ...] | None = None


@dataclass(frozen=True, slots=True)
class StaticTuple:
    elements: tuple[Expression, ...]


Expression: TypeAlias = (
    MlirValue
    | Literal
    | ConstexprBinding
    | ShapeDimension
    | ShapeValue
    | IterationSpec
    | StreamSpec
    | StaticTuple
    | AutoExtent
    | object
)


@dataclass(slots=True)
class LoopContext:
    opcode: OperationKind
    carried_names: tuple[str, ...]
    stream: StreamSpec | None = None
    pending_stream_state: tuple[MlirValue, ...] | None = None


__all__ = [
    "ConstexprBinding",
    "Expression",
    "IterationSpec",
    "Literal",
    "LoopContext",
    "ShapeDimension",
    "ShapeValue",
    "StaticTuple",
    "StreamSpec",
]
