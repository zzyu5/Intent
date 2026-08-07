from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TypeAlias

from intent.ir import AutoExtent
from intent.ir import DimExpr
from intent.ir import OpCode
from intent.ir import Value


@dataclass(frozen=True, slots=True)
class Literal:
    value: bool | int | float


@dataclass(frozen=True, slots=True)
class ConstexprBinding:
    python_value: object
    ir_value: Value


@dataclass(frozen=True, slots=True)
class ShapeDimension:
    dimension: DimExpr
    source: Value
    axis: int


@dataclass(frozen=True, slots=True)
class ShapeValue:
    dimensions: tuple[ShapeDimension, ...]


@dataclass(frozen=True, slots=True)
class IterationSpec:
    opcode: OpCode
    source: Value


@dataclass(slots=True)
class StreamSpec:
    axis: Value
    initial_state: tuple[Value, ...]
    extent: AutoExtent | Value
    ast_node: ast.AST
    results: tuple[Value, ...] | None = None


@dataclass(frozen=True, slots=True)
class StaticTuple:
    elements: tuple[Expression, ...]


Expression: TypeAlias = (
    Value
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
    opcode: OpCode
    carried_names: tuple[str, ...]
    stream: StreamSpec | None = None
    pending_stream_state: tuple[Value, ...] | None = None
