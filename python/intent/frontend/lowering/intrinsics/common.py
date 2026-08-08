from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from intent.frontend.semantics import DimExpr
from intent.frontend.semantics import DynamicDim
from intent.frontend.semantics import RegionType
from intent.frontend.semantics import StaticDim
from intent.frontend.semantics import SymbolDim
from intent.frontend.mlir import MlirValue
from intent.language import DType

from ..ast.expressions import compile_time_value
from ..ast.model import Literal
from ..ast.model import ShapeDimension
from ..ast.model import StaticTuple

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer


def bind_call(
    lowerer: FunctionLowerer,
    node: ast.Call,
    positional: tuple[str, ...],
    *,
    required: tuple[str, ...] = (),
) -> dict[str, ast.AST]:
    if len(node.args) > len(positional):
        lowerer.error(node, "too many positional intrinsic arguments")
    bound = {name: value for name, value in zip(positional, node.args)}
    for keyword in node.keywords:
        if keyword.arg is None:
            lowerer.error(keyword, "**kwargs expansion is not supported in Intent source")
        if keyword.arg not in positional:
            lowerer.error(keyword, f"unknown intrinsic argument {keyword.arg!r}")
        if keyword.arg in bound:
            lowerer.error(keyword, f"duplicate intrinsic argument {keyword.arg!r}")
        bound[keyword.arg] = keyword.value
    missing = [name for name in required if name not in bound]
    if missing:
        lowerer.error(node, f"missing intrinsic argument {missing[0]!r}")
    return bound


def require_dtype(lowerer: FunctionLowerer, node: ast.AST) -> DType:
    value = lowerer.lower_expression(node)
    if not isinstance(value, DType):
        lowerer.error(node, "dtype argument must be an Intent dtype such as I.f32")
    return value


def require_static_bool(lowerer: FunctionLowerer, node: ast.AST) -> bool:
    expression = lowerer.lower_expression(node)
    known, value = compile_time_value(expression)
    if not known or not isinstance(value, bool):
        lowerer.error(node, "argument must be a compile-time bool")
    return value


def require_static_int(lowerer: FunctionLowerer, node: ast.AST) -> int:
    expression = lowerer.lower_expression(node)
    known, value = compile_time_value(expression)
    if not known or isinstance(value, bool) or not isinstance(value, int):
        lowerer.error(node, "argument must be a compile-time integer")
    return value


def require_axes(lowerer: FunctionLowerer, node: ast.AST) -> tuple[int, ...]:
    expression = lowerer.lower_expression(node)
    if isinstance(expression, Literal):
        if isinstance(expression.value, int) and not isinstance(expression.value, bool):
            return (expression.value,)
    if isinstance(expression, StaticTuple):
        axes: list[int] = []
        for element in expression.elements:
            known, value = compile_time_value(element)
            if not known or isinstance(value, bool) or not isinstance(value, int):
                lowerer.error(node, "axes must contain compile-time integers")
            axes.append(value)
        if axes:
            return tuple(axes)
    lowerer.error(node, "axis must be an integer or non-empty integer tuple")


def normalize_axes(
    lowerer: FunctionLowerer,
    axes: tuple[int, ...],
    rank: int,
    node: ast.AST,
) -> tuple[int, ...]:
    normalized: list[int] = []
    for axis in axes:
        if not -rank <= axis < rank:
            lowerer.error(node, f"axis {axis} is outside rank {rank}")
        value = axis % rank
        if value in normalized:
            lowerer.error(node, "axes must be unique")
        normalized.append(value)
    return tuple(normalized)


def lower_shape(lowerer: FunctionLowerer, node: ast.AST) -> tuple[DimExpr, ...]:
    expression = lowerer.lower_expression(node)
    elements = expression.elements if isinstance(expression, StaticTuple) else (expression,)
    dimensions: list[DimExpr] = []
    for element in elements:
        if isinstance(element, ShapeDimension):
            dimensions.append(element.dimension)
        elif isinstance(element, MlirValue) and isinstance(element.type, RegionType):
            dimensions.extend(lowerer.dynamic_shape_for_region(element))
        elif isinstance(element, Literal) and isinstance(element.value, int) and not isinstance(
            element.value, bool
        ):
            if element.value < 0:
                lowerer.error(node, "shape dimensions must be non-negative")
            dimensions.append(StaticDim(element.value))
        elif isinstance(element, str):
            dimensions.append(SymbolDim(element))
        elif element is Ellipsis:
            dimensions.append(DynamicDim("reshape_inferred"))
        else:
            lowerer.error(node, "shape elements must be static, symbolic, shape-derived, or region")
    return tuple(dimensions)


def optional_node(bound: dict[str, ast.AST], name: str) -> ast.AST | None:
    return bound.get(name)
