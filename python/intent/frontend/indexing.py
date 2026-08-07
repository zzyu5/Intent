from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from intent.ir import BufferType
from intent.ir import DynamicDim
from intent.ir import DomainType
from intent.ir import Effect
from intent.ir import EffectKind
from intent.ir import IndexRelation
from intent.ir import IndexTerm
from intent.ir import IndexTermKind
from intent.ir import LogicalIndexType
from intent.ir import OpCode
from intent.ir import RaggedType
from intent.ir import RegionType
from intent.ir import ResourceKind
from intent.ir import ScalarType
from intent.ir import StaticDim
from intent.ir import TensorType
from intent.ir import Value
from intent.ir import broadcast_shape
from intent.language import bool as intent_bool

from .model import Literal
from .model import ShapeDimension
from .model import ShapeValue
from .model import StaticTuple

if TYPE_CHECKING:
    from .lowering import FunctionLowerer


@dataclass(frozen=True, slots=True)
class LoweredIndex:
    relation: IndexRelation
    operands: tuple[Value, ...]
    result_shape: tuple[object, ...]


def validate_indexed_value(
    lowerer: FunctionLowerer,
    value: Value,
    indexed_shape: tuple[object, ...],
    node: ast.AST,
    *,
    expected_dtype: object | None = None,
) -> None:
    value_dtype, value_shape = lowerer.dtype_and_shape(value.type, node)
    if expected_dtype is not None and value_dtype != expected_dtype:
        lowerer.error(node, "stored/scattered value dtype does not match destination")
    try:
        broadcasted = broadcast_shape(value_shape, indexed_shape)
    except ValueError as error:
        lowerer.error(node, str(error))
    if tuple(broadcasted) != tuple(indexed_shape):
        lowerer.error(node, "stored/scattered value cannot broadcast to indexed shape")


def lower_subscript(
    lowerer: FunctionLowerer,
    node: ast.Subscript,
) -> object:
    source_expression = lowerer.lower_expression(node.value)
    if isinstance(source_expression, ShapeValue):
        return _subscript_shape(lowerer, source_expression, node)
    if isinstance(source_expression, StaticTuple):
        return _subscript_static_tuple(lowerer, source_expression, node)
    if not isinstance(source_expression, Value):
        lowerer.error(node, "subscript base must be tensor, buffer, ragged descriptor, or tuple")
    source = source_expression
    if isinstance(source.type, RaggedType):
        selector = lowerer.materialize(lowerer.lower_expression(node.slice), node.slice)
        operation = lowerer.emit(
            OpCode.RAGGED_MEMBER,
            lowerer.location(node),
            operands=(source, selector),
            result_types=(source.type.member,),
        )
        return operation.results[0]
    if not isinstance(source.type, (TensorType, BufferType)):
        lowerer.error(node, "only tensor/view/buffer values support positional indexing")
    lowered = lower_index(lowerer, source, node.slice, first_operand_position=1)
    result_type = lowerer.value_result_type(source.type.dtype, lowered.result_shape)
    if isinstance(source.type, BufferType):
        operation = lowerer.emit(
            OpCode.BUFFER_LOAD,
            lowerer.location(node),
            operands=(source, *lowered.operands),
            result_types=(result_type,),
            attributes={"index": lowered.relation},
            effects=(Effect(EffectKind.READ, ResourceKind.LOGICAL_BUFFER, source),),
        )
        return operation.results[0]
    if source in lowerer.view_kinds:
        lowerer.require_readable_view(source, node)
        operation = lowerer.emit(
            OpCode.VIEW_LOAD,
            lowerer.location(node),
            operands=(source, *lowered.operands),
            result_types=(result_type,),
            attributes={"index": lowered.relation},
            effects=(Effect(EffectKind.READ, ResourceKind.EXTERNAL_VIEW, source),),
        )
        return operation.results[0]
    valid = lowerer.emit_literal(True, node, ScalarType(intent_bool))
    fill = lowerer.emit_literal(False if source.type.dtype == intent_bool else 0, node, ScalarType(source.type.dtype))
    operands = (source, *lowered.operands, valid, fill)
    operation = lowerer.emit(
        OpCode.GATHER,
        lowerer.location(node),
        operands=operands,
        result_types=(result_type,),
        attributes={
            "index": lowered.relation,
            "valid_operand_index": len(operands) - 2,
            "fill_operand_index": len(operands) - 1,
        },
    )
    return operation.results[0]


def lower_index(
    lowerer: FunctionLowerer,
    source: Value,
    slice_node: ast.AST,
    *,
    first_operand_position: int,
) -> LoweredIndex:
    source_type = source.type
    if not isinstance(source_type, (TensorType, BufferType)):
        lowerer.error(slice_node, "index source must be tensor or buffer")
    raw_terms = list(slice_node.elts) if isinstance(slice_node, ast.Tuple) else [slice_node]
    raw_terms = [
        ast.Slice(lower=None, upper=None, step=None) if _is_full_slice_call(term) else term
        for term in raw_terms
    ]
    raw_terms = _expand_ellipsis(lowerer, raw_terms, len(source_type.shape), slice_node)
    consuming = sum(not _is_new_axis(term) for term in raw_terms)
    if consuming > len(source_type.shape):
        lowerer.error(slice_node, "too many positional indices for tensor rank")
    raw_terms.extend(
        ast.Slice(lower=None, upper=None, step=None)
        for _ in range(len(source_type.shape) - consuming)
    )

    operands: list[Value] = []
    terms: list[IndexTerm] = []
    result_shape: list[object] = []
    source_axis = 0
    for raw_term in raw_terms:
        if _is_new_axis(raw_term):
            terms.append(IndexTerm(IndexTermKind.NEW_AXIS))
            result_shape.append(StaticDim(1))
            continue
        source_dimension = source_type.shape[source_axis]
        if isinstance(raw_term, ast.Slice):
            if raw_term.lower is None and raw_term.upper is None and raw_term.step is None:
                terms.append(IndexTerm(IndexTermKind.FULL_SLICE))
                result_shape.append(source_dimension)
            else:
                positions: list[int | None] = []
                static_values: list[int | None] = []
                for component in (raw_term.lower, raw_term.upper, raw_term.step):
                    if component is None:
                        positions.append(None)
                        static_values.append(None)
                    else:
                        static = _static_integer(component)
                        if static is not None:
                            positions.append(None)
                            static_values.append(static)
                        else:
                            value = lowerer.materialize(lowerer.lower_expression(component), component)
                            positions.append(first_operand_position + len(operands))
                            static_values.append(None)
                            operands.append(value)
                terms.append(
                    IndexTerm(
                        IndexTermKind.SLICE,
                        tuple(positions),
                        tuple(static_values),
                    )
                )
                result_shape.append(DynamicDim(f"slice_{source.id}_{source_axis}"))
            source_axis += 1
            continue
        static = _static_integer(raw_term)
        if static is not None:
            terms.append(IndexTerm(IndexTermKind.STATIC_INDEX, static_values=(static,)))
            source_axis += 1
            continue
        value = lowerer.materialize(lowerer.lower_expression(raw_term), raw_term)
        position = first_operand_position + len(operands)
        operands.append(value)
        if isinstance(value.type, (DomainType, RegionType)):
            terms.append(IndexTerm(IndexTermKind.REGION_INDEX, (position,)))
            result_shape.extend(lowerer.dynamic_shape_for_region(value))
        elif isinstance(value.type, TensorType):
            terms.append(IndexTerm(IndexTermKind.VALUE_INDEX, (position,)))
            result_shape.extend(value.type.shape)
        elif isinstance(value.type, (ScalarType, LogicalIndexType)):
            terms.append(IndexTerm(IndexTermKind.VALUE_INDEX, (position,)))
        else:
            lowerer.error(raw_term, "index expression must be integer, index tensor, or region")
        source_axis += 1
    return LoweredIndex(IndexRelation(tuple(terms)), tuple(operands), tuple(result_shape))


def _expand_ellipsis(
    lowerer: FunctionLowerer,
    terms: list[ast.AST],
    rank: int,
    node: ast.AST,
) -> list[ast.AST]:
    positions = [index for index, term in enumerate(terms) if _is_ellipsis(term)]
    if len(positions) > 1:
        lowerer.error(node, "tensor index accepts at most one ellipsis")
    if not positions:
        return terms
    consuming_without = sum(
        not _is_new_axis(term) and not _is_ellipsis(term) for term in terms
    )
    fill = rank - consuming_without
    if fill < 0:
        lowerer.error(node, "ellipsis cannot fit tensor rank")
    position = positions[0]
    return (
        terms[:position]
        + [ast.Slice(lower=None, upper=None, step=None) for _ in range(fill)]
        + terms[position + 1 :]
    )


def _subscript_shape(
    lowerer: FunctionLowerer,
    shape: ShapeValue,
    node: ast.Subscript,
) -> object:
    index = _static_integer(node.slice)
    if index is None:
        lowerer.error(node, "shape tuple index must be a compile-time integer")
    if index < 0:
        index += len(shape.dimensions)
    if not 0 <= index < len(shape.dimensions):
        lowerer.error(node, "shape tuple index is outside rank")
    return shape.dimensions[index]


def _subscript_static_tuple(
    lowerer: FunctionLowerer,
    value: StaticTuple,
    node: ast.Subscript,
) -> object:
    index = _static_integer(node.slice)
    if index is None:
        lowerer.error(node, "tuple index must be a compile-time integer")
    if index < 0:
        index += len(value.elements)
    if not 0 <= index < len(value.elements):
        lowerer.error(node, "tuple index is outside value length")
    return value.elements[index]


def _static_integer(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    ):
        return -node.operand.value
    return None


def _is_new_axis(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_ellipsis(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is Ellipsis


def _is_full_slice_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "slice"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value is None
        and not node.keywords
    )
