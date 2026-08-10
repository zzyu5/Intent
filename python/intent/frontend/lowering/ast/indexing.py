from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from intent.frontend.semantics import BufferType
from intent.frontend.semantics import DynamicDim
from intent.frontend.semantics import DomainType
from intent.frontend.semantics import Effect
from intent.frontend.semantics import EffectKind
from intent.frontend.semantics import IndexRelation
from intent.frontend.semantics import IndexTerm
from intent.frontend.semantics import IndexTermKind
from intent.frontend.semantics import LogicalIndexType
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import RaggedType
from intent.frontend.semantics import RegionType
from intent.frontend.semantics import ResourceKind
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import StaticDim
from intent.frontend.semantics import TensorType
from intent.frontend.mlir import MlirValue
from intent.frontend.semantics import broadcast_shape
from intent.frontend.semantics.types import is_integer
from intent.language import bool as intent_bool
from intent.language import DTypeCategory

from .model import ShapeValue
from .model import StaticTuple

if TYPE_CHECKING:
    from .context import FunctionLowerer


@dataclass(frozen=True, slots=True)
class LoweredIndex:
    relation: IndexRelation
    operands: tuple[MlirValue, ...]
    result_shape: tuple[object, ...]


def validate_indexed_value(
    lowerer: FunctionLowerer,
    value: MlirValue,
    indexed_shape: tuple[object, ...],
    node: ast.AST,
    *,
    expected_dtype: object | None = None,
) -> MlirValue:
    value_dtype, value_shape = lowerer.dtype_and_shape(value.type, node)
    if expected_dtype is not None and value_dtype != expected_dtype:
        lowerer.error(node, "stored/scattered value dtype does not match destination")
    try:
        broadcasted = broadcast_shape(value_shape, indexed_shape)
    except ValueError as error:
        lowerer.error(node, str(error))
    if tuple(broadcasted) != tuple(indexed_shape):
        lowerer.error(node, "stored/scattered value cannot broadcast to indexed shape")
    return lowerer.broadcast_value(value, tuple(indexed_shape), node)


def lower_subscript(
    lowerer: FunctionLowerer,
    node: ast.Subscript,
) -> object:
    source_expression = lowerer.lower_expression(node.value)
    if isinstance(source_expression, ShapeValue):
        return _subscript_shape(lowerer, source_expression, node)
    if isinstance(source_expression, StaticTuple):
        return _subscript_static_tuple(lowerer, source_expression, node)
    if not isinstance(source_expression, MlirValue):
        lowerer.error(node, "subscript base must be tensor, buffer, ragged descriptor, or tuple")
    source = source_expression
    if isinstance(source.type, RaggedType):
        selector = lowerer.materialize(lowerer.lower_expression(node.slice), node.slice)
        if not is_integer(selector.type):
            lowerer.error(node.slice, "ragged member selector must be integer/index")
        operation = lowerer.emit(
            OperationKind.RAGGED_MEMBER,
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
            OperationKind.BUFFER_LOAD,
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
            OperationKind.VIEW_LOAD,
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
        OperationKind.GATHER,
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
    source: MlirValue,
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

    operands: list[MlirValue] = []
    terms: list[IndexTerm] = []
    result_shape: list[object] = []
    advanced_shape: tuple[object, ...] = ()
    advanced_position: int | None = None
    advanced_indices_contiguous = True
    saw_non_advanced_after_advanced = False
    source_axis = 0
    for raw_term in raw_terms:
        if _is_new_axis(raw_term):
            if advanced_position is not None:
                saw_non_advanced_after_advanced = True
            terms.append(IndexTerm(IndexTermKind.NEW_AXIS))
            result_shape.append(StaticDim(1))
            continue
        source_dimension = source_type.shape[source_axis]
        if isinstance(raw_term, ast.Slice):
            if advanced_position is not None:
                saw_non_advanced_after_advanced = True
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
                            if not is_integer(value.type):
                                lowerer.error(component, "dynamic slice bound must be scalar integer/index")
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
            if advanced_position is not None:
                saw_non_advanced_after_advanced = True
            if isinstance(source_dimension, StaticDim) and not (
                -source_dimension.value <= static < source_dimension.value
            ):
                lowerer.error(raw_term, "static tensor index is outside the source dimension")
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
            if value.type.dtype.category not in (
                DTypeCategory.SIGNED_INTEGER,
                DTypeCategory.UNSIGNED_INTEGER,
                DTypeCategory.INDEX,
            ):
                lowerer.error(raw_term, "tensor index must have integer/index dtype")
            if saw_non_advanced_after_advanced:
                advanced_indices_contiguous = False
            if not advanced_indices_contiguous:
                lowerer.error(
                    raw_term,
                    "multiple tensor indices must be adjacent in one positional index relation",
                )
            terms.append(IndexTerm(IndexTermKind.VALUE_INDEX, (position,)))
            try:
                merged_shape = broadcast_shape(advanced_shape, value.type.shape)
            except ValueError as error:
                lowerer.error(raw_term, str(error))
            if advanced_position is None:
                advanced_position = len(result_shape)
            else:
                del result_shape[
                    advanced_position : advanced_position + len(advanced_shape)
                ]
            result_shape[advanced_position:advanced_position] = merged_shape
            advanced_shape = tuple(merged_shape)
        elif isinstance(value.type, (ScalarType, LogicalIndexType)) and is_integer(value.type):
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
