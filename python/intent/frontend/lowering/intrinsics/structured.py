from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from intent.api import HelperDefinition
from intent.frontend.semantics import ValueType
from intent.frontend.semantics import BinaryOperator
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import RecordType
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import TensorType
from intent.frontend.mlir import MlirValue
from intent.frontend.semantics.types import dims_compatible
from intent.language.builtins import Intrinsic
from intent.language.dtypes import i16
from intent.language.dtypes import i32

from ..ast.expressions import compile_time_value
from ..ast.model import Literal
from ..ast.model import StaticTuple
from .common import bind_call
from .common import require_axes
from .common import require_dtype
from .common import require_static_bool
from .common import normalize_axes

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer


def lower_structured_intrinsic(
    lowerer: FunctionLowerer,
    name: str,
    node: ast.Call,
) -> object:
    if name in ("reduce", "reduce.max", "reduce.sum"):
        return _reduce(lowerer, name, node)
    if name == "arg_reduce.max":
        return _arg_reduce_max(lowerer, node)
    if name == "scan":
        return _scan(lowerer, node)
    if name == "contract":
        return _contract(lowerer, node)
    if name == "sparse_contract_2to4":
        return _sparse_contract_2to4(lowerer, node)
    return NotImplemented


def _reduce(lowerer: FunctionLowerer, name: str, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("value", "axis", "identity", "combine", "combine_operands", "acc_dtype"),
        required=("value", "axis", "identity"),
    )
    source = lowerer.read_value(
        lowerer.lower_expression(bound["value"]), bound["value"]
    )
    axes = require_axes(lowerer, bound["axis"])
    if isinstance(source.type, TensorType):
        axes = normalize_axes(lowerer, axes, source.type.rank, node)
        acc_dtype = (
            require_dtype(lowerer, bound["acc_dtype"])
            if "acc_dtype" in bound
            else source.type.dtype
        )
        identity_expression = lowerer.lower_expression(bound["identity"])
        identity = lowerer.materialize(
            identity_expression,
            bound["identity"],
            ScalarType(acc_dtype) if isinstance(identity_expression, Literal) else None,
        )
        if identity.type != ScalarType(acc_dtype):
            lowerer.error(node, "reduce identity must have accumulator dtype")
        if source.type.dtype != acc_dtype:
            source = lowerer.emit(
                OperationKind.CAST,
                lowerer.location(node),
                operands=(source,),
                result_types=(TensorType(acc_dtype, source.type.shape),),
            ).results[0]
        normalized = {axis % source.type.rank for axis in axes}
        result_shape = tuple(
            dimension
            for axis, dimension in enumerate(source.type.shape)
            if axis not in normalized
        )
        result_type: ValueType = lowerer.value_result_type(acc_dtype, result_shape)
        attributes: dict[str, object] = {"axes": axes, "acc_dtype": acc_dtype}
    elif isinstance(source.type, RecordType):
        if "acc_dtype" in bound:
            lowerer.error(node, "record reduce uses explicit identity field dtypes")
        identity = lowerer.materialize(
            lowerer.lower_expression(bound["identity"]), bound["identity"]
        )
        if not isinstance(identity.type, RecordType):
            lowerer.error(node, "record reduce identity must be I.record(...)")
        result_type = _record_result_type(lowerer, source.type, identity.type, axes, scan=False, node=node)
        attributes = {"axes": axes}
    else:
        lowerer.error(node, "I.reduce input must be tensor or tensor record")

    source_components, identity_components, component_names, result_components = (
        _flatten_accumulator(lowerer, source, identity, result_type, node)
    )
    captures = _combine_operands(lowerer, bound.get("combine_operands"), node)
    if name == "reduce.max":
        if captures:
            lowerer.error(node, "built-in reduction does not accept combine_operands=")
        combine: object = "maximum"
    elif name == "reduce.sum":
        if captures:
            lowerer.error(node, "built-in reduction does not accept combine_operands=")
        combine = "add"
    else:
        if "combine" not in bound:
            lowerer.error(node, "generic I.reduce requires combine=")
        combine = _combiner(
            lowerer,
            bound["combine"],
            identity.type,
            component_names,
            tuple(value.type for value in identity_components),
            captures,
            node,
        )
    attributes.update(
        {
            "combine": combine,
            "component_count": len(source_components),
            "capture_count": len(captures),
        }
    )
    operation = lowerer.emit(
        OperationKind.REDUCE,
        lowerer.location(node),
        operands=source_components + identity_components + captures,
        result_types=result_components,
        attributes=attributes,
    )
    return _rebuild_accumulator(
        lowerer, operation.results, result_type, component_names, node
    )


def _arg_reduce_max(lowerer: FunctionLowerer, node: ast.Call) -> StaticTuple:
    bound = bind_call(
        lowerer,
        node,
        ("value", "axis", "identity", "acc_dtype"),
        required=("value", "axis", "identity"),
    )
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    if not isinstance(source.type, TensorType):
        lowerer.error(node, "I.arg_reduce.max input must be a tensor")
    axes = normalize_axes(
        lowerer,
        require_axes(lowerer, bound["axis"]),
        source.type.rank,
        node,
    )
    if len(axes) != 1:
        lowerer.error(node, "I.arg_reduce.max requires exactly one axis")
    acc_dtype = (
        require_dtype(lowerer, bound["acc_dtype"])
        if "acc_dtype" in bound
        else source.type.dtype
    )
    identity_expression = lowerer.lower_expression(bound["identity"])
    identity = lowerer.materialize(
        identity_expression,
        bound["identity"],
        ScalarType(acc_dtype) if isinstance(identity_expression, Literal) else None,
    )
    if identity.type != ScalarType(acc_dtype):
        lowerer.error(node, "arg-reduce identity must have accumulator dtype")
    if source.type.dtype != acc_dtype:
        source = lowerer.emit(
            OperationKind.CAST,
            lowerer.location(node),
            operands=(source,),
            result_types=(TensorType(acc_dtype, source.type.shape),),
        ).results[0]
    result_shape = tuple(
        dimension
        for axis, dimension in enumerate(source.type.shape)
        if axis != axes[0]
    )
    value_type = lowerer.value_result_type(acc_dtype, result_shape)
    index_type = lowerer.value_result_type(i32, result_shape)
    indices = lowerer.emit(
        OperationKind.INDICES,
        lowerer.location(node),
        operands=(source,),
        result_types=(TensorType(i32, source.type.shape),),
        attributes={"axis": axes[0], "mode": "tensor_axis"},
    ).results[0]
    index_identity = lowerer.emit_literal(2147483647, node, ScalarType(i32))
    combine = lowerer.compiler.lower_argmax_combiner(
        ScalarType(acc_dtype), ScalarType(i32), lowerer.location(node)
    )
    operation = lowerer.emit(
        OperationKind.REDUCE,
        lowerer.location(node),
        operands=(source, indices, identity, index_identity),
        result_types=(value_type, index_type),
        attributes={
            "axes": axes,
            "combine": combine,
            "combine_builtin": "argmax_lowest",
            "component_count": 2,
            "capture_count": 0,
        },
    )
    return StaticTuple(operation.results)


def _scan(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        (
            "value",
            "axis",
            "identity",
            "combine",
            "combine_operands",
            "inclusive",
            "acc_dtype",
        ),
        required=("value", "axis", "identity", "combine", "inclusive"),
    )
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    axes = require_axes(lowerer, bound["axis"])
    if len(axes) != 1:
        lowerer.error(node, "I.scan requires exactly one axis")
    inclusive = require_static_bool(lowerer, bound["inclusive"])
    if isinstance(source.type, TensorType):
        axes = normalize_axes(lowerer, axes, source.type.rank, node)
        if len(axes) != 1:
            lowerer.error(node, "I.scan requires exactly one axis")
        acc_dtype = (
            require_dtype(lowerer, bound["acc_dtype"])
            if "acc_dtype" in bound
            else source.type.dtype
        )
        identity_expression = lowerer.lower_expression(bound["identity"])
        identity = lowerer.materialize(
            identity_expression,
            bound["identity"],
            ScalarType(acc_dtype) if isinstance(identity_expression, Literal) else None,
        )
        if identity.type != ScalarType(acc_dtype):
            lowerer.error(node, "scan identity must have accumulator dtype")
        if source.type.dtype != acc_dtype:
            source = lowerer.emit(
                OperationKind.CAST,
                lowerer.location(node),
                operands=(source,),
                result_types=(TensorType(acc_dtype, source.type.shape),),
            ).results[0]
        result_type: ValueType = TensorType(acc_dtype, source.type.shape)
        attributes: dict[str, object] = {
            "axis": axes[0],
            "inclusive": inclusive,
            "acc_dtype": acc_dtype,
        }
    elif isinstance(source.type, RecordType):
        if "acc_dtype" in bound:
            lowerer.error(node, "record scan uses explicit identity field dtypes")
        identity = lowerer.materialize(
            lowerer.lower_expression(bound["identity"]), bound["identity"]
        )
        if not isinstance(identity.type, RecordType):
            lowerer.error(node, "record scan identity must be I.record(...)")
        result_type = _record_result_type(
            lowerer, source.type, identity.type, axes, scan=True, node=node
        )
        attributes = {"axis": axes[0], "inclusive": inclusive}
    else:
        lowerer.error(node, "I.scan input must be tensor or tensor record")
    source_components, identity_components, component_names, result_components = (
        _flatten_accumulator(lowerer, source, identity, result_type, node)
    )
    captures = _combine_operands(lowerer, bound.get("combine_operands"), node)
    attributes["combine"] = _combiner(
        lowerer,
        bound["combine"],
        identity.type,
        component_names,
        tuple(value.type for value in identity_components),
        captures,
        node,
    )
    attributes["component_count"] = len(source_components)
    attributes["capture_count"] = len(captures)
    operation = lowerer.emit(
        OperationKind.SCAN,
        lowerer.location(node),
        operands=source_components + identity_components + captures,
        result_types=result_components,
        attributes=attributes,
    )
    return _rebuild_accumulator(
        lowerer, operation.results, result_type, component_names, node
    )


def _contract(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("lhs", "rhs", "reduce", "acc_dtype", "batch", "multiply", "combine"),
        required=("lhs", "rhs", "reduce", "acc_dtype"),
    )
    lhs = lowerer.read_value(lowerer.lower_expression(bound["lhs"]), bound["lhs"])
    rhs = lowerer.read_value(lowerer.lower_expression(bound["rhs"]), bound["rhs"])
    if not isinstance(lhs.type, TensorType) or not isinstance(rhs.type, TensorType):
        lowerer.error(node, "I.contract operands must be tensors")
    pairs = _reduction_pairs(lowerer, bound["reduce"])
    lhs_seen: set[int] = set()
    rhs_seen: set[int] = set()
    normalized_pairs: list[tuple[int, int]] = []
    for lhs_axis, rhs_axis in pairs:
        if not -lhs.type.rank <= lhs_axis < lhs.type.rank:
            lowerer.error(node, "contract lhs reduction axis is outside rank")
        if not -rhs.type.rank <= rhs_axis < rhs.type.rank:
            lowerer.error(node, "contract rhs reduction axis is outside rank")
        lhs_axis %= lhs.type.rank
        rhs_axis %= rhs.type.rank
        if lhs_axis in lhs_seen or rhs_axis in rhs_seen:
            lowerer.error(node, "contract reduction axes must be unique")
        if not dims_compatible(lhs.type.shape[lhs_axis], rhs.type.shape[rhs_axis]):
            lowerer.error(node, "contract paired dimensions are incompatible")
        lhs_seen.add(lhs_axis)
        rhs_seen.add(rhs_axis)
        normalized_pairs.append((lhs_axis, rhs_axis))
    pairs = tuple(normalized_pairs)
    batch_pairs = _axis_pairs(lowerer, bound["batch"], "batch") if "batch" in bound else ()
    lhs_batch_seen: set[int] = set()
    rhs_batch_seen: set[int] = set()
    normalized_batch_pairs: list[tuple[int, int]] = []
    for lhs_axis, rhs_axis in batch_pairs:
        if not -lhs.type.rank <= lhs_axis < lhs.type.rank:
            lowerer.error(node, "contract lhs batch axis is outside rank")
        if not -rhs.type.rank <= rhs_axis < rhs.type.rank:
            lowerer.error(node, "contract rhs batch axis is outside rank")
        lhs_axis %= lhs.type.rank
        rhs_axis %= rhs.type.rank
        if lhs_axis in lhs_seen or rhs_axis in rhs_seen:
            lowerer.error(node, "contract batch axes cannot also be reduction axes")
        if lhs_axis in lhs_batch_seen or rhs_axis in rhs_batch_seen:
            lowerer.error(node, "contract batch axes must be unique")
        if not dims_compatible(lhs.type.shape[lhs_axis], rhs.type.shape[rhs_axis]):
            lowerer.error(node, "contract paired batch dimensions are incompatible")
        lhs_batch_seen.add(lhs_axis)
        rhs_batch_seen.add(rhs_axis)
        normalized_batch_pairs.append((lhs_axis, rhs_axis))
    batch_pairs = tuple(normalized_batch_pairs)
    lhs_axes = {pair[0] for pair in pairs}
    rhs_axes = {pair[1] for pair in pairs}
    rhs_batch_axes = {pair[1] for pair in batch_pairs}
    result_shape = tuple(
        dimension for axis, dimension in enumerate(lhs.type.shape) if axis not in lhs_axes
    ) + tuple(
        dimension
        for axis, dimension in enumerate(rhs.type.shape)
        if axis not in rhs_axes and axis not in rhs_batch_axes
    )
    acc_dtype = require_dtype(lowerer, bound["acc_dtype"])
    multiply = "multiply"
    combine = "add"
    if "multiply" in bound:
        multiply = _callable_symbol(
            lowerer,
            bound["multiply"],
        )
    if "combine" in bound:
        combine = _callable_symbol(
            lowerer,
            bound["combine"],
        )
    if multiply != "multiply" or combine != "add":
        lowerer.error(
            node,
            "I.contract currently supports only multiply/add; spell a different "
            "semiring explicitly with pointwise operations and I.reduce",
        )
    operation = lowerer.emit(
        OperationKind.CONTRACT,
        lowerer.location(node),
        operands=(lhs, rhs),
        result_types=(TensorType(acc_dtype, result_shape),),
        attributes={
            "reduce": pairs,
            "batch": batch_pairs,
            "acc_dtype": acc_dtype,
            "multiply": multiply,
            "combine": combine,
        },
    )
    return operation.results[0]


def _sparse_contract_2to4(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("compressed_lhs", "metadata", "rhs", "acc_dtype"),
        required=("compressed_lhs", "metadata", "rhs", "acc_dtype"),
    )
    compressed = lowerer.read_value(
        lowerer.lower_expression(bound["compressed_lhs"]),
        bound["compressed_lhs"],
    )
    metadata = lowerer.read_value(
        lowerer.lower_expression(bound["metadata"]),
        bound["metadata"],
    )
    rhs = lowerer.read_value(
        lowerer.lower_expression(bound["rhs"]),
        bound["rhs"],
    )
    if not all(
        isinstance(value.type, TensorType)
        for value in (compressed, metadata, rhs)
    ):
        lowerer.error(node, "I.sparse_contract_2to4 operands must be tensors")
    if compressed.type.rank != 2 or metadata.type.rank != 2 or rhs.type.rank != 2:
        lowerer.error(node, "I.sparse_contract_2to4 requires rank-two operands")
    if compressed.type.dtype != rhs.type.dtype:
        lowerer.error(node, "2:4 sparse contraction data operands must share a dtype")
    if metadata.type.dtype != i16:
        lowerer.error(node, "2:4 sparse contraction metadata must be i16")
    result_shape = (compressed.type.shape[0], rhs.type.shape[1])
    acc_dtype = require_dtype(lowerer, bound["acc_dtype"])
    operation = lowerer.emit(
        OperationKind.SPARSE_CONTRACT,
        lowerer.location(node),
        operands=(compressed, metadata, rhs),
        result_types=(TensorType(acc_dtype, result_shape),),
        attributes={
            "format": "two_of_four",
            "compressed_axis": 1,
            "metadata_axis": 1,
            "rhs_reduction_axis": 0,
            "acc_dtype": acc_dtype,
        },
    )
    return operation.results[0]


def _callable_symbol(
    lowerer: FunctionLowerer,
    node: ast.AST,
) -> object:
    expression = lowerer.lower_expression(node)
    if isinstance(expression, Intrinsic):
        return expression.name
    lowerer.error(node, "combine/multiply must be an Intent intrinsic")


def _combiner(
    lowerer: FunctionLowerer,
    node: ast.AST,
    accumulator_type: ValueType,
    component_names: tuple[str, ...],
    component_types: tuple[ValueType, ...],
    captures: tuple[MlirValue, ...],
    call_node: ast.AST,
) -> object:
    expression = lowerer.lower_expression(node)
    if isinstance(expression, Intrinsic):
        if captures:
            lowerer.error(node, "built-in combiner does not accept combine_operands=")
        return expression.name
    if isinstance(expression, HelperDefinition):
        return lowerer.compiler.lower_combiner(
            lowerer,
            expression,
            accumulator_type,
            component_names,
            component_types,
            captures,
            lowerer.location(call_node),
        )
    lowerer.error(node, "combine= must be an Intent intrinsic or @intent.fn")


def _combine_operands(
    lowerer: FunctionLowerer,
    node: ast.AST | None,
    call_node: ast.AST,
) -> tuple[MlirValue, ...]:
    if node is None:
        return ()
    expression = lowerer.lower_expression(node)
    elements = expression.elements if isinstance(expression, StaticTuple) else (expression,)
    values = tuple(lowerer.materialize(element, node) for element in elements)
    if any(isinstance(value.type, (TensorType, RecordType)) for value in values):
        lowerer.error(
            call_node,
            "combine_operands must be explicit scalar SSA values",
        )
    return values


def _extract_record_components(
    lowerer: FunctionLowerer,
    value: MlirValue,
    node: ast.AST,
) -> tuple[MlirValue, ...]:
    if not isinstance(value.type, RecordType):
        return (value,)
    return tuple(
        lowerer.emit(
            OperationKind.EXTRACT,
            lowerer.location(node),
            operands=(value,),
            result_types=(field_type,),
            attributes={"key": field},
        ).results[0]
        for field, field_type in value.type.fields
    )


def _flatten_accumulator(
    lowerer: FunctionLowerer,
    source: MlirValue,
    identity: MlirValue,
    result_type: ValueType,
    node: ast.AST,
) -> tuple[
    tuple[MlirValue, ...],
    tuple[MlirValue, ...],
    tuple[str, ...],
    tuple[ValueType, ...],
]:
    source_components = _extract_record_components(lowerer, source, node)
    identity_components = _extract_record_components(lowerer, identity, node)
    if len(source_components) != len(identity_components):
        lowerer.error(node, "source and identity accumulator schemas must match")
    if isinstance(result_type, RecordType):
        component_names = tuple(name for name, _ in result_type.fields)
        result_components = tuple(field_type for _, field_type in result_type.fields)
    else:
        component_names = ("value",)
        result_components = (result_type,)
    if len(result_components) != len(source_components):
        lowerer.error(node, "result accumulator schema must match source and identity")
    return (
        source_components,
        identity_components,
        component_names,
        result_components,
    )


def _rebuild_accumulator(
    lowerer: FunctionLowerer,
    results: tuple[MlirValue, ...],
    result_type: ValueType,
    component_names: tuple[str, ...],
    node: ast.AST,
) -> MlirValue:
    if not isinstance(result_type, RecordType):
        if len(results) != 1:
            lowerer.error(node, "scalar accumulator requires one result")
        return results[0]
    return lowerer.emit(
        OperationKind.MAKE_RECORD,
        lowerer.location(node),
        operands=results,
        result_types=(result_type,),
        attributes={"fields": component_names},
    ).results[0]


def _reduction_pairs(lowerer: FunctionLowerer, node: ast.AST) -> tuple[tuple[int, int], ...]:
    pairs = _axis_pairs(lowerer, node, "reduction")
    if not pairs:
        lowerer.error(node, "contract requires at least one reduction pair")
    return pairs


def _axis_pairs(
    lowerer: FunctionLowerer,
    node: ast.AST,
    purpose: str,
) -> tuple[tuple[int, int], ...]:
    expression = lowerer.lower_expression(node)
    if not isinstance(expression, StaticTuple):
        lowerer.error(node, f"contract {purpose}= must be a tuple of axis pairs")
    pairs: list[tuple[int, int]] = []
    for element in expression.elements:
        if not isinstance(element, StaticTuple) or len(element.elements) != 2:
            lowerer.error(node, f"contract {purpose} entry must be an axis pair")
        pair: list[int] = []
        for axis in element.elements:
            known, value = compile_time_value(axis)
            if not known or isinstance(value, bool) or not isinstance(value, int):
                lowerer.error(
                    node,
                    f"contract {purpose} axes must be compile-time integers",
                )
            pair.append(value)
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


def _record_result_type(
    lowerer: FunctionLowerer,
    source: RecordType,
    identity: RecordType,
    axes: tuple[int, ...],
    *,
    scan: bool,
    node: ast.AST,
) -> RecordType:
    if tuple(name for name, _ in source.fields) != tuple(name for name, _ in identity.fields):
        lowerer.error(node, "record source/identity field names must match")
    tensor_fields = [field_type for _, field_type in source.fields]
    if not tensor_fields or any(not isinstance(field_type, TensorType) for field_type in tensor_fields):
        lowerer.error(node, "record reduction source fields must all be tensors")
    shape = tensor_fields[0].shape
    if any(field_type.shape != shape for field_type in tensor_fields):
        lowerer.error(node, "record reduction source fields must share shape")
    axes = normalize_axes(lowerer, axes, len(shape), node)
    normalized = set(axes)
    result_shape = shape if scan else tuple(
        dimension for axis, dimension in enumerate(shape) if axis not in normalized
    )
    fields: list[tuple[str, ValueType]] = []
    for (name, source_type), (_, identity_type) in zip(source.fields, identity.fields):
        if not isinstance(source_type, TensorType) or not isinstance(identity_type, ScalarType):
            lowerer.error(node, "record identity fields must be scalars")
        if source_type.dtype != identity_type.dtype:
            lowerer.error(node, "record accumulation dtype changes require explicit field casts")
        result_type: ValueType = identity_type
        if result_shape:
            result_type = TensorType(identity_type.dtype, result_shape)
        fields.append((name, result_type))
    return RecordType(tuple(fields))
