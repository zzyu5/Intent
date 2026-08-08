from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from intent.api import HelperDefinition
from intent.frontend.semantics import ValueType
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import RecordType
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import TensorType
from intent.frontend.mlir import MlirValue
from intent.frontend.semantics.types import dims_compatible
from intent.language.builtins import Intrinsic

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
    if name == "scan":
        return _scan(lowerer, node)
    if name == "contract":
        return _contract(lowerer, node)
    return NotImplemented


def _reduce(lowerer: FunctionLowerer, name: str, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("value", "axis", "identity", "combine", "acc_dtype"),
        required=("value", "axis", "identity"),
    )
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
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
        normalized = {axis % source.type.rank for axis in axes}
        result_shape = tuple(
            dimension
            for axis, dimension in enumerate(source.type.shape)
            if axis not in normalized
        )
        result_type: ValueType = lowerer.value_result_type(acc_dtype, result_shape)
        accumulator_type: ValueType = ScalarType(acc_dtype)
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
        accumulator_type = identity.type
        attributes = {"axes": axes}
    else:
        lowerer.error(node, "I.reduce input must be tensor or tensor record")

    if name == "reduce.max":
        combine: object = "maximum"
    elif name == "reduce.sum":
        combine = "add"
    else:
        if "combine" not in bound:
            lowerer.error(node, "generic I.reduce requires combine=")
        combine = _callable_symbol(
            lowerer,
            bound["combine"],
            (accumulator_type, accumulator_type),
            (accumulator_type,),
        )
    attributes["combine"] = combine
    operation = lowerer.emit(
        OperationKind.REDUCE,
        lowerer.location(node),
        operands=(source, identity),
        result_types=(result_type,),
        attributes=attributes,
    )
    return operation.results[0]


def _scan(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("value", "axis", "identity", "combine", "inclusive", "acc_dtype"),
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
        accumulator_type: ValueType = ScalarType(acc_dtype)
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
        accumulator_type = identity.type
        result_type = _record_result_type(
            lowerer, source.type, identity.type, axes, scan=True, node=node
        )
        attributes = {"axis": axes[0], "inclusive": inclusive}
    else:
        lowerer.error(node, "I.scan input must be tensor or tensor record")
    attributes["combine"] = _callable_symbol(
        lowerer,
        bound["combine"],
        (accumulator_type, accumulator_type),
        (accumulator_type,),
    )
    operation = lowerer.emit(
        OperationKind.SCAN,
        lowerer.location(node),
        operands=(source, identity),
        result_types=(result_type,),
        attributes=attributes,
    )
    return operation.results[0]


def _contract(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("lhs", "rhs", "reduce", "acc_dtype", "multiply", "combine"),
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
    lhs_axes = {pair[0] for pair in pairs}
    rhs_axes = {pair[1] for pair in pairs}
    result_shape = tuple(
        dimension for axis, dimension in enumerate(lhs.type.shape) if axis not in lhs_axes
    ) + tuple(
        dimension for axis, dimension in enumerate(rhs.type.shape) if axis not in rhs_axes
    )
    acc_dtype = require_dtype(lowerer, bound["acc_dtype"])
    accumulator_type = ScalarType(acc_dtype)
    multiply = "multiply"
    combine = "add"
    if "multiply" in bound:
        multiply = _callable_symbol(
            lowerer,
            bound["multiply"],
            (ScalarType(lhs.type.dtype), ScalarType(rhs.type.dtype)),
            (accumulator_type,),
        )
    if "combine" in bound:
        combine = _callable_symbol(
            lowerer,
            bound["combine"],
            (accumulator_type, accumulator_type),
            (accumulator_type,),
        )
    operation = lowerer.emit(
        OperationKind.CONTRACT,
        lowerer.location(node),
        operands=(lhs, rhs),
        result_types=(TensorType(acc_dtype, result_shape),),
        attributes={
            "reduce": pairs,
            "acc_dtype": acc_dtype,
            "multiply": multiply,
            "combine": combine,
        },
    )
    return operation.results[0]


def _callable_symbol(
    lowerer: FunctionLowerer,
    node: ast.AST,
    operand_types: tuple[ValueType, ...],
    result_types: tuple[ValueType, ...],
) -> object:
    expression = lowerer.lower_expression(node)
    if isinstance(expression, Intrinsic):
        return expression.name
    if isinstance(expression, HelperDefinition):
        helper = lowerer.compiler.lower_helper(
            expression,
            operand_types,
            lowerer.location(node),
        )
        if len(helper.result_types) != len(result_types) or any(
            actual != expected for actual, expected in zip(helper.result_types, result_types)
        ):
            lowerer.error(node, "combiner helper result schema is incompatible")
        return helper.name
    lowerer.error(node, "combine/multiply must be an Intent intrinsic or @intent.fn")


def _reduction_pairs(lowerer: FunctionLowerer, node: ast.AST) -> tuple[tuple[int, int], ...]:
    expression = lowerer.lower_expression(node)
    if not isinstance(expression, StaticTuple):
        lowerer.error(node, "contract reduce= must be a tuple of axis pairs")
    pairs: list[tuple[int, int]] = []
    for element in expression.elements:
        if not isinstance(element, StaticTuple) or len(element.elements) != 2:
            lowerer.error(node, "contract reduction entry must be an axis pair")
        pair: list[int] = []
        for axis in element.elements:
            known, value = compile_time_value(axis)
            if not known or isinstance(value, bool) or not isinstance(value, int):
                lowerer.error(node, "contract reduction axes must be compile-time integers")
            pair.append(value)
        pairs.append((pair[0], pair[1]))
    if not pairs:
        lowerer.error(node, "contract requires at least one reduction pair")
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
