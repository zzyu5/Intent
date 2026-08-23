from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from intent.frontend.semantics import BinaryOperator
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import RecordType
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import TensorType
from intent.frontend.semantics import UnaryOperator
from intent.frontend.mlir import MlirValue
from intent.frontend.semantics import broadcast_shape
from intent.language import DTypeCategory
from intent.language import bool as intent_bool

from ..ast.model import Literal
from .common import bind_call
from .common import lower_shape
from .common import normalize_axes
from .common import require_axes
from .common import require_dtype

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer


def lower_tensor_intrinsic(
    lowerer: FunctionLowerer,
    name: str,
    node: ast.Call,
) -> object:
    handlers = {
        "reshape": _reshape,
        "transpose": _transpose,
        "full": _full,
        "zeros": _zeros,
        "record": _record,
        "cast": _cast,
        "bitcast": _bitcast,
        "mask": _mask,
        "exp": lambda context, call: _unary(context, call, UnaryOperator.EXP),
        "exp2": lambda context, call: _unary(context, call, UnaryOperator.EXP2),
        "log": lambda context, call: _unary(context, call, UnaryOperator.LOG),
        "sin": lambda context, call: _unary(context, call, UnaryOperator.SIN),
        "cos": lambda context, call: _unary(context, call, UnaryOperator.COS),
        "floor": lambda context, call: _unary(context, call, UnaryOperator.FLOOR),
        "rsqrt": lambda context, call: _unary(context, call, UnaryOperator.RSQRT),
        "sigmoid": lambda context, call: _unary(context, call, UnaryOperator.SIGMOID),
        "tanh": lambda context, call: _unary(context, call, UnaryOperator.TANH),
        "abs": lambda context, call: _unary(context, call, UnaryOperator.ABS),
        "maximum": lambda context, call: _binary(context, call, BinaryOperator.MAXIMUM),
        "minimum": lambda context, call: _binary(context, call, BinaryOperator.MINIMUM),
        "add": lambda context, call: _binary(context, call, BinaryOperator.ADD),
        "any": lambda context, call: _logical_reduce(context, call, any_value=True),
        "all": lambda context, call: _logical_reduce(context, call, any_value=False),
    }
    handler = handlers.get(name)
    if handler is None:
        return NotImplemented
    return handler(lowerer, node)


def _reshape(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("value", "shape"), required=("value", "shape"))
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    if not isinstance(source.type, TensorType):
        lowerer.error(node, "I.reshape input must be a tensor")
    shape = lower_shape(lowerer, bound["shape"])
    operation = lowerer.emit(
        OperationKind.RESHAPE,
        lowerer.location(node),
        operands=(source,),
        result_types=(TensorType(source.type.dtype, shape),),
    )
    return operation.results[0]


def _transpose(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("value", "permutation"), required=("value",))
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    if not isinstance(source.type, TensorType):
        lowerer.error(node, "I.transpose input must be a tensor")
    if "permutation" in bound:
        permutation = require_axes(lowerer, bound["permutation"])
    else:
        permutation = tuple(reversed(range(source.type.rank)))
    if sorted(permutation) != list(range(source.type.rank)):
        lowerer.error(node, "transpose permutation must cover every axis")
    result_shape = tuple(source.type.shape[axis] for axis in permutation)
    operation = lowerer.emit(
        OperationKind.TRANSPOSE,
        lowerer.location(node),
        operands=(source,),
        result_types=(TensorType(source.type.dtype, result_shape),),
        attributes={"permutation": permutation},
    )
    return operation.results[0]


def _full(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("shape", "fill", "dtype"),
        required=("shape", "fill", "dtype"),
    )
    dtype = require_dtype(lowerer, bound["dtype"])
    shape = lower_shape(lowerer, bound["shape"])
    fill = lowerer.materialize(
        lowerer.lower_expression(bound["fill"]),
        bound["fill"],
        ScalarType(dtype),
    )
    operation = lowerer.emit(
        OperationKind.FULL,
        lowerer.location(node),
        operands=(fill,),
        result_types=(TensorType(dtype, shape),),
    )
    return operation.results[0]


def _zeros(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("shape", "dtype"), required=("shape", "dtype"))
    dtype = require_dtype(lowerer, bound["dtype"])
    shape = lower_shape(lowerer, bound["shape"])
    operation = lowerer.emit(
        OperationKind.ZEROS,
        lowerer.location(node),
        result_types=(TensorType(dtype, shape),),
    )
    return operation.results[0]


def _record(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    if node.args or not node.keywords:
        lowerer.error(node, "I.record requires one or more named fields")
    names: list[str] = []
    values: list[MlirValue] = []
    for keyword in node.keywords:
        if keyword.arg is None:
            lowerer.error(keyword, "record does not support **field expansion")
        names.append(keyword.arg)
        values.append(lowerer.materialize(lowerer.lower_expression(keyword.value), keyword.value))
    result_type = RecordType(tuple((name, value.type) for name, value in zip(names, values)))
    operation = lowerer.emit(
        OperationKind.MAKE_RECORD,
        lowerer.location(node),
        operands=tuple(values),
        result_types=(result_type,),
        attributes={"fields": tuple(names)},
    )
    return operation.results[0]


def _cast(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("value", "dtype"), required=("value", "dtype"))
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    dtype = require_dtype(lowerer, bound["dtype"])
    source_dtype, shape = lowerer.dtype_and_shape(source.type, node)
    attributes = {}
    if source_dtype.category in (DTypeCategory.FLOAT, DTypeCategory.BFLOAT) and (
        dtype.category
        in (DTypeCategory.SIGNED_INTEGER, DTypeCategory.UNSIGNED_INTEGER)
    ):
        attributes["rounding"] = "toward_zero"
    operation = lowerer.emit(
        OperationKind.CAST,
        lowerer.location(node),
        operands=(source,),
        result_types=(lowerer.value_result_type(dtype, shape),),
        attributes=attributes,
    )
    return operation.results[0]


def _bitcast(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("value", "dtype"), required=("value", "dtype"))
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    dtype = require_dtype(lowerer, bound["dtype"])
    source_dtype, shape = lowerer.dtype_and_shape(source.type, node)
    if source_dtype.category in (DTypeCategory.BOOL, DTypeCategory.INDEX) or (
        dtype.category in (DTypeCategory.BOOL, DTypeCategory.INDEX)
    ):
        lowerer.error(node, "I.bitcast does not accept bool or index dtypes")
    if source_dtype.bits is None or source_dtype.bits != dtype.bits:
        lowerer.error(node, "I.bitcast requires equal source and result bit widths")
    operation = lowerer.emit(
        OperationKind.BITCAST,
        lowerer.location(node),
        operands=(source,),
        result_types=(lowerer.value_result_type(dtype, shape),),
    )
    return operation.results[0]


def _mask(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("value", "valid", "fill"),
        required=("value", "valid", "fill"),
    )
    value_expression = lowerer.lower_expression(bound["value"])
    value = lowerer.read_value(value_expression, bound["value"])
    dtype, _ = lowerer.dtype_and_shape(value.type, node)
    predicate = lowerer.materialize(lowerer.lower_expression(bound["valid"]), bound["valid"])
    fill_expression = lowerer.lower_expression(bound["fill"])
    fill = lowerer.materialize(
        fill_expression,
        bound["fill"],
        ScalarType(dtype) if isinstance(fill_expression, Literal) else None,
    )
    _, value_shape = lowerer.dtype_and_shape(value.type, node)
    fill_dtype, fill_shape = lowerer.dtype_and_shape(fill.type, node)
    if fill_dtype != dtype:
        lowerer.error(node, "mask fill dtype must match value dtype")
    predicate_dtype, predicate_shape = lowerer.dtype_and_shape(predicate.type, node)
    if predicate_dtype != intent_bool:
        lowerer.error(node, "mask valid predicate must have bool dtype")
    try:
        result_shape = broadcast_shape(broadcast_shape(value_shape, fill_shape), predicate_shape)
    except ValueError as error:
        lowerer.error(node, str(error))
    result_type = lowerer.value_result_type(dtype, result_shape)
    value = lowerer.broadcast_value(value, result_shape, node)
    predicate = lowerer.broadcast_value(predicate, result_shape, node)
    fill = lowerer.broadcast_value(fill, result_shape, node)
    operation = lowerer.emit(
        OperationKind.MASK,
        lowerer.location(node),
        operands=(value, predicate, fill),
        result_types=(result_type,),
    )
    return operation.results[0]


def _unary(
    lowerer: FunctionLowerer,
    node: ast.Call,
    operator: UnaryOperator,
) -> MlirValue:
    bound = bind_call(lowerer, node, ("value",), required=("value",))
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    operation = lowerer.emit(
        OperationKind.UNARY,
        lowerer.location(node),
        operands=(source,),
        result_types=(source.type,),
        attributes={"operator": operator},
    )
    return operation.results[0]


def _binary(
    lowerer: FunctionLowerer,
    node: ast.Call,
    operator: BinaryOperator,
) -> MlirValue:
    bound = bind_call(lowerer, node, ("lhs", "rhs"), required=("lhs", "rhs"))
    lhs_expression = lowerer.lower_expression(bound["lhs"])
    rhs_expression = lowerer.lower_expression(bound["rhs"])
    lhs, rhs = lowerer.coerce_pair(lhs_expression, rhs_expression, node)
    result_type = lowerer.broadcast_result_type(lhs.type, rhs.type, node)
    _, result_shape = lowerer.dtype_and_shape(result_type, node)
    lhs = lowerer.broadcast_value(lhs, result_shape, node)
    rhs = lowerer.broadcast_value(rhs, result_shape, node)
    operation = lowerer.emit(
        OperationKind.BINARY,
        lowerer.location(node),
        operands=(lhs, rhs),
        result_types=(result_type,),
        attributes={"operator": operator},
    )
    return operation.results[0]


def _logical_reduce(
    lowerer: FunctionLowerer,
    node: ast.Call,
    *,
    any_value: bool,
) -> MlirValue:
    bound = bind_call(lowerer, node, ("value", "axis"), required=("value",))
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    if not isinstance(source.type, TensorType) or source.type.dtype != intent_bool:
        lowerer.error(node, "I.any/I.all input must be a bool tensor")
    axes = require_axes(lowerer, bound["axis"]) if "axis" in bound else tuple(range(source.type.rank))
    axes = normalize_axes(lowerer, axes, source.type.rank, node)
    identity = lowerer.emit_literal(not any_value, node, ScalarType(intent_bool))
    normalized = set(axes)
    result_shape = tuple(
        dimension for axis, dimension in enumerate(source.type.shape) if axis not in normalized
    )
    result_type = lowerer.value_result_type(intent_bool, result_shape)
    operation = lowerer.emit(
        OperationKind.REDUCE,
        lowerer.location(node),
        operands=(source, identity),
        result_types=(result_type,),
        attributes={
            "axes": axes,
            "acc_dtype": intent_bool,
            "combine": "logical_or" if any_value else "logical_and",
            "component_count": 1,
            "capture_count": 0,
        },
    )
    return operation.results[0]
