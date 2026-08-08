from __future__ import annotations

import ast
import operator
from enum import IntEnum

from intent.api import HelperDefinition
from intent.frontend.semantics import BinaryOperator
from intent.frontend.semantics import ComparePredicate
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import RaggedType
from intent.frontend.semantics import RecordType
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import UnaryOperator
from intent.frontend.mlir import MlirValue
from intent.language import DType
from intent.language import bool as intent_bool
from intent.language.builtins import Intrinsic
from intent.language.builtins import IntrinsicNamespace

from .indexing import lower_subscript
from .model import ConstexprBinding
from .model import Expression
from .model import Literal
from .model import ShapeValue
from .model import StaticTuple
from .model import StreamSpec


_BINARY_OPERATORS = {
    ast.Add: (BinaryOperator.ADD, operator.add),
    ast.Sub: (BinaryOperator.SUBTRACT, operator.sub),
    ast.Mult: (BinaryOperator.MULTIPLY, operator.mul),
    ast.Div: (BinaryOperator.TRUE_DIVIDE, operator.truediv),
    ast.FloorDiv: (BinaryOperator.FLOOR_DIVIDE, operator.floordiv),
    ast.Mod: (BinaryOperator.REMAINDER, operator.mod),
    ast.Pow: (BinaryOperator.POWER, operator.pow),
}

_COMPARE_PREDICATES = {
    ast.Eq: (ComparePredicate.EQ, operator.eq),
    ast.NotEq: (ComparePredicate.NE, operator.ne),
    ast.Lt: (ComparePredicate.LT, operator.lt),
    ast.LtE: (ComparePredicate.LE, operator.le),
    ast.Gt: (ComparePredicate.GT, operator.gt),
    ast.GtE: (ComparePredicate.GE, operator.ge),
}


def lower_expression(lowerer: object, node: ast.AST) -> Expression:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (bool, int, float)):
            return Literal(node.value)
        if node.value in (None, Ellipsis) or isinstance(node.value, str):
            return node.value
        lowerer.error(node, f"unsupported literal {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in lowerer.environment:
            return lowerer.environment[node.id]
        if node.id in lowerer.source.bindings:
            value = lowerer.source.bindings[node.id]
            if isinstance(value, (bool, int, float)):
                return Literal(value)
            return value
        lowerer.error(node, f"undefined name {node.id!r}")
    if isinstance(node, (ast.Tuple, ast.List)):
        return StaticTuple(tuple(lowerer.lower_expression(element) for element in node.elts))
    if isinstance(node, ast.Attribute):
        return _lower_attribute(lowerer, node)
    if isinstance(node, ast.Subscript):
        return lower_subscript(lowerer, node)
    if isinstance(node, ast.UnaryOp):
        return _lower_unary(lowerer, node)
    if isinstance(node, ast.BinOp):
        return _lower_binary(lowerer, node)
    if isinstance(node, ast.BoolOp):
        return _lower_bool(lowerer, node)
    if isinstance(node, ast.Compare):
        return _lower_compare(lowerer, node)
    if isinstance(node, ast.IfExp):
        return _lower_if_expression(lowerer, node)
    if isinstance(node, ast.Call):
        return _lower_call(lowerer, node)
    lowerer.error(node, f"unsupported Python expression {type(node).__name__}")


def compile_time_value(expression: Expression) -> tuple[bool, object]:
    if isinstance(expression, Literal):
        return True, expression.value
    if isinstance(expression, ConstexprBinding):
        return True, expression.python_value
    if isinstance(expression, (DType, IntEnum, str, type(None))):
        return True, expression
    return False, None


def _lower_attribute(lowerer: object, node: ast.Attribute) -> Expression:
    base = lowerer.lower_expression(node.value)
    if isinstance(base, MlirValue):
        if node.attr == "shape":
            return lowerer.shape_value(base, node)
        if isinstance(base.type, RecordType):
            field_type = dict(base.type.fields).get(node.attr)
            if field_type is None:
                lowerer.error(node, f"record has no field {node.attr!r}")
            operation = lowerer.emit(
                OperationKind.EXTRACT,
                lowerer.location(node),
                operands=(base,),
                result_types=(field_type,),
                attributes={"key": node.attr},
            )
            return operation.results[0]
        if isinstance(base.type, RaggedType) and node.attr == "outer":
            operation = lowerer.emit(
                OperationKind.RAGGED_OUTER,
                lowerer.location(node),
                operands=(base,),
                result_types=(base.type.outer,),
            )
            return operation.results[0]
        lowerer.error(node, f"SSA value has no source attribute {node.attr!r}")
    if isinstance(base, StreamSpec):
        if node.attr != "result":
            lowerer.error(node, f"state stream has no attribute {node.attr!r}")
        if base.results is None:
            lowerer.error(node, "stream.result is only available after its with block")
        if len(base.results) == 1:
            return base.results[0]
        return StaticTuple(tuple(base.results))
    if isinstance(base, ShapeValue):
        lowerer.error(node, "shape tuple has no named attributes")
    try:
        value = getattr(base, node.attr)
        if isinstance(value, (bool, int, float)) and not isinstance(value, IntEnum):
            return Literal(value)
        return value
    except AttributeError:
        lowerer.error(node, f"compile-time object has no attribute {node.attr!r}")


def _lower_unary(lowerer: object, node: ast.UnaryOp) -> Expression:
    operand = lowerer.lower_expression(node.operand)
    known, value = compile_time_value(operand)
    if known:
        if isinstance(node.op, (ast.USub, ast.UAdd)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                lowerer.error(node, "numeric unary operator requires int/float constexpr")
            return Literal(-value if isinstance(node.op, ast.USub) else +value)
        if isinstance(node.op, ast.Not):
            return Literal(not value)
        lowerer.error(node, "unsupported compile-time unary operator")
    operand_value = lowerer.materialize(operand, node.operand)
    if isinstance(node.op, ast.USub):
        operator_value = UnaryOperator.NEGATE
    elif isinstance(node.op, ast.Not):
        operator_value = UnaryOperator.NOT
    else:
        lowerer.error(node, f"unsupported runtime unary operator {type(node.op).__name__}")
    operation = lowerer.emit(
        OperationKind.UNARY,
        lowerer.location(node),
        operands=(operand_value,),
        result_types=(operand_value.type,),
        attributes={"operator": operator_value},
    )
    return operation.results[0]


def _lower_binary(lowerer: object, node: ast.BinOp) -> Expression:
    entry = _BINARY_OPERATORS.get(type(node.op))
    if entry is None:
        lowerer.error(node, f"unsupported binary operator {type(node.op).__name__}")
    ir_operator, python_operator = entry
    lhs = lowerer.lower_expression(node.left)
    rhs = lowerer.lower_expression(node.right)
    lhs_known, lhs_static = compile_time_value(lhs)
    rhs_known, rhs_static = compile_time_value(rhs)
    if lhs_known and rhs_known:
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (lhs_static, rhs_static)
        ):
            lowerer.error(node, "compile-time arithmetic requires numeric scalars")
        try:
            result = python_operator(lhs_static, rhs_static)
        except (ArithmeticError, TypeError, ValueError) as error:
            lowerer.error(node, f"invalid compile-time arithmetic: {error}")
        if not isinstance(result, (bool, int, float)):
            lowerer.error(node, "compile-time arithmetic must produce a scalar")
        return Literal(result)
    lhs_value, rhs_value = lowerer.coerce_pair(lhs, rhs, node)
    result_type = lowerer.broadcast_result_type(lhs_value.type, rhs_value.type, node)
    _, result_shape = lowerer.dtype_and_shape(result_type, node)
    lhs_value = lowerer.broadcast_value(lhs_value, result_shape, node)
    rhs_value = lowerer.broadcast_value(rhs_value, result_shape, node)
    operation = lowerer.emit(
        OperationKind.BINARY,
        lowerer.location(node),
        operands=(lhs_value, rhs_value),
        result_types=(result_type,),
        attributes={"operator": ir_operator},
    )
    return operation.results[0]


def _lower_bool(lowerer: object, node: ast.BoolOp) -> Expression:
    values = [lowerer.lower_expression(value) for value in node.values]
    if all(compile_time_value(value)[0] for value in values):
        static_values = [bool(compile_time_value(value)[1]) for value in values]
        return Literal(all(static_values) if isinstance(node.op, ast.And) else any(static_values))
    operator_value = (
        BinaryOperator.LOGICAL_AND if isinstance(node.op, ast.And) else BinaryOperator.LOGICAL_OR
    )
    result = values[0]
    for next_value in values[1:]:
        lhs, rhs = lowerer.coerce_pair(result, next_value, node)
        result_type = lowerer.broadcast_result_type(lhs.type, rhs.type, node)
        _, result_shape = lowerer.dtype_and_shape(result_type, node)
        lhs = lowerer.broadcast_value(lhs, result_shape, node)
        rhs = lowerer.broadcast_value(rhs, result_shape, node)
        operation = lowerer.emit(
            OperationKind.BINARY,
            lowerer.location(node),
            operands=(lhs, rhs),
            result_types=(result_type,),
            attributes={"operator": operator_value},
        )
        result = operation.results[0]
    return result


def _lower_compare(lowerer: object, node: ast.Compare) -> Expression:
    left = lowerer.lower_expression(node.left)
    comparisons: list[Expression] = []
    for operator_node, comparator_node in zip(node.ops, node.comparators):
        right = lowerer.lower_expression(comparator_node)
        entry = _COMPARE_PREDICATES.get(type(operator_node))
        if entry is None:
            lowerer.error(node, f"unsupported comparison {type(operator_node).__name__}")
        predicate, python_operator = entry
        lhs_known, lhs_static = compile_time_value(left)
        rhs_known, rhs_static = compile_time_value(right)
        if lhs_known and rhs_known:
            try:
                comparison = python_operator(lhs_static, rhs_static)
            except (TypeError, ValueError) as error:
                lowerer.error(node, f"invalid compile-time comparison: {error}")
            comparisons.append(Literal(bool(comparison)))
        else:
            lhs, rhs = lowerer.coerce_pair(left, right, node)
            value_type = lowerer.broadcast_result_type(lhs.type, rhs.type, node)
            _, shape = lowerer.dtype_and_shape(value_type, node)
            lhs = lowerer.broadcast_value(lhs, shape, node)
            rhs = lowerer.broadcast_value(rhs, shape, node)
            result_type = lowerer.value_result_type(intent_bool, shape)
            operation = lowerer.emit(
                OperationKind.COMPARE,
                lowerer.location(node),
                operands=(lhs, rhs),
                result_types=(result_type,),
                attributes={"predicate": predicate},
            )
            comparisons.append(operation.results[0])
        left = right
    if len(comparisons) == 1:
        return comparisons[0]
    synthetic = ast.BoolOp(op=ast.And(), values=[])
    ast.copy_location(synthetic, node)
    result = comparisons[0]
    for comparison in comparisons[1:]:
        lhs, rhs = lowerer.coerce_pair(result, comparison, node)
        result_type = lowerer.broadcast_result_type(lhs.type, rhs.type, node)
        _, result_shape = lowerer.dtype_and_shape(result_type, node)
        lhs = lowerer.broadcast_value(lhs, result_shape, node)
        rhs = lowerer.broadcast_value(rhs, result_shape, node)
        operation = lowerer.emit(
            OperationKind.BINARY,
            lowerer.location(node),
            operands=(lhs, rhs),
            result_types=(result_type,),
            attributes={"operator": BinaryOperator.LOGICAL_AND},
        )
        result = operation.results[0]
    return result


def _lower_if_expression(lowerer: object, node: ast.IfExp) -> Expression:
    condition = lowerer.lower_expression(node.test)
    known, value = compile_time_value(condition)
    if known:
        return lowerer.lower_expression(node.body if bool(value) else node.orelse)
    condition_value = lowerer.materialize(condition, node.test, ScalarType(intent_bool))
    true_value = lowerer.lower_expression(node.body)
    false_value = lowerer.lower_expression(node.orelse)
    lhs, rhs = lowerer.coerce_pair(true_value, false_value, node)
    result_type = lowerer.broadcast_result_type(lhs.type, rhs.type, node)
    _, result_shape = lowerer.dtype_and_shape(result_type, node)
    condition_value = lowerer.broadcast_value(condition_value, result_shape, node)
    lhs = lowerer.broadcast_value(lhs, result_shape, node)
    rhs = lowerer.broadcast_value(rhs, result_shape, node)
    operation = lowerer.emit(
        OperationKind.SELECT,
        lowerer.location(node),
        operands=(condition_value, lhs, rhs),
        result_types=(result_type,),
    )
    return operation.results[0]


def _lower_call(lowerer: object, node: ast.Call) -> Expression:
    if isinstance(node.func, ast.Attribute):
        base = lowerer.lower_expression(node.func.value)
        if isinstance(base, StreamSpec) and node.func.attr == "yield_":
            return _lower_stream_yield(lowerer, base, node)
    callee = lowerer.lower_expression(node.func)
    if isinstance(callee, (Intrinsic, IntrinsicNamespace)):
        from ..intrinsics import lower_intrinsic

        return lower_intrinsic(lowerer, callee.name, node)
    if isinstance(callee, HelperDefinition):
        if node.keywords:
            lowerer.error(node, "direct @intent.fn calls use positional SSA arguments")
        arguments = tuple(
            lowerer.read_value(lowerer.lower_expression(argument), argument)
            for argument in node.args
        )
        helper = lowerer.compiler.lower_helper(
            callee,
            tuple(argument.type for argument in arguments),
            lowerer.location(node),
        )
        operation = lowerer.emit(
            OperationKind.CALL,
            lowerer.location(node),
            operands=arguments,
            result_types=helper.result_types,
            attributes={"callee": helper.name},
            effects=lowerer.helper_call_effects(helper, arguments),
        )
        if len(operation.results) == 1:
            return operation.results[0]
        return StaticTuple(tuple(operation.results))
    if callee is slice:
        lowerer.error(node, "slice(...) is only valid inside an explicit index relation")
    lowerer.error(node, "only Intent intrinsics and @intent.fn values are callable in DSL code")


def _lower_stream_yield(lowerer: object, stream: StreamSpec, node: ast.Call) -> Expression:
    if node.keywords:
        lowerer.error(node, "stream.yield_ accepts positional carried values")
    if not lowerer.loop_stack or lowerer.loop_stack[-1].stream is not stream:
        lowerer.error(node, "stream.yield_ is only legal in its stream body")
    values: list[MlirValue] = []
    if len(node.args) == 1:
        expression = lowerer.lower_expression(node.args[0])
        if isinstance(expression, StaticTuple):
            values.extend(lowerer.materialize(value, node.args[0]) for value in expression.elements)
        else:
            values.append(lowerer.materialize(expression, node.args[0]))
    else:
        values.extend(
            lowerer.materialize(lowerer.lower_expression(argument), argument)
            for argument in node.args
        )
    if len(values) != len(stream.initial_state):
        lowerer.error(node, "stream.yield_ state count does not match init schema")
    for value, initial in zip(values, stream.initial_state):
        if not lowerer.types_compatible_for_literal(value.type, initial.type):
            lowerer.error(node, "stream.yield_ state type does not match init schema")
    lowerer.loop_stack[-1].pending_stream_state = tuple(values)
    return StaticTuple(())
