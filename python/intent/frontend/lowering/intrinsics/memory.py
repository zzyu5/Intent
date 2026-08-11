from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from intent.frontend.semantics import AtomicOrdering
from intent.frontend.semantics import BufferType
from intent.frontend.semantics import Effect
from intent.frontend.semantics import EffectKind
from intent.frontend.semantics import LogicalIndexType
from intent.frontend.semantics import MemoryScope
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import ResourceKind
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import TensorType
from intent.frontend.mlir import MlirValue
from intent.frontend.semantics import broadcast_shape
from intent.frontend.semantics.types import is_integer
from intent.language import f32
from intent.language import bool as intent_bool

from ..ast.indexing import lower_index
from ..ast.indexing import validate_indexed_value
from ..ast.model import Literal
from ..ast.model import StaticTuple
from .common import bind_call
from .common import lower_shape
from .common import require_dtype
from .structured import _callable_symbol

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer


def lower_memory_intrinsic(
    lowerer: FunctionLowerer,
    name: str,
    node: ast.Call,
) -> object:
    handlers = {
        "gather": _gather,
        "scatter_unique": lambda context, call: _scatter(context, call, reduce=False),
        "scatter_reduce": lambda context, call: _scatter(context, call, reduce=True),
        "buffer": _buffer,
        "store": _store,
        "mutable_load": _mutable_load,
        "atomic_add": _atomic_add,
        "atomic_cas": _atomic_cas,
        "fence": _fence,
        "random": _random,
    }
    handler = handlers.get(name)
    if handler is None:
        return NotImplemented
    return handler(lowerer, node)


def _gather(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("source", "index", "valid", "fill", "ordering", "scope"),
        required=("source", "index"),
    )
    source = lowerer.materialize(lowerer.lower_expression(bound["source"]), bound["source"])
    if not isinstance(source.type, TensorType):
        lowerer.error(node, "I.gather source must be a tensor/view")
    lowered = lower_index(lowerer, source, bound["index"], first_operand_position=1)
    if "valid" in bound:
        valid = lowerer.materialize(lowerer.lower_expression(bound["valid"]), bound["valid"])
    else:
        valid = lowerer.emit_literal(True, node, ScalarType(intent_bool))
    if "fill" in bound:
        fill_expression = lowerer.lower_expression(bound["fill"])
        fill = lowerer.materialize(
            fill_expression,
            bound["fill"],
            ScalarType(source.type.dtype) if isinstance(fill_expression, Literal) else None,
        )
    else:
        fill = lowerer.emit_literal(
            False if source.type.dtype == intent_bool else 0,
            node,
            ScalarType(source.type.dtype),
        )
    valid_dtype, valid_shape = lowerer.dtype_and_shape(valid.type, node)
    if valid_dtype.name != "bool":
        lowerer.error(node, "gather valid predicate must have bool dtype")
    fill_dtype, fill_shape = lowerer.dtype_and_shape(fill.type, node)
    if fill_dtype != source.type.dtype:
        lowerer.error(node, "gather fill dtype must match source dtype")
    result_shape = tuple(lowered.result_shape)
    for shape, subject in ((valid_shape, "valid"), (fill_shape, "fill")):
        try:
            broadcasted = tuple(broadcast_shape(shape, result_shape))
        except ValueError as error:
            lowerer.error(node, str(error))
        if broadcasted != result_shape:
            lowerer.error(node, f"gather {subject} cannot broadcast to indexed shape")
    valid = lowerer.broadcast_value(valid, result_shape, node)
    fill = lowerer.broadcast_value(fill, result_shape, node)
    operands = (source, *lowered.operands, valid, fill)
    effects = ()
    if source in lowerer.view_kinds:
        lowerer.require_readable_view(source, node)
        effects = (Effect(EffectKind.READ, ResourceKind.EXTERNAL_VIEW, source),)
    attributes: dict[str, object] = {
        "index": lowered.relation,
        "valid_operand_index": len(operands) - 2,
        "fill_operand_index": len(operands) - 1,
    }
    attributes.update(_optional_ordering(lowerer, bound, node))
    operation = lowerer.emit(
        OperationKind.GATHER,
        lowerer.location(node),
        operands=operands,
        result_types=(lowerer.value_result_type(source.type.dtype, result_shape),),
        attributes=attributes,
        effects=effects,
    )
    return operation.results[0]


def _scatter(lowerer: FunctionLowerer, node: ast.Call, *, reduce: bool) -> StaticTuple:
    positional = ("destination", "index", "value", "combine", "ordering", "scope")
    required = ("destination", "index", "value", "combine") if reduce else (
        "destination",
        "index",
        "value",
    )
    bound = bind_call(lowerer, node, positional, required=required)
    destination = lowerer.materialize(
        lowerer.lower_expression(bound["destination"]), bound["destination"]
    )
    if not isinstance(destination.type, TensorType) or destination not in lowerer.view_kinds:
        lowerer.error(node, "scatter destination must be an external output/InOut view")
    lowerer.require_writable_view(destination, node)
    lowered = lower_index(lowerer, destination, bound["index"], first_operand_position=1)
    value_expression = lowerer.lower_expression(bound["value"])
    value = lowerer.materialize(
        value_expression,
        bound["value"],
        ScalarType(destination.type.dtype) if isinstance(value_expression, Literal) else None,
    )
    value = validate_indexed_value(
        lowerer, value, lowered.result_shape, node, expected_dtype=destination.type.dtype
    )
    operands = (destination, *lowered.operands, value)
    attributes: dict[str, object] = {
        "index": lowered.relation,
        "value_operand_index": len(operands) - 1,
    }
    if reduce:
        attributes["combine"] = _callable_symbol(
            lowerer,
            bound["combine"],
        )
    attributes.update(_optional_ordering(lowerer, bound, node))
    lowerer.emit(
        OperationKind.SCATTER_REDUCE if reduce else OperationKind.SCATTER_UNIQUE,
        lowerer.location(node),
        operands=operands,
        attributes=attributes,
        effects=(Effect(EffectKind.WRITE, ResourceKind.EXTERNAL_VIEW, destination),),
    )
    return StaticTuple(())


def _buffer(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("shape", "dtype", "init"),
        required=("shape", "dtype"),
    )
    dtype = require_dtype(lowerer, bound["dtype"])
    shape = lower_shape(lowerer, bound["shape"])
    operands: tuple[MlirValue, ...] = ()
    if "init" in bound:
        expression = lowerer.lower_expression(bound["init"])
        initializer = lowerer.materialize(
            expression,
            bound["init"],
            ScalarType(dtype) if isinstance(expression, Literal) else None,
        )
        operands = (initializer,)
    operation = lowerer.emit(
        OperationKind.BUFFER,
        lowerer.location(node),
        operands=operands,
        result_types=(BufferType(dtype, shape),),
    )
    return operation.results[0]


def _store(lowerer: FunctionLowerer, node: ast.Call) -> StaticTuple:
    bound = bind_call(
        lowerer,
        node,
        ("target", "index", "value", "ordering", "scope"),
        required=("target", "index", "value"),
    )
    target = lowerer.materialize(lowerer.lower_expression(bound["target"]), bound["target"])
    if not isinstance(target.type, (TensorType, BufferType)):
        lowerer.error(node, "I.store target must be a view or logical buffer")
    lowered = lower_index(lowerer, target, bound["index"], first_operand_position=2)
    value_expression = lowerer.lower_expression(bound["value"])
    value = lowerer.materialize(
        value_expression,
        bound["value"],
        ScalarType(target.type.dtype) if isinstance(value_expression, Literal) else None,
    )
    value = validate_indexed_value(
        lowerer, value, lowered.result_shape, node, expected_dtype=target.type.dtype
    )
    operands = (target, value, *lowered.operands)
    attributes: dict[str, object] = {
        "index": lowered.relation,
        "value_operand_index": 1,
    }
    attributes.update(_optional_ordering(lowerer, bound, node))
    if isinstance(target.type, BufferType):
        opcode = OperationKind.BUFFER_STORE
        effect = Effect(EffectKind.WRITE, ResourceKind.LOGICAL_BUFFER, target)
    else:
        if target not in lowerer.view_kinds:
            lowerer.error(node, "pure tensor SSA cannot be mutated")
        lowerer.require_writable_view(target, node)
        opcode = OperationKind.VIEW_STORE
        effect = Effect(EffectKind.WRITE, ResourceKind.EXTERNAL_VIEW, target)
    lowerer.emit(
        opcode,
        lowerer.location(node),
        operands=operands,
        attributes=attributes,
        effects=(effect,),
    )
    return StaticTuple(())


def _mutable_load(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("target", "index", "ordering", "scope"),
        required=("target", "index"),
    )
    target = lowerer.materialize(lowerer.lower_expression(bound["target"]), bound["target"])
    if not isinstance(target.type, (TensorType, BufferType)):
        lowerer.error(node, "I.mutable_load target must be a view or logical buffer")
    lowered = lower_index(lowerer, target, bound["index"], first_operand_position=1)
    attributes: dict[str, object] = {"index": lowered.relation}
    attributes.update(_optional_ordering(lowerer, bound, node))
    if isinstance(target.type, BufferType):
        opcode = OperationKind.BUFFER_LOAD
        resource = ResourceKind.LOGICAL_BUFFER
    else:
        if target not in lowerer.view_kinds:
            lowerer.error(node, "mutable_load tensor target must be an external view")
        lowerer.require_readable_view(target, node)
        opcode = OperationKind.VIEW_LOAD
        resource = ResourceKind.EXTERNAL_VIEW
    operation = lowerer.emit(
        opcode,
        lowerer.location(node),
        operands=(target, *lowered.operands),
        result_types=(lowerer.value_result_type(target.type.dtype, lowered.result_shape),),
        attributes=attributes,
        effects=(Effect(EffectKind.READ, resource, target),),
    )
    return operation.results[0]


def _atomic_add(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("target", "index", "value", "ordering", "scope"),
        required=("target", "index", "value"),
    )
    target, lowered, value = _atomic_inputs(lowerer, bound, node, first_position=1)
    operands = (target, *lowered.operands, value)
    attributes = {
        "index": lowered.relation,
        "value_operand_index": len(operands) - 1,
        **_required_ordering(lowerer, bound, node),
    }
    operation = lowerer.emit(
        OperationKind.ATOMIC_ADD,
        lowerer.location(node),
        operands=operands,
        result_types=(
            lowerer.value_result_type(target.type.dtype, lowered.result_shape),
        ),
        attributes=attributes,
        effects=(_atomic_effect(target),),
    )
    return operation.results[0]


def _atomic_cas(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("target", "index", "compare", "value", "ordering", "scope"),
        required=("target", "index", "compare", "value"),
    )
    target = lowerer.materialize(lowerer.lower_expression(bound["target"]), bound["target"])
    if not isinstance(target.type, (TensorType, BufferType)):
        lowerer.error(node, "atomic target must be view or buffer")
    if isinstance(target.type, TensorType):
        lowerer.require_atomic_view(target, node)
    lowered = lower_index(lowerer, target, bound["index"], first_operand_position=1)
    compare_expression = lowerer.lower_expression(bound["compare"])
    value_expression = lowerer.lower_expression(bound["value"])
    compare = lowerer.materialize(
        compare_expression,
        bound["compare"],
        ScalarType(target.type.dtype) if isinstance(compare_expression, Literal) else None,
    )
    value = lowerer.materialize(
        value_expression,
        bound["value"],
        ScalarType(target.type.dtype) if isinstance(value_expression, Literal) else None,
    )
    compare = validate_indexed_value(
        lowerer, compare, lowered.result_shape, node, expected_dtype=target.type.dtype
    )
    value = validate_indexed_value(
        lowerer, value, lowered.result_shape, node, expected_dtype=target.type.dtype
    )
    if not lowerer.types_compatible_for_literal(compare.type, value.type):
        lowerer.error(node, "atomic_cas compare and value types must match exactly")
    operands = (target, *lowered.operands, compare, value)
    operation = lowerer.emit(
        OperationKind.ATOMIC_CAS,
        lowerer.location(node),
        operands=operands,
        result_types=(value.type,),
        attributes={
            "index": lowered.relation,
            "compare_operand_index": len(operands) - 2,
            "value_operand_index": len(operands) - 1,
            **_required_ordering(lowerer, bound, node),
        },
        effects=(_atomic_effect(target),),
    )
    return operation.results[0]


def _fence(lowerer: FunctionLowerer, node: ast.Call) -> StaticTuple:
    lowerer.error(
        node,
        "I.fence has no portable semantics across the supported tile languages",
    )


def _random(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("seed", "index", "dtype"),
        required=("seed", "index"),
    )
    seed = lowerer.materialize(
        lowerer.lower_expression(bound["seed"]), bound["seed"]
    )
    identity = lowerer.materialize(
        lowerer.lower_expression(bound["index"]), bound["index"]
    )
    if not is_integer(seed.type):
        lowerer.error(bound["seed"], "random seed must be integer/index")
    dtype = require_dtype(lowerer, bound["dtype"]) if "dtype" in bound else f32
    if dtype != f32:
        lowerer.error(node, "I.random currently produces f32 uniform values")
    if is_integer(identity.type):
        shape: tuple[object, ...] = ()
    elif isinstance(identity.type, TensorType):
        if not is_integer(ScalarType(identity.type.dtype)):
            lowerer.error(bound["index"], "random counter tensor must be integer/index")
        shape = tuple(identity.type.shape)
    else:
        lowerer.error(node, "random counter must be an integer scalar/tensor")
    operation = lowerer.emit(
        OperationKind.RANDOM,
        lowerer.location(node),
        operands=(seed, identity),
        result_types=(lowerer.value_result_type(dtype, shape),),
        attributes={"algorithm": "counter_xorshift32"},
    )
    return operation.results[0]


def _atomic_inputs(
    lowerer: FunctionLowerer,
    bound: dict[str, ast.AST],
    node: ast.Call,
    *,
    first_position: int,
) -> tuple[MlirValue, object, MlirValue]:
    target = lowerer.materialize(lowerer.lower_expression(bound["target"]), bound["target"])
    if not isinstance(target.type, (TensorType, BufferType)):
        lowerer.error(node, "atomic target must be view or buffer")
    if isinstance(target.type, TensorType):
        lowerer.require_atomic_view(target, node)
    lowered = lower_index(lowerer, target, bound["index"], first_operand_position=first_position)
    value_expression = lowerer.lower_expression(bound["value"])
    value = lowerer.materialize(
        value_expression,
        bound["value"],
        ScalarType(target.type.dtype) if isinstance(value_expression, Literal) else None,
    )
    value = validate_indexed_value(
        lowerer, value, lowered.result_shape, node, expected_dtype=target.type.dtype
    )
    return target, lowered, value


def _atomic_effect(target: MlirValue) -> Effect:
    resource = (
        ResourceKind.LOGICAL_BUFFER
        if isinstance(target.type, BufferType)
        else ResourceKind.EXTERNAL_VIEW
    )
    return Effect(EffectKind.ATOMIC, resource, target)


def _optional_ordering(
    lowerer: FunctionLowerer,
    bound: dict[str, ast.AST],
    node: ast.AST,
) -> dict[str, object]:
    if "ordering" not in bound and "scope" not in bound:
        return {}
    if "ordering" not in bound or "scope" not in bound:
        lowerer.error(node, "ordering and scope must be specified together")
    return {
        "ordering": _ordering(lowerer, bound["ordering"]),
        "scope": _scope(lowerer, bound["scope"]),
    }


def _required_ordering(
    lowerer: FunctionLowerer,
    bound: dict[str, ast.AST],
    node: ast.AST,
) -> dict[str, object]:
    if "ordering" not in bound and "scope" not in bound:
        return {"ordering": AtomicOrdering.RELAXED, "scope": MemoryScope.DEVICE}
    return _optional_ordering(lowerer, bound, node)


def _ordering(lowerer: FunctionLowerer, node: ast.AST) -> AtomicOrdering:
    expression = lowerer.lower_expression(node)
    value = expression.value if isinstance(expression, Literal) else expression
    if isinstance(value, AtomicOrdering):
        return value
    if isinstance(value, str):
        try:
            return AtomicOrdering(value)
        except ValueError:
            pass
    lowerer.error(node, "invalid atomic ordering")


def _scope(lowerer: FunctionLowerer, node: ast.AST) -> MemoryScope:
    expression = lowerer.lower_expression(node)
    value = expression.value if isinstance(expression, Literal) else expression
    if isinstance(value, MemoryScope):
        return value
    if isinstance(value, str):
        try:
            return MemoryScope(value)
        except ValueError:
            pass
    lowerer.error(node, "invalid memory scope")
