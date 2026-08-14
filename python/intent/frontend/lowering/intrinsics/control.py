from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from intent.frontend.semantics import AutoExtent
from intent.frontend.semantics import BufferType
from intent.frontend.semantics import DomainFlavor
from intent.frontend.semantics import DomainType
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import PartitionMode
from intent.frontend.semantics import PartitionType
from intent.frontend.semantics import RaggedType
from intent.frontend.semantics import RegionType
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import TensorType
from intent.frontend.semantics import LogicalIndexType
from intent.frontend.mlir import MlirValue
from intent.frontend.semantics.types import is_integer
from intent.language import index as intent_index
from intent.language.dtypes import DTypeCategory

from ..ast.expressions import compile_time_value
from ..ast.model import IterationSpec
from ..ast.model import Literal
from ..ast.model import StaticTuple
from ..ast.model import StreamSpec
from .common import bind_call
from .common import require_static_int

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer


def lower_control_intrinsic(
    lowerer: FunctionLowerer,
    name: str,
    node: ast.Call,
) -> object:
    handlers = {
        "auto": _auto,
        "domain": _domain,
        "partition": _partition,
        "parallel": _parallel,
        "ordered": _ordered,
        "state_stream": _state_stream,
        "indices": _indices,
        "end": _region_end,
        "assume_in_bounds": _assume_in_bounds,
        "ragged": _ragged,
        "members": _members,
    }
    handler = handlers.get(name)
    if handler is None:
        return NotImplemented
    return handler(lowerer, node)


def _auto(lowerer: FunctionLowerer, node: ast.Call) -> AutoExtent:
    bound = bind_call(lowerer, node, ("name",), required=("name",))
    expression = lowerer.lower_expression(bound["name"])
    known, value = compile_time_value(expression)
    if not known or not isinstance(value, str):
        lowerer.error(node, "I.auto name must be a compile-time string")
    return AutoExtent(value)


def _domain(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("start", "stop", "step"), required=("start", "stop"))
    expressions = [lowerer.lower_expression(bound["start"]), lowerer.lower_expression(bound["stop"])]
    if "step" in bound:
        step = lowerer.lower_expression(bound["step"])
        known, value = compile_time_value(step)
        if not known or isinstance(value, bool) or not isinstance(value, int) or value != 1:
            lowerer.error(
                bound["step"],
                "I.domain currently requires a compile-time unit step",
            )
        expressions.append(step)
    runtime = any(not compile_time_value(expression)[0] for expression in expressions)
    operands = tuple(_integer_value(lowerer, expression, node) for expression in expressions)
    flavor = DomainFlavor.RUNTIME if runtime else (
        DomainFlavor.STRIDED if len(operands) == 3 else DomainFlavor.DENSE
    )
    operation = lowerer.emit(
        OperationKind.DOMAIN,
        lowerer.location(node),
        operands=operands,
        result_types=(DomainType(flavor, 1),),
    )
    return operation.results[0]


def _partition(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("axis", "extent", "count"), required=("axis",))
    if ("extent" in bound) == ("count" in bound):
        lowerer.error(node, "I.partition requires exactly one of extent= or count=")
    if "count" in bound:
        lowerer.error(
            bound["count"],
            "I.partition(count=...) is reserved but not supported by the current realization contract; use extent=...",
        )
    source = lowerer.materialize(lowerer.lower_expression(bound["axis"]), bound["axis"])
    if not isinstance(source.type, (DomainType, RegionType)):
        lowerer.error(node, "partition axis must be a domain or region")
    if isinstance(source.type, RegionType):
        relation = source.type.relation
    elif source.type.flavor is DomainFlavor.RAGGED_MEMBER:
        relation = "ragged_member"
    elif source.type.flavor is DomainFlavor.RAGGED_OUTER:
        relation = "ragged_outer"
    else:
        relation = "domain"
    region_type = RegionType(source.type.rank, relation)
    operands = [source]
    attributes: dict[str, object]
    expression = lowerer.lower_expression(bound["extent"])
    mode = PartitionMode.EXTENT
    attributes = {"mode": mode}
    if isinstance(expression, AutoExtent):
        attributes["extent"] = expression
    else:
        operands.append(_integer_value(lowerer, expression, bound["extent"]))
    operation = lowerer.emit(
        OperationKind.PARTITION,
        lowerer.location(node),
        operands=tuple(operands),
        result_types=(PartitionType(mode, region_type),),
        attributes=attributes,
    )
    return operation.results[0]


def _parallel(lowerer: FunctionLowerer, node: ast.Call) -> IterationSpec:
    source = _iteration_source(lowerer, node)
    return IterationSpec(OperationKind.PARALLEL, source)


def _ordered(lowerer: FunctionLowerer, node: ast.Call) -> IterationSpec:
    source = _iteration_source(lowerer, node)
    return IterationSpec(OperationKind.ORDERED, source)


def _iteration_source(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("source",), required=("source",))
    expression = lowerer.lower_expression(bound["source"])
    if isinstance(expression, StaticTuple):
        domains = tuple(lowerer.materialize(element, bound["source"]) for element in expression.elements)
        if not domains or any(not isinstance(value.type, DomainType) for value in domains):
            lowerer.error(node, "domain product requires one or more domains")
        operation = lowerer.emit(
            OperationKind.DOMAIN_PRODUCT,
            lowerer.location(node),
            operands=domains,
            result_types=(DomainType(DomainFlavor.PRODUCT, sum(value.type.rank for value in domains)),),
        )
        return operation.results[0]
    source = lowerer.materialize(expression, bound["source"])
    if not isinstance(source.type, (DomainType, RegionType, PartitionType)):
        lowerer.error(node, "iteration source must be domain, region, or partition")
    return source


def _state_stream(lowerer: FunctionLowerer, node: ast.Call) -> StreamSpec:
    bound = bind_call(
        lowerer,
        node,
        ("axis", "extent", "init", "stop"),
        required=("axis", "extent", "init"),
    )
    axis = lowerer.materialize(lowerer.lower_expression(bound["axis"]), bound["axis"])
    if not isinstance(axis.type, (DomainType, RegionType)):
        lowerer.error(node, "state_stream axis must be domain or region")
    extent_expression = lowerer.lower_expression(bound["extent"])
    if isinstance(extent_expression, AutoExtent):
        extent: AutoExtent | MlirValue = extent_expression
    else:
        known, value = compile_time_value(extent_expression)
        if not known or isinstance(value, bool) or not isinstance(value, int):
            lowerer.error(
                bound["extent"],
                "state_stream extent must be a compile-time integer or I.auto(...)",
            )
        extent = _integer_value(lowerer, extent_expression, bound["extent"])
    initial_expression = lowerer.lower_expression(bound["init"])
    if isinstance(initial_expression, StaticTuple):
        initial = tuple(
            lowerer.materialize(element, bound["init"])
            for element in initial_expression.elements
        )
    else:
        initial = (lowerer.materialize(initial_expression, bound["init"]),)
    if not initial:
        lowerer.error(node, "state_stream requires at least one carried value")
    stop = None
    if "stop" in bound:
        stop = lowerer.materialize(
            lowerer.lower_expression(bound["stop"]), bound["stop"]
        )
        if not isinstance(stop.type, ScalarType) or not is_integer(stop.type):
            lowerer.error(bound["stop"], "state_stream stop must be an integer index")
    return StreamSpec(axis, initial, extent, stop, node)


def _indices(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("region",), required=("region",))
    region = lowerer.materialize(lowerer.lower_expression(bound["region"]), bound["region"])
    if not isinstance(region.type, (DomainType, RegionType)):
        lowerer.error(node, "I.indices requires a logical domain or region")
    operation = lowerer.emit(
        OperationKind.INDICES,
        lowerer.location(node),
        operands=(region,),
        result_types=(TensorType(intent_index, lowerer.dynamic_shape_for_region(region)),),
    )
    return operation.results[0]


def _region_end(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("region",), required=("region",))
    region = lowerer.materialize(
        lowerer.lower_expression(bound["region"]), bound["region"]
    )
    if not isinstance(region.type, (DomainType, RegionType)) or region.type.rank != 1:
        lowerer.error(node, "I.end requires a rank-one logical domain or region")
    operation = lowerer.emit(
        OperationKind.REGION_END,
        lowerer.location(node),
        operands=(region,),
        result_types=(ScalarType(intent_index),),
    )
    return operation.results[0]


def _assume_in_bounds(lowerer: FunctionLowerer, node: ast.Call) -> StaticTuple:
    bound = bind_call(
        lowerer,
        node,
        ("index", "view", "axis"),
        required=("index", "view", "axis"),
    )
    index = lowerer.materialize(
        lowerer.lower_expression(bound["index"]), bound["index"]
    )
    view = lowerer.materialize(
        lowerer.lower_expression(bound["view"]), bound["view"]
    )
    scalar_index = isinstance(index.type, (ScalarType, LogicalIndexType)) and is_integer(
        index.type
    )
    tensor_index = isinstance(index.type, TensorType) and index.type.dtype.category in (
        DTypeCategory.SIGNED_INTEGER,
        DTypeCategory.UNSIGNED_INTEGER,
        DTypeCategory.INDEX,
    )
    if not scalar_index and not tensor_index:
        lowerer.error(bound["index"], "assumed index must be an integer scalar or tensor")
    if view in lowerer.view_kinds and isinstance(view.type, TensorType):
        shape = view.type.shape
    elif isinstance(view.type, BufferType):
        shape = view.type.shape
    else:
        lowerer.error(
            bound["view"],
            "assumed bound must reference an ABI view or logical buffer",
        )
    axis = require_static_int(lowerer, bound["axis"])
    if not -len(shape) <= axis < len(shape):
        lowerer.error(bound["axis"], "assumed bound axis is outside the target rank")
    lowerer.emit(
        OperationKind.ASSUME_IN_BOUNDS,
        lowerer.location(node),
        operands=(index, view),
        attributes={"axis": axis % len(shape)},
    )
    return StaticTuple(())


def _ragged(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("outer", "members", "offsets", "indices"),
        required=("outer", "members", "offsets"),
    )
    outer = lowerer.materialize(lowerer.lower_expression(bound["outer"]), bound["outer"])
    members = lowerer.materialize(
        lowerer.lower_expression(bound["members"]), bound["members"]
    )
    offsets = lowerer.read_value(lowerer.lower_expression(bound["offsets"]), bound["offsets"])
    if not isinstance(outer.type, DomainType):
        lowerer.error(node, "ragged outer must be a domain")
    if not isinstance(members.type, DomainType) or members.type.rank != 1:
        lowerer.error(node, "ragged members must be a rank-one domain")
    operands = [outer, members, offsets]
    if "indices" in bound:
        indices = lowerer.read_value(
            lowerer.lower_expression(bound["indices"]), bound["indices"]
        )
        operands.append(indices)
    result_type = RaggedType(
        DomainType(DomainFlavor.RAGGED_OUTER, outer.type.rank),
        DomainType(DomainFlavor.RAGGED_MEMBER, members.type.rank),
    )
    operation = lowerer.emit(
        OperationKind.RAGGED,
        lowerer.location(node),
        operands=tuple(operands),
        result_types=(result_type,),
    )
    return operation.results[0]


def _members(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("region",), required=("region",))
    region = lowerer.materialize(lowerer.lower_expression(bound["region"]), bound["region"])
    if not isinstance(region.type, (DomainType, RegionType)):
        lowerer.error(node, "I.members requires a ragged member domain/region")
    if isinstance(region.type, DomainType) and region.type.flavor is not DomainFlavor.RAGGED_MEMBER:
        lowerer.error(node, "I.members requires a ragged member domain")
    if isinstance(region.type, RegionType) and region.type.relation != "ragged_member":
        lowerer.error(node, "I.members requires a ragged_member region")
    rank = getattr(region.type, "rank", 1)
    shape = tuple(lowerer.dynamic_shape_for_region(region))
    operation = lowerer.emit(
        OperationKind.MEMBERS,
        lowerer.location(node),
        operands=(region,),
        result_types=(TensorType(intent_index, shape[:rank]),),
    )
    return operation.results[0]


def _integer_value(lowerer: FunctionLowerer, expression: object, node: ast.AST) -> MlirValue:
    known, static_value = compile_time_value(expression)
    if known:
        if isinstance(static_value, bool) or not isinstance(static_value, int):
            lowerer.error(node, "domain/partition extent must be an integer constexpr")
        return lowerer.materialize(
            Literal(static_value),
            node,
            ScalarType(intent_index),
        )
    value = lowerer.materialize(expression, node)
    if not is_integer(value.type):
        lowerer.error(node, "domain/partition extent must be integer/index")
    return value
