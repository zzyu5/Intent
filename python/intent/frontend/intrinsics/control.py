from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from intent.ir import AutoExtent
from intent.ir import DomainFlavor
from intent.ir import DomainType
from intent.ir import OpCode
from intent.ir import PartitionMode
from intent.ir import PartitionType
from intent.ir import RaggedType
from intent.ir import RegionType
from intent.ir import ScalarType
from intent.ir import TensorType
from intent.ir import Value
from intent.ir.types import is_integer
from intent.language import index as intent_index

from ..expressions import compile_time_value
from ..model import IterationSpec
from ..model import Literal
from ..model import StaticTuple
from ..model import StreamSpec
from .common import bind_call

if TYPE_CHECKING:
    from ..lowering import FunctionLowerer


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


def _domain(lowerer: FunctionLowerer, node: ast.Call) -> Value:
    bound = bind_call(lowerer, node, ("start", "stop", "step"), required=("start", "stop"))
    expressions = [lowerer.lower_expression(bound["start"]), lowerer.lower_expression(bound["stop"])]
    if "step" in bound:
        expressions.append(lowerer.lower_expression(bound["step"]))
    runtime = any(not compile_time_value(expression)[0] for expression in expressions)
    operands = tuple(_integer_value(lowerer, expression, node) for expression in expressions)
    flavor = DomainFlavor.RUNTIME if runtime else (
        DomainFlavor.STRIDED if len(operands) == 3 else DomainFlavor.DENSE
    )
    operation = lowerer.emit(
        OpCode.DOMAIN,
        lowerer.location(node),
        operands=operands,
        result_types=(DomainType(flavor, 1),),
    )
    return operation.results[0]


def _partition(lowerer: FunctionLowerer, node: ast.Call) -> Value:
    bound = bind_call(lowerer, node, ("axis", "extent", "count"), required=("axis",))
    if ("extent" in bound) == ("count" in bound):
        lowerer.error(node, "I.partition requires exactly one of extent= or count=")
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
    if "count" in bound:
        expression = lowerer.lower_expression(bound["count"])
        if isinstance(expression, AutoExtent):
            lowerer.error(bound["count"], "partition count cannot use I.auto")
        operands.append(_integer_value(lowerer, expression, bound["count"]))
        mode = PartitionMode.COUNT
        attributes = {"mode": mode}
    else:
        expression = lowerer.lower_expression(bound["extent"])
        mode = PartitionMode.EXTENT
        attributes = {"mode": mode}
        if isinstance(expression, AutoExtent):
            attributes["extent"] = expression
        else:
            operands.append(_integer_value(lowerer, expression, bound["extent"]))
    operation = lowerer.emit(
        OpCode.PARTITION,
        lowerer.location(node),
        operands=tuple(operands),
        result_types=(PartitionType(mode, region_type),),
        attributes=attributes,
    )
    return operation.results[0]


def _parallel(lowerer: FunctionLowerer, node: ast.Call) -> IterationSpec:
    source = _iteration_source(lowerer, node)
    return IterationSpec(OpCode.PARALLEL, source)


def _ordered(lowerer: FunctionLowerer, node: ast.Call) -> IterationSpec:
    source = _iteration_source(lowerer, node)
    return IterationSpec(OpCode.ORDERED, source)


def _iteration_source(lowerer: FunctionLowerer, node: ast.Call) -> Value:
    bound = bind_call(lowerer, node, ("source",), required=("source",))
    expression = lowerer.lower_expression(bound["source"])
    if isinstance(expression, StaticTuple):
        domains = tuple(lowerer.materialize(element, bound["source"]) for element in expression.elements)
        if not domains or any(not isinstance(value.type, DomainType) for value in domains):
            lowerer.error(node, "domain product requires one or more domains")
        operation = lowerer.emit(
            OpCode.DOMAIN_PRODUCT,
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
        ("axis", "extent", "init"),
        required=("axis", "extent", "init"),
    )
    axis = lowerer.materialize(lowerer.lower_expression(bound["axis"]), bound["axis"])
    if not isinstance(axis.type, (DomainType, RegionType)):
        lowerer.error(node, "state_stream axis must be domain or region")
    extent_expression = lowerer.lower_expression(bound["extent"])
    if isinstance(extent_expression, AutoExtent):
        extent: AutoExtent | Value = extent_expression
    else:
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
    return StreamSpec(axis, initial, extent, node)


def _indices(lowerer: FunctionLowerer, node: ast.Call) -> Value:
    bound = bind_call(lowerer, node, ("region",), required=("region",))
    region = lowerer.materialize(lowerer.lower_expression(bound["region"]), bound["region"])
    if not isinstance(region.type, RegionType):
        lowerer.error(node, "I.indices requires a logical region")
    operation = lowerer.emit(
        OpCode.INDICES,
        lowerer.location(node),
        operands=(region,),
        result_types=(TensorType(intent_index, lowerer.dynamic_shape_for_region(region)),),
    )
    return operation.results[0]


def _ragged(lowerer: FunctionLowerer, node: ast.Call) -> Value:
    bound = bind_call(
        lowerer,
        node,
        ("outer", "offsets", "indices"),
        required=("outer", "offsets", "indices"),
    )
    outer = lowerer.materialize(lowerer.lower_expression(bound["outer"]), bound["outer"])
    offsets = lowerer.read_value(lowerer.lower_expression(bound["offsets"]), bound["offsets"])
    indices = lowerer.read_value(lowerer.lower_expression(bound["indices"]), bound["indices"])
    if not isinstance(outer.type, DomainType):
        lowerer.error(node, "ragged outer must be a domain")
    result_type = RaggedType(
        DomainType(DomainFlavor.RAGGED_OUTER, outer.type.rank),
        DomainType(DomainFlavor.RAGGED_MEMBER, 1),
    )
    operation = lowerer.emit(
        OpCode.RAGGED,
        lowerer.location(node),
        operands=(outer, offsets, indices),
        result_types=(result_type,),
    )
    return operation.results[0]


def _members(lowerer: FunctionLowerer, node: ast.Call) -> Value:
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
        OpCode.MEMBERS,
        lowerer.location(node),
        operands=(region,),
        result_types=(TensorType(intent_index, shape[:rank]),),
    )
    return operation.results[0]


def _integer_value(lowerer: FunctionLowerer, expression: object, node: ast.AST) -> Value:
    if isinstance(expression, Literal):
        return lowerer.materialize(expression, node, ScalarType(intent_index))
    value = lowerer.materialize(expression, node)
    if not is_integer(value.type):
        lowerer.error(node, "domain/partition extent must be integer/index")
    return value
