from __future__ import annotations

import ast
from dataclasses import replace
from typing import TYPE_CHECKING

from intent.api import HelperDefinition
from intent.frontend.mlir import MlirValue
from intent.frontend.mlir.attributes import SparseFormatAttribute
from intent.frontend.semantics import BinaryOperator
from intent.frontend.semantics import ComparePredicate
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import RecordType
from intent.frontend.semantics import ScaledFormatKind
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import StaticDim
from intent.frontend.semantics import TensorType
from intent.frontend.semantics import TupleType
from intent.frontend.semantics import ValueType
from intent.frontend.semantics.types import dims_compatible
from intent.language import DTypeCategory
from intent.language import bool as intent_bool
from intent.language import i32
from intent.language import index as intent_index
from intent.language import u32
from intent.language.builtins import Intrinsic
from intent.language.builtins import ScaledFormat

from ..ast.expressions import compile_time_value
from ..ast.model import ConstexprBinding
from ..ast.model import Expression
from ..ast.model import Literal
from ..ast.model import ShapeDimension
from ..ast.model import SparseFormatSpec
from ..ast.model import StaticTuple
from .common import bind_call
from .common import bind_declared_call
from .common import optional_dtype
from .common import require_axes
from .common import require_dtype
from .common import require_static_bool
from .common import require_static_int
from .common import normalize_axes

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer


def lower_structured_intrinsic(
    lowerer: FunctionLowerer,
    name: str,
    node: ast.Call,
) -> object:
    if name in (
        "reduce",
        "reduce.max",
        "reduce.sum",
        "reduce.any",
        "reduce.all",
    ):
        return _reduce(lowerer, name, node)
    if name == "arg_reduce.max":
        return _arg_reduce_max(lowerer, node)
    if name in ("scan", "cumsum", "cummax"):
        return _scan(lowerer, node, name)
    if name == "region_fold":
        return _region_fold(lowerer, node)
    if name == "region_scan":
        return _region_scan(lowerer, node)
    if name == "contract":
        return _contract(lowerer, node)
    if name in ("quantize", "quantized_dot"):
        from .quantization import lower_quantization

        return lower_quantization(lowerer, name, node)
    if name == "scaled_contract":
        return _scaled_contract(lowerer, node)
    if name in ("sparse.one_of_two", "sparse.two_of_four"):
        return _sparse_format(lowerer, name, node)
    if name == "sparse_contract":
        return _sparse_contract(lowerer, node)
    if name == "sparse_contract_2to4":
        return _sparse_contract_2to4(lowerer, node)
    if name == "histogram":
        return _histogram(lowerer, node)
    return NotImplemented


def _reduce(lowerer: FunctionLowerer, name: str, node: ast.Call) -> MlirValue:
    bound = (
        bind_call(
            lowerer,
            node,
            ("value", "axis", "identity", "combine", "combine_operands", "acc_dtype"),
            required=("value", "axis", "identity", "combine"),
        )
        if name == "reduce"
        else bind_declared_call(lowerer, node, name)
    )
    source = lowerer.read_value(
        lowerer.lower_expression(bound["value"]), bound["value"]
    )
    axes = require_axes(lowerer, bound["axis"])
    source_components, component_names = _source_components(lowerer, source, node)
    if not source_components or any(
        not isinstance(value.type, TensorType) for value in source_components
    ):
        lowerer.error(node, "I.reduce source components must be tensors")
    rank = source_components[0].type.rank
    axes = normalize_axes(lowerer, axes, rank, node)
    if any(value.type.rank != rank for value in source_components):
        lowerer.error(node, "I.reduce source components must have equal rank")
    if name != "reduce":
        if component_names != ("value",):
            lowerer.error(node, f"I.{name} requires one tensor; use I.reduce for products")
        dtype = source_components[0].type.dtype
        if name in ("reduce.any", "reduce.all"):
            if dtype != intent_bool:
                lowerer.error(node, f"I.{name} requires a bool tensor")
        elif dtype == intent_bool:
            lowerer.error(node, f"I.{name} requires a numeric tensor")

    acc_dtype = optional_dtype(lowerer, bound.get("acc_dtype"))
    if acc_dtype is not None:
        if len(source_components) != 1:
            lowerer.error(node, "record reduce uses explicit field dtypes")
    elif name == "reduce.sum":
        acc_dtype = _sum_accumulator_dtype(source_components[0].type.dtype)
    if acc_dtype is not None and source_components[0].type.dtype != acc_dtype:
        source_components = (
            lowerer.emit(
                OperationKind.CAST,
                lowerer.location(node),
                operands=(source_components[0],),
                result_types=(
                    TensorType(acc_dtype, source_components[0].type.shape),
                ),
            ).results[0],
        )

    reduced = set(axes)
    result_components = tuple(
        lowerer.value_result_type(
            source_component.type.dtype,
            tuple(
                dimension
                for axis, dimension in enumerate(source_component.type.shape)
                if axis not in reduced
            ),
        )
        for source_component in source_components
    )
    builtin_operator = {
        "reduce.max": BinaryOperator.MAXIMUM,
        "reduce.sum": BinaryOperator.ADD,
        "reduce.any": BinaryOperator.LOGICAL_OR,
        "reduce.all": BinaryOperator.LOGICAL_AND,
    }.get(name)
    identity = (
        _builtin_identity(lowerer, source_components[0].type.dtype, builtin_operator, node)
        if builtin_operator is not None
        else _lower_component_identity(
            lowerer, bound["identity"], source_components, component_names, node
        )
    )
    identity_components, identity_names = _identity_components(lowerer, identity, node)
    if component_names != identity_names or len(source_components) != len(identity_components):
        lowerer.error(node, "reduce source and identity schemas must match")
    for source_component, expected, identity_component in zip(
        source_components, result_components, identity_components
    ):
        scalar_identity = ScalarType(source_component.type.dtype)
        if identity_component.type not in (scalar_identity, expected):
            lowerer.error(
                node,
                "reduce identity must be an element scalar or the complete accumulator result",
            )

    captures, capture_bindings = _combine_captures(
        lowerer, bound.get("combine_operands"), node
    )
    accumulator_types = tuple(value.type for value in identity_components)
    if builtin_operator is not None:
        combine_region = _builtin_combine_region(
            lowerer, accumulator_types, builtin_operator, node
        )
    else:
        combine_region, combine_results = _combine_region(
            lowerer,
            bound["combine"],
            accumulator_types,
            captures,
            capture_bindings,
            component_names,
            node,
        )
        if combine_results != accumulator_types:
            lowerer.error(node, "reduce combine result schema must match identity")

    operation = lowerer.emit(
        OperationKind.REDUCE,
        lowerer.location(node),
        operands=source_components + identity_components + captures,
        result_types=result_components,
        attributes={
            "axes": axes,
            "source_count": len(source_components),
            "identity_count": len(identity_components),
            "capture_count": len(captures),
        },
        regions=(combine_region,),
    )
    return _rebuild_components(
        lowerer, operation.results, component_names, node
    )


def _arg_reduce_max(lowerer: FunctionLowerer, node: ast.Call) -> StaticTuple:
    bound = bind_call(
        lowerer,
        node,
        ("value", "axis", "identity", "acc_dtype"),
        required=("value", "axis", "identity"),
    )
    source = lowerer.read_value(
        lowerer.lower_expression(bound["value"]), bound["value"]
    )
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
    if source.type.dtype != acc_dtype:
        source = lowerer.emit(
            OperationKind.CAST,
            lowerer.location(node),
            operands=(source,),
            result_types=(TensorType(acc_dtype, source.type.shape),),
        ).results[0]
    identity = lowerer.materialize(
        lowerer.lower_expression(bound["identity"]),
        bound["identity"],
        ScalarType(acc_dtype),
    )
    indices = lowerer.emit(
        OperationKind.INDICES,
        lowerer.location(node),
        operands=(source,),
        result_types=(TensorType(intent_index, source.type.shape),),
        attributes={"tensor_axis": axes[0]},
    ).results[0]
    indices = lowerer.emit(
        OperationKind.CAST,
        lowerer.location(node),
        operands=(indices,),
        result_types=(TensorType(i32, source.type.shape),),
    ).results[0]
    index_identity = lowerer.emit_literal(2147483647, node, ScalarType(i32))
    combine = _argmax_combine_region(lowerer, ScalarType(acc_dtype), node)
    result_shape = tuple(
        dimension
        for axis, dimension in enumerate(source.type.shape)
        if axis != axes[0]
    )
    operation = lowerer.emit(
        OperationKind.REDUCE,
        lowerer.location(node),
        operands=(source, indices, identity, index_identity),
        result_types=(
            lowerer.value_result_type(acc_dtype, result_shape),
            lowerer.value_result_type(i32, result_shape),
        ),
        attributes={
            "axes": axes,
            "source_count": 2,
            "identity_count": 2,
            "capture_count": 0,
        },
        regions=(combine,),
    )
    return StaticTuple(operation.results)


def _scan(lowerer: FunctionLowerer, node: ast.Call, name: str) -> MlirValue:
    bound = (
        bind_call(
            lowerer,
            node,
            (
                "value", "axis", "identity", "combine", "combine_operands",
                "inclusive", "reverse", "acc_dtype",
            ),
            required=("value", "axis", "identity", "combine", "inclusive"),
        )
        if name == "scan"
        else bind_declared_call(lowerer, node, name)
    )
    source = lowerer.read_value(
        lowerer.lower_expression(bound["value"]), bound["value"]
    )
    source_components, component_names = _source_components(lowerer, source, node)
    if not source_components or any(
        not isinstance(value.type, TensorType) for value in source_components
    ):
        lowerer.error(node, "I.scan source components must be tensors")
    if name != "scan" and (
        component_names != ("value",) or source_components[0].type.dtype == intent_bool
    ):
        lowerer.error(node, f"I.{name} requires one numeric tensor")
    axes = normalize_axes(
        lowerer,
        require_axes(lowerer, bound["axis"]),
        source_components[0].type.rank,
        node,
    )
    if len(axes) != 1:
        lowerer.error(node, "I.scan requires exactly one axis")
    acc_dtype = optional_dtype(lowerer, bound.get("acc_dtype"))
    if acc_dtype is None and name == "cumsum":
        acc_dtype = _sum_accumulator_dtype(source_components[0].type.dtype)
    if acc_dtype is not None:
        if len(source_components) != 1:
            lowerer.error(node, "record scan uses explicit field dtypes")
        if source_components[0].type.dtype != acc_dtype:
            source_components = (
                lowerer.emit(
                    OperationKind.CAST,
                    lowerer.location(node),
                    operands=(source_components[0],),
                    result_types=(
                        TensorType(acc_dtype, source_components[0].type.shape),
                    ),
                ).results[0],
            )
    builtin_operator = {
        "cumsum": BinaryOperator.ADD,
        "cummax": BinaryOperator.MAXIMUM,
    }.get(name)
    identity = (
        _builtin_identity(lowerer, source_components[0].type.dtype, builtin_operator, node)
        if builtin_operator is not None
        else _lower_component_identity(
            lowerer, bound["identity"], source_components, component_names, node
        )
    )
    identity_components, identity_names = _identity_components(lowerer, identity, node)
    if component_names != identity_names or len(source_components) != len(identity_components):
        lowerer.error(node, "scan source and identity schemas must match")
    for source_component, identity_component in zip(source_components, identity_components):
        scalar_identity = ScalarType(source_component.type.dtype)
        slice_identity = lowerer.value_result_type(
            source_component.type.dtype,
            tuple(
                dimension
                for source_axis, dimension in enumerate(source_component.type.shape)
                if source_axis != axes[0]
            ),
        )
        if identity_component.type not in (scalar_identity, slice_identity):
            lowerer.error(
                node,
                "scan identity must be an element scalar or one source-axis slice",
            )
    captures, capture_bindings = _combine_captures(
        lowerer, bound.get("combine_operands"), node
    )
    accumulator_types = tuple(value.type for value in identity_components)
    if builtin_operator is not None:
        combine_region = _builtin_combine_region(
            lowerer, accumulator_types, builtin_operator, node
        )
    else:
        combine_region, combine_results = _combine_region(
            lowerer, bound["combine"], accumulator_types, captures,
            capture_bindings, component_names, node,
        )
        if combine_results != accumulator_types:
            lowerer.error(node, "scan combine result schema must match identity")
    operation = lowerer.emit(
        OperationKind.SCAN,
        lowerer.location(node),
        operands=source_components + identity_components + captures,
        result_types=tuple(value.type for value in source_components),
        attributes={
            "axis": axes[0],
            "inclusive": require_static_bool(lowerer, bound["inclusive"]),
            "reverse": (
                require_static_bool(lowerer, bound["reverse"])
                if "reverse" in bound
                else False
            ),
            "source_count": len(source_components),
            "identity_count": len(identity_components),
            "capture_count": len(captures),
        },
        regions=(combine_region,),
    )
    return _rebuild_components(lowerer, operation.results, component_names, node)


def _region_fold(lowerer: FunctionLowerer, node: ast.Call) -> object:
    bound = bind_call(
        lowerer,
        node,
        ("source", "axis", "summarize", "combine", "identity", "operands"),
        required=("source", "axis", "summarize", "combine", "identity"),
    )
    sources = _tensor_sources(lowerer, bound["source"], node)
    axis = _shared_source_axis(lowerer, sources, bound["axis"], node)
    captures, capture_bindings = _region_captures(
        lowerer, bound.get("operands"), node
    )
    slice_types = _slice_types(lowerer, sources, axis, "region_fold")
    summarize, summary_types = _helper_region(
        lowerer,
        bound["summarize"],
        slice_types + tuple(value.type for value in captures),
        (None,) * len(slice_types) + capture_bindings,
        node,
    )
    identity_values = _values(lowerer, bound["identity"], node)
    if len(identity_values) != len(summary_types):
        lowerer.error(node, "region_fold identity arity must match summary schema")
    identity = tuple(
        _align_value_schema(lowerer, value, target, bound["identity"])
        for value, target in zip(identity_values, summary_types)
    )
    identity_types = tuple(value.type for value in identity)
    if summary_types != identity_types:
        lowerer.error(
            node,
            f"region_fold summarize result {summary_types} must match identity schema {identity_types}",
        )
    combine, combine_types = _helper_region(
        lowerer,
        bound["combine"],
        summary_types + summary_types,
        (None,) * (2 * len(summary_types)),
        node,
    )
    if combine_types != summary_types:
        lowerer.error(node, "region_fold combine result must match summary schema")
    operation = lowerer.emit(
        OperationKind.REGION_FOLD,
        lowerer.location(node),
        operands=sources + identity + captures,
        result_types=summary_types,
        attributes={
            "axis": axis,
            "source_count": len(sources),
            "identity_count": len(identity),
            "capture_count": len(captures),
        },
        regions=(summarize, combine),
    )
    return operation.results[0] if len(operation.results) == 1 else StaticTuple(operation.results)


def _region_scan(lowerer: FunctionLowerer, node: ast.Call) -> StaticTuple:
    bound = bind_call(
        lowerer,
        node,
        (
            "source",
            "axis",
            "summarize",
            "combine",
            "identity",
            "initial_state",
            "apply",
            "emit",
            "operands",
        ),
        required=(
            "source",
            "axis",
            "summarize",
            "combine",
            "identity",
            "initial_state",
            "apply",
            "emit",
        ),
    )
    sources = _tensor_sources(lowerer, bound["source"], node)
    axis = _shared_source_axis(lowerer, sources, bound["axis"], node)
    captures, capture_bindings = _region_captures(
        lowerer, bound.get("operands"), node
    )
    slice_types = _slice_types(lowerer, sources, axis, "region_scan")
    summarize, transition_types = _helper_region(
        lowerer,
        bound["summarize"],
        slice_types + tuple(value.type for value in captures),
        (None,) * len(slice_types) + capture_bindings,
        node,
    )
    identity_values = _values(lowerer, bound["identity"], node)
    if len(identity_values) != len(transition_types):
        lowerer.error(node, "region_scan identity arity must match transition schema")
    identity = tuple(
        _align_value_schema(lowerer, value, target, bound["identity"])
        for value, target in zip(identity_values, transition_types)
    )
    if tuple(value.type for value in identity) != transition_types:
        lowerer.error(node, "region_scan summarize result must match identity schema")
    combine, combine_types = _helper_region(
        lowerer,
        bound["combine"],
        transition_types + transition_types,
        (None,) * (2 * len(transition_types)),
        node,
    )
    if combine_types != transition_types:
        lowerer.error(node, "region_scan combine result must match transition schema")
    initial = _values(lowerer, bound["initial_state"], node)
    initial_types = tuple(value.type for value in initial)
    apply, state_types = _helper_region(
        lowerer,
        bound["apply"],
        transition_types + initial_types,
        (None,) * (len(transition_types) + len(initial_types)),
        node,
    )
    if state_types != initial_types:
        lowerer.error(node, "region_scan apply result must match initial-state schema")
    emit, slice_output_types = _helper_region(
        lowerer,
        bound["emit"],
        slice_types + state_types + tuple(value.type for value in captures),
        (None,) * (len(slice_types) + len(state_types)) + capture_bindings,
        node,
    )
    source_extent = sources[0].type.shape[axis]
    output_types = tuple(
        _replace_slice_extent(value_type, slice_types[0].shape[axis], source_extent)
        for value_type in slice_output_types
    )
    operation = lowerer.emit(
        OperationKind.REGION_SCAN,
        lowerer.location(node),
        operands=sources + identity + initial + captures,
        result_types=output_types + state_types,
        attributes={
            "axis": axis,
            "source_count": len(sources),
            "identity_count": len(identity),
            "state_count": len(initial),
            "capture_count": len(captures),
            "output_count": len(output_types),
        },
        regions=(summarize, combine, apply, emit),
    )
    outputs = operation.results[: len(output_types)]
    states = operation.results[len(output_types) :]
    output: object = outputs[0] if len(outputs) == 1 else StaticTuple(outputs)
    state: object = states[0] if len(states) == 1 else StaticTuple(states)
    return StaticTuple((output, state))


def _contract(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("lhs", "rhs", "reduce", "acc_dtype", "batch"),
        required=("lhs", "rhs", "reduce", "acc_dtype"),
    )
    lhs = lowerer.read_value(lowerer.lower_expression(bound["lhs"]), bound["lhs"])
    rhs = lowerer.read_value(lowerer.lower_expression(bound["rhs"]), bound["rhs"])
    if not isinstance(lhs.type, TensorType) or not isinstance(rhs.type, TensorType):
        lowerer.error(node, "I.contract operands must be tensors")
    return emit_contract(
        lowerer, lhs, rhs,
        _axis_pairs(lowerer, bound["reduce"], "reduction"),
        _axis_pairs(lowerer, bound["batch"], "batch") if "batch" in bound else (),
        require_dtype(lowerer, bound["acc_dtype"]), node,
    )


def emit_contract(lowerer, lhs, rhs, reduce, batch, acc_dtype, node):
    if lhs.type.dtype != rhs.type.dtype or lhs.type.dtype == intent_bool:
        lowerer.error(node, "contraction requires matching numeric input dtypes; cast explicitly")
    reduce, batch, result_shape = _paired_relations(lowerer, lhs, rhs, reduce, batch, node)
    return lowerer.emit(
        OperationKind.CONTRACT,
        lowerer.location(node),
        operands=(lhs, rhs),
        result_types=(TensorType(acc_dtype, result_shape),),
        attributes={"reduce": reduce, "batch": batch},
    ).results[0]


def _scaled_contract(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        (
            "lhs",
            "lhs_scale",
            "rhs",
            "rhs_scale",
            "lhs_format",
            "rhs_format",
            "lhs_group_size",
            "rhs_group_size",
            "reduce",
            "batch",
            "acc_dtype",
        ),
        required=(
            "lhs",
            "lhs_scale",
            "rhs",
            "rhs_scale",
            "lhs_format",
            "rhs_format",
            "lhs_group_size",
            "rhs_group_size",
            "reduce",
            "acc_dtype",
        ),
    )
    lhs = lowerer.read_value(lowerer.lower_expression(bound["lhs"]), bound["lhs"])
    lhs_scale = lowerer.read_value(
        lowerer.lower_expression(bound["lhs_scale"]), bound["lhs_scale"]
    )
    rhs = lowerer.read_value(lowerer.lower_expression(bound["rhs"]), bound["rhs"])
    rhs_scale = lowerer.read_value(
        lowerer.lower_expression(bound["rhs_scale"]), bound["rhs_scale"]
    )
    if any(
        not isinstance(value.type, TensorType)
        for value in (lhs, lhs_scale, rhs, rhs_scale)
    ):
        lowerer.error(node, "I.scaled_contract operands and scales must be tensors")
    lhs_format = _scaled_format(lowerer, bound["lhs_format"])
    rhs_format = _scaled_format(lowerer, bound["rhs_format"])
    lhs_group = require_static_int(lowerer, bound["lhs_group_size"])
    rhs_group = require_static_int(lowerer, bound["rhs_group_size"])
    reduce = _axis_pairs(lowerer, bound["reduce"], "reduction")
    batch = _axis_pairs(lowerer, bound["batch"], "batch") if "batch" in bound else ()
    return emit_scaled_contract(
        lowerer, lhs, lhs_scale, rhs, rhs_scale, lhs_format, rhs_format,
        lhs_group, rhs_group, reduce, batch, require_dtype(lowerer, bound["acc_dtype"]), node,
    )


def emit_scaled_contract(
    lowerer, lhs, lhs_scale, rhs, rhs_scale, lhs_format, rhs_format,
    lhs_group, rhs_group, reduce, batch, acc_dtype, node,
):
    if lhs_group <= 0 or lhs_group != rhs_group:
        lowerer.error(node, "scaled-contract requires one equal positive group size")
    reduce = tuple(
        (
            normalize_axes(lowerer, (left,), lhs.type.rank, node)[0],
            normalize_axes(lowerer, (right,), rhs.type.rank, node)[0],
        )
        for left, right in reduce
    )
    if (
        lhs.type.rank != 3
        or lhs_scale.type.rank != 2
        or rhs.type.rank != 3
        or rhs_scale.type.rank != 2
        or reduce != ((1, 0), (2, 1))
        or batch
    ):
        lowerer.error(
            node,
            "scaled-contract requires lhs/lhs-scale [M,G,C]/[M,G], "
            "rhs/rhs-scale [G,C,N]/[N,G], reduce=((1,0),(2,1)), and no batch axes",
        )
    lhs_carrier = lhs_group // (2 if lhs_format is ScaledFormatKind.E2M1 else 1)
    rhs_carrier = rhs_group // (2 if rhs_format is ScaledFormatKind.E2M1 else 1)
    if (
        lhs_group % (2 if lhs_format is ScaledFormatKind.E2M1 else 1)
        or rhs_group % (2 if rhs_format is ScaledFormatKind.E2M1 else 1)
        or lhs.type.shape[2] != StaticDim(lhs_carrier)
        or rhs.type.shape[1] != StaticDim(rhs_carrier)
        or not dims_compatible(lhs.type.shape[0], lhs_scale.type.shape[0])
        or not dims_compatible(lhs.type.shape[1], lhs_scale.type.shape[1])
        or not dims_compatible(lhs.type.shape[1], rhs.type.shape[0])
        or not dims_compatible(lhs.type.shape[1], rhs_scale.type.shape[1])
        or not dims_compatible(rhs.type.shape[2], rhs_scale.type.shape[0])
    ):
        lowerer.error(node, "scaled-contract operands violate the closed scale-axis relation")
    result_shape = (lhs.type.shape[0], rhs.type.shape[2])
    return lowerer.emit(
        OperationKind.SCALED_CONTRACT,
        lowerer.location(node),
        operands=(lhs, lhs_scale, rhs, rhs_scale),
        result_types=(TensorType(acc_dtype, result_shape),),
        attributes={
            "reduce": reduce,
            "batch": batch,
            "lhs_format": lhs_format,
            "rhs_format": rhs_format,
            "lhs_group_size": lhs_group,
            "rhs_group_size": rhs_group,
        },
    ).results[0]


def _sparse_format(
    lowerer: FunctionLowerer,
    name: str,
    node: ast.Call,
) -> SparseFormatSpec:
    bound = bind_call(
        lowerer,
        node,
        ("compression_axis", "logical_extent"),
        required=("compression_axis", "logical_extent"),
    )
    axis = require_static_int(lowerer, bound["compression_axis"])
    extent = lowerer.materialize(
        lowerer.lower_expression(bound["logical_extent"]),
        bound["logical_extent"],
        ScalarType(intent_index),
    )
    kind = {"sparse.one_of_two": 0, "sparse.two_of_four": 1}[name]
    return SparseFormatSpec(kind, axis, extent)


def _sparse_contract(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("compressed", "metadata", "dense_rhs", "format", "reduce", "batch", "acc_dtype"),
        required=("compressed", "metadata", "dense_rhs", "format", "reduce", "acc_dtype"),
    )
    compressed = lowerer.read_value(
        lowerer.lower_expression(bound["compressed"]), bound["compressed"]
    )
    metadata = lowerer.read_value(
        lowerer.lower_expression(bound["metadata"]), bound["metadata"]
    )
    rhs = lowerer.read_value(
        lowerer.lower_expression(bound["dense_rhs"]), bound["dense_rhs"]
    )
    format_value = lowerer.lower_expression(bound["format"])
    if not isinstance(format_value, SparseFormatSpec):
        lowerer.error(bound["format"], "sparse contract format must be a typed I.sparse schema")
    if not isinstance(compressed.type, TensorType) or not isinstance(rhs.type, TensorType):
        lowerer.error(node, "sparse contract data operands must be tensors")
    return emit_sparse_contract(
        lowerer, compressed, metadata, rhs, format_value,
        _axis_pairs(lowerer, bound["reduce"], "reduction"),
        _axis_pairs(lowerer, bound["batch"], "batch") if "batch" in bound else (),
        require_dtype(lowerer, bound["acc_dtype"]), node,
    )


def emit_sparse_contract(lowerer, compressed, metadata, rhs, format_value, reduce, batch, acc_dtype, node):
    reduce, batch, result_shape = _paired_relations(
        lowerer, compressed, rhs, reduce, batch, node,
        logical_lhs_axis=format_value.compression_axis,
    )
    return lowerer.emit(
        OperationKind.SPARSE_CONTRACT,
        lowerer.location(node),
        operands=(compressed, metadata, rhs, format_value.logical_extent),
        result_types=(TensorType(acc_dtype, result_shape),),
        attributes={
            "format": SparseFormatAttribute(
                format_value.kind, format_value.compression_axis
            ),
            "reduce": reduce,
            "batch": batch,
        },
    ).results[0]


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
        lowerer.lower_expression(bound["metadata"]), bound["metadata"]
    )
    rhs = lowerer.read_value(lowerer.lower_expression(bound["rhs"]), bound["rhs"])
    if (
        not isinstance(compressed.type, TensorType)
        or not isinstance(rhs.type, TensorType)
        or compressed.type.rank != 2
        or rhs.type.rank != 2
    ):
        lowerer.error(node, "I.sparse_contract_2to4 requires rank-two data operands")
    logical_extent = lowerer.materialize_dimension(
        ShapeDimension(rhs.type.shape[0], rhs, 0), bound["rhs"]
    )
    return lowerer.emit(
        OperationKind.SPARSE_CONTRACT,
        lowerer.location(node),
        operands=(compressed, metadata, rhs, logical_extent),
        result_types=(
            TensorType(
                require_dtype(lowerer, bound["acc_dtype"]),
                (compressed.type.shape[0], rhs.type.shape[1]),
            ),
        ),
        attributes={
            "format": SparseFormatAttribute(1, 1),
            "reduce": ((1, 0),),
            "batch": (),
        },
    ).results[0]


def _histogram(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("values", "bins", "valid", "count_dtype"),
        required=("values", "bins", "valid", "count_dtype"),
    )
    values = lowerer.read_value(
        lowerer.lower_expression(bound["values"]), bound["values"]
    )
    if not isinstance(values.type, TensorType) or values.type.dtype.category not in (
        DTypeCategory.SIGNED_INTEGER,
        DTypeCategory.UNSIGNED_INTEGER,
        DTypeCategory.INDEX,
    ):
        lowerer.error(node, "I.histogram values must be an integer tensor")
    bins_expression = lowerer.lower_expression(bound["bins"])
    bins = lowerer.materialize(
        bins_expression,
        bound["bins"],
        ScalarType(intent_index),
    )
    valid = lowerer.materialize(
        lowerer.lower_expression(bound["valid"]), bound["valid"]
    )
    valid_dtype, _ = lowerer.dtype_and_shape(valid.type, node)
    if valid_dtype != intent_bool:
        lowerer.error(node, "I.histogram valid must be bool")
    valid = lowerer.broadcast_value(valid, values.type.shape, node)
    count_dtype = require_dtype(lowerer, bound["count_dtype"])
    if count_dtype.category not in (
        DTypeCategory.SIGNED_INTEGER,
        DTypeCategory.UNSIGNED_INTEGER,
    ):
        lowerer.error(node, "I.histogram count dtype must be fixed-width integer")
    result_extent = (
        bins_expression.dimension
        if isinstance(bins_expression, ShapeDimension)
        else lowerer.fresh_dynamic_dimension("histogram_bins")
    )
    known, static_bins = compile_time_value(bins_expression)
    if known:
        if isinstance(static_bins, bool) or not isinstance(static_bins, int) or static_bins <= 0:
            lowerer.error(bound["bins"], "histogram bins must be positive")
        result_extent = StaticDim(static_bins)
    return lowerer.emit(
        OperationKind.HISTOGRAM,
        lowerer.location(node),
        operands=(values, bins, valid),
        result_types=(TensorType(count_dtype, (result_extent,)),),
        attributes={
            "result_dimension": lowerer.compiler.builder.dimension_id(result_extent)
        },
    ).results[0]


def _helper_region(
    lowerer: FunctionLowerer,
    helper_node: ast.AST,
    argument_types: tuple[ValueType, ...],
    static_bindings: tuple[ConstexprBinding | None, ...],
    call_node: ast.AST,
) -> tuple[object, tuple[ValueType, ...]]:
    helper = lowerer.lower_expression(helper_node)
    if not isinstance(helper, HelperDefinition):
        lowerer.error(helper_node, "structured callable must be an @intent.fn")
    if len(static_bindings) != len(argument_types):
        raise ValueError("helper static-binding schema mismatch")
    region = lowerer.make_region(lowerer.location(call_node), argument_types)
    arguments: list[Expression] = []
    for value, binding in zip(region.blocks[0].arguments, static_bindings):
        arguments.append(
            ConstexprBinding(binding.python_value, value)
            if binding is not None
            else value
        )
    saved_block = lowerer.current_block
    lowerer.current_block = region.blocks[0]
    results = lowerer.compiler.lower_helper_inline(
        lowerer,
        helper,
        tuple(arguments),
        lowerer.location(call_node),
    )
    if region.effects:
        lowerer.error(call_node, "structured callable regions must be pure")
    lowerer.emit(OperationKind.YIELD, lowerer.location(call_node), operands=results)
    lowerer.current_block = saved_block
    return region, tuple(value.type for value in results)


def _combine_region(
    lowerer: FunctionLowerer,
    helper_node: ast.AST,
    accumulator_types: tuple[ValueType, ...],
    captures: tuple[MlirValue, ...],
    capture_bindings: tuple[ConstexprBinding | None, ...],
    component_names: tuple[str, ...],
    node: ast.AST,
) -> tuple[object, tuple[ValueType, ...]]:
    expression = lowerer.lower_expression(helper_node)
    if isinstance(expression, Intrinsic):
        if captures:
            lowerer.error(node, "built-in combine does not accept captures")
        operator = {
            "add": BinaryOperator.ADD,
            "maximum": BinaryOperator.MAXIMUM,
            "minimum": BinaryOperator.MINIMUM,
        }.get(expression.name)
        if operator is None:
            lowerer.error(helper_node, "unsupported built-in combine")
        return (
            _builtin_combine_region(lowerer, accumulator_types, operator, node),
            accumulator_types,
        )
    if not isinstance(expression, HelperDefinition):
        lowerer.error(helper_node, "combine must be an Intent intrinsic or @intent.fn")
    region = lowerer.make_region(
        lowerer.location(node),
        accumulator_types + accumulator_types + tuple(value.type for value in captures),
    )
    saved_block = lowerer.current_block
    lowerer.current_block = region.blocks[0]
    block_arguments = region.blocks[0].arguments
    arguments: list[Expression]
    product_kind = _component_product_kind(component_names)
    if product_kind == "scalar":
        arguments = list(block_arguments)
    else:
        count = len(accumulator_types)
        lhs = _make_product_value(
            lowerer, block_arguments[:count], component_names, product_kind, node
        )
        rhs = _make_product_value(
            lowerer,
            block_arguments[count : 2 * count],
            component_names,
            product_kind,
            node,
        )
        arguments = [lhs, rhs, *block_arguments[2 * count :]]
    capture_offset = 2 * len(accumulator_types)
    for index, binding in enumerate(capture_bindings):
        if binding is not None:
            argument_index = (
                capture_offset + index
                if product_kind == "scalar"
                else 2 + index
            )
            arguments[argument_index] = ConstexprBinding(
                binding.python_value,
                region.blocks[0].arguments[capture_offset + index],
            )
    helper_results = lowerer.compiler.lower_helper_inline(
        lowerer,
        expression,
        tuple(arguments),
        lowerer.location(node),
    )
    if product_kind == "scalar":
        results = helper_results
    else:
        if len(helper_results) != 1:
            lowerer.error(node, "product combine must return one typed product")
        results, result_names = _source_components(lowerer, helper_results[0], node)
        if result_names != component_names:
            lowerer.error(node, "product combine result schema does not match accumulator")
    if region.effects:
        lowerer.error(node, "combine region must be pure")
    lowerer.emit(OperationKind.YIELD, lowerer.location(node), operands=results)
    lowerer.current_block = saved_block
    return region, tuple(value.type for value in results)


def _lower_component_identity(
    lowerer: FunctionLowerer,
    identity_node: ast.AST,
    source_components: tuple[MlirValue, ...],
    component_names: tuple[str, ...],
    call_node: ast.AST,
) -> MlirValue:
    if component_names != ("value",) and isinstance(identity_node, ast.Call):
        callee = lowerer.lower_expression(identity_node.func)
        if isinstance(callee, Intrinsic) and callee.name == "record":
            if identity_node.args or any(
                keyword.arg is None for keyword in identity_node.keywords
            ):
                lowerer.error(identity_node, "record identity requires named fields")
            names = tuple(keyword.arg for keyword in identity_node.keywords)
            if names != component_names:
                lowerer.error(
                    identity_node,
                    "record identity fields must match source record order",
                )
            values: list[MlirValue] = []
            for keyword, source_component in zip(
                identity_node.keywords, source_components
            ):
                expression = lowerer.lower_expression(keyword.value)
                expected = (
                    ScalarType(source_component.type.dtype)
                    if isinstance(expression, Literal)
                    and isinstance(source_component.type, TensorType)
                    else None
                )
                values.append(
                    lowerer.materialize(expression, keyword.value, expected)
                )
            return _make_record_value(lowerer, tuple(values), component_names, call_node)
    expression = lowerer.lower_expression(identity_node)
    expected = (
        ScalarType(source_components[0].type.dtype)
        if len(source_components) == 1
        and isinstance(source_components[0].type, TensorType)
        and isinstance(expression, Literal)
        else None
    )
    return lowerer.materialize(expression, identity_node, expected)


def _make_record_value(
    lowerer: FunctionLowerer,
    values: tuple[MlirValue, ...] | list[MlirValue],
    names: tuple[str, ...],
    node: ast.AST,
) -> MlirValue:
    value_tuple = tuple(values)
    result_type = RecordType(
        tuple((name, value.type) for name, value in zip(names, value_tuple))
    )
    return lowerer.emit(
        OperationKind.MAKE_RECORD,
        lowerer.location(node),
        operands=value_tuple,
        result_types=(result_type,),
    ).results[0]


def _make_tuple_value(
    lowerer: FunctionLowerer,
    values: tuple[MlirValue, ...] | list[MlirValue],
    node: ast.AST,
) -> MlirValue:
    value_tuple = tuple(values)
    return lowerer.emit(
        OperationKind.MAKE_TUPLE,
        lowerer.location(node),
        operands=value_tuple,
        result_types=(TupleType(tuple(value.type for value in value_tuple)),),
    ).results[0]


def _make_product_value(
    lowerer: FunctionLowerer,
    values: tuple[MlirValue, ...] | list[MlirValue],
    names: tuple[str, ...],
    product_kind: str,
    node: ast.AST,
) -> MlirValue:
    if product_kind == "tuple":
        return _make_tuple_value(lowerer, values, node)
    if product_kind == "record":
        return _make_record_value(lowerer, values, names, node)
    raise ValueError("scalar components do not form a product value")


def _builtin_combine_region(
    lowerer: FunctionLowerer,
    accumulator_types: tuple[ValueType, ...],
    operator: BinaryOperator,
    node: ast.AST,
):
    region = lowerer.make_region(
        lowerer.location(node), accumulator_types + accumulator_types
    )
    saved_block = lowerer.current_block
    lowerer.current_block = region.blocks[0]
    count = len(accumulator_types)
    results = tuple(
        lowerer.emit(
            OperationKind.BINARY,
            lowerer.location(node),
            operands=(region.blocks[0].arguments[index], region.blocks[0].arguments[count + index]),
            result_types=(accumulator_types[index],),
            attributes={"operator_kind": operator},
        ).results[0]
        for index in range(count)
    )
    lowerer.emit(OperationKind.YIELD, lowerer.location(node), operands=results)
    lowerer.current_block = saved_block
    return region


def _argmax_combine_region(
    lowerer: FunctionLowerer,
    value_type: ValueType,
    node: ast.AST,
):
    types = (value_type, ScalarType(i32))
    region = lowerer.make_region(lowerer.location(node), types + types)
    saved = lowerer.current_block
    lowerer.current_block = region.blocks[0]
    lhs_value, lhs_index, rhs_value, rhs_index = region.blocks[0].arguments
    greater = lowerer.emit(
        OperationKind.COMPARE,
        lowerer.location(node),
        operands=(lhs_value, rhs_value),
        result_types=(ScalarType(intent_bool),),
        attributes={"predicate": ComparePredicate.GT},
    ).results[0]
    equal = lowerer.emit(
        OperationKind.COMPARE,
        lowerer.location(node),
        operands=(lhs_value, rhs_value),
        result_types=(ScalarType(intent_bool),),
        attributes={"predicate": ComparePredicate.EQ},
    ).results[0]
    lower = lowerer.emit(
        OperationKind.COMPARE,
        lowerer.location(node),
        operands=(lhs_index, rhs_index),
        result_types=(ScalarType(intent_bool),),
        attributes={"predicate": ComparePredicate.LE},
    ).results[0]
    tied = lowerer.emit(
        OperationKind.BINARY,
        lowerer.location(node),
        operands=(equal, lower),
        result_types=(ScalarType(intent_bool),),
        attributes={"operator_kind": BinaryOperator.LOGICAL_AND},
    ).results[0]
    choose = lowerer.emit(
        OperationKind.BINARY,
        lowerer.location(node),
        operands=(greater, tied),
        result_types=(ScalarType(intent_bool),),
        attributes={"operator_kind": BinaryOperator.LOGICAL_OR},
    ).results[0]
    selected = tuple(
        lowerer.emit(
            OperationKind.SELECT,
            lowerer.location(node),
            operands=(choose, lhs, rhs),
            result_types=(result_type,),
        ).results[0]
        for lhs, rhs, result_type in zip(
            (lhs_value, lhs_index), (rhs_value, rhs_index), types
        )
    )
    lowerer.emit(OperationKind.YIELD, lowerer.location(node), operands=selected)
    lowerer.current_block = saved
    return region


def _source_components(
    lowerer: FunctionLowerer,
    source: MlirValue,
    node: ast.AST,
) -> tuple[tuple[MlirValue, ...], tuple[str, ...]]:
    if isinstance(source.type, TupleType):
        values = tuple(
            lowerer.emit(
                OperationKind.EXTRACT,
                lowerer.location(node),
                operands=(source,),
                result_types=(component_type,),
                attributes={"field": index},
            ).results[0]
            for index, component_type in enumerate(source.type.components)
        )
        return values, tuple(f"#{index}" for index in range(len(values)))
    if not isinstance(source.type, RecordType):
        return (source,), ("value",)
    values = tuple(
        lowerer.emit(
            OperationKind.EXTRACT,
            lowerer.location(node),
            operands=(source,),
            result_types=(field_type,),
            attributes={"field": index},
        ).results[0]
        for index, (_, field_type) in enumerate(source.type.fields)
    )
    return values, tuple(name for name, _ in source.type.fields)


def _identity_components(
    lowerer: FunctionLowerer,
    identity: MlirValue,
    node: ast.AST,
) -> tuple[tuple[MlirValue, ...], tuple[str, ...]]:
    return _source_components(lowerer, identity, node)


def _rebuild_components(
    lowerer: FunctionLowerer,
    values: tuple[MlirValue, ...],
    names: tuple[str, ...],
    node: ast.AST,
) -> MlirValue:
    if names == ("value",):
        return values[0]
    if _component_product_kind(names) == "tuple":
        return _make_tuple_value(lowerer, values, node)
    result_type = RecordType(tuple(zip(names, (value.type for value in values))))
    return lowerer.emit(
        OperationKind.MAKE_RECORD,
        lowerer.location(node),
        operands=values,
        result_types=(result_type,),
    ).results[0]


def _component_product_kind(names: tuple[str, ...]) -> str:
    if names == ("value",):
        return "scalar"
    if names == tuple(f"#{index}" for index in range(len(names))):
        return "tuple"
    return "record"


def _values(
    lowerer: FunctionLowerer,
    node: ast.AST,
    call_node: ast.AST,
) -> tuple[MlirValue, ...]:
    expression = lowerer.lower_expression(node)
    elements = expression.elements if isinstance(expression, StaticTuple) else (expression,)
    return tuple(lowerer.materialize(element, call_node) for element in elements)


def _align_value_schema(
    lowerer: FunctionLowerer,
    value: MlirValue,
    target: ValueType,
    node: ast.AST,
) -> MlirValue:
    if isinstance(value.type, TensorType) and isinstance(target, TensorType):
        if (
            value.type.dtype != target.dtype
            or value.type.rank != target.rank
            or any(
                not dims_compatible(source, destination)
                for source, destination in zip(value.type.shape, target.shape)
            )
        ):
            lowerer.error(node, "structured identity tensor does not match its summary schema")
        return lowerer.broadcast_value(value, target.shape, node)
    if isinstance(value.type, RecordType) and isinstance(target, RecordType):
        source_names = tuple(name for name, _ in value.type.fields)
        target_names = tuple(name for name, _ in target.fields)
        if source_names != target_names:
            lowerer.error(node, "structured identity record fields do not match its summary schema")
        components, _ = _source_components(lowerer, value, node)
        aligned = tuple(
            _align_value_schema(lowerer, component, field_type, node)
            for component, (_, field_type) in zip(components, target.fields)
        )
        return _make_record_value(lowerer, aligned, target_names, node)
    if isinstance(value.type, TupleType) and isinstance(target, TupleType):
        if len(value.type.components) != len(target.components):
            lowerer.error(node, "structured identity tuple arity does not match its summary schema")
        components, _ = _source_components(lowerer, value, node)
        aligned = tuple(
            _align_value_schema(lowerer, component, component_type, node)
            for component, component_type in zip(components, target.components)
        )
        return _make_tuple_value(lowerer, aligned, node)
    if value.type == target:
        return value
    lowerer.error(
        node,
        f"structured identity value {value.type} does not match summary schema {target}",
    )
    raise AssertionError("unreachable after frontend diagnostic")


def _tensor_sources(
    lowerer: FunctionLowerer,
    node: ast.AST,
    call_node: ast.AST,
) -> tuple[MlirValue, ...]:
    expression = lowerer.lower_expression(node)
    elements = expression.elements if isinstance(expression, StaticTuple) else (expression,)
    values = tuple(lowerer.read_value(element, call_node) for element in elements)
    if not values or any(not isinstance(value.type, TensorType) for value in values):
        lowerer.error(call_node, "region source must contain one or more tensors")
    return values


def _shared_source_axis(
    lowerer: FunctionLowerer,
    sources: tuple[MlirValue, ...],
    axis_node: ast.AST,
    call_node: ast.AST,
) -> int:
    axis = require_static_int(lowerer, axis_node)
    rank = sources[0].type.rank
    if not -rank <= axis < rank:
        lowerer.error(axis_node, "region source axis is outside rank")
    axis %= rank
    extent = sources[0].type.shape[axis]
    for source in sources[1:]:
        if source.type.rank <= axis or not dims_compatible(source.type.shape[axis], extent):
            lowerer.error(call_node, "region source components must share the source extent")
    return axis


def _slice_types(
    lowerer: FunctionLowerer,
    sources: tuple[MlirValue, ...],
    axis: int,
    role: str,
) -> tuple[TensorType, ...]:
    extent = lowerer.fresh_dynamic_dimension(f"{role}_slice")
    return tuple(
        TensorType(
            source.type.dtype,
            tuple(
                extent if position == axis else dimension
                for position, dimension in enumerate(source.type.shape)
            ),
        )
        for source in sources
    )


def _replace_slice_extent(
    value_type: ValueType,
    slice_extent: object,
    full_extent: object,
) -> ValueType:
    if isinstance(value_type, TensorType):
        matches = [axis for axis, dim in enumerate(value_type.shape) if dim == slice_extent]
        if len(matches) != 1:
            raise ValueError(
                "region_scan emit result must contain its source slice extent exactly once"
            )
        shape = list(value_type.shape)
        shape[matches[0]] = full_extent
        return TensorType(value_type.dtype, tuple(shape))
    if isinstance(value_type, RecordType):
        return RecordType(
            tuple(
                (name, _replace_slice_extent(field_type, slice_extent, full_extent))
                for name, field_type in value_type.fields
            )
        )
    if isinstance(value_type, TupleType):
        return TupleType(
            tuple(
                _replace_slice_extent(component, slice_extent, full_extent)
                for component in value_type.components
            )
        )
    raise ValueError("region_scan emit result must be a tensor or tensor record")


def _combine_captures(
    lowerer: FunctionLowerer,
    node: ast.AST | None,
    call_node: ast.AST,
) -> tuple[tuple[MlirValue, ...], tuple[ConstexprBinding | None, ...]]:
    return _region_captures(lowerer, node, call_node)


def _region_captures(
    lowerer: FunctionLowerer,
    node: ast.AST | None,
    call_node: ast.AST,
) -> tuple[tuple[MlirValue, ...], tuple[ConstexprBinding | None, ...]]:
    if node is None:
        return (), ()
    expression = lowerer.lower_expression(node)
    elements = expression.elements if isinstance(expression, StaticTuple) else (expression,)
    values: list[MlirValue] = []
    bindings: list[ConstexprBinding | None] = []
    for element in elements:
        value = lowerer.materialize(element, call_node)
        if isinstance(element, ConstexprBinding):
            binding = element
        elif isinstance(element, Literal):
            binding = ConstexprBinding(element.value, value)
        else:
            binding = None
        values.append(value)
        bindings.append(binding)
    return tuple(values), tuple(bindings)


def _sum_accumulator_dtype(dtype):
    from intent.language import f32

    if dtype.name in ("i8", "i16"):
        return i32
    if dtype.name in ("u8", "u16"):
        return u32
    if dtype.name in ("f8e4m3fn", "f8e5m2", "f16", "bf16"):
        return f32
    return dtype


def _builtin_identity(lowerer, dtype, operator, node):
    if operator is BinaryOperator.LOGICAL_OR:
        value = False
    elif operator is BinaryOperator.LOGICAL_AND:
        value = True
    elif operator is BinaryOperator.ADD:
        value = 0
    elif operator is BinaryOperator.MAXIMUM:
        if dtype.category in (DTypeCategory.FLOAT, DTypeCategory.BFLOAT):
            value = -448.0 if dtype.name == "f8e4m3fn" else -float("inf")
        elif dtype.category in (DTypeCategory.SIGNED_INTEGER, DTypeCategory.INDEX):
            value = -(1 << (dtype.bits - 1))
        elif dtype.category is DTypeCategory.UNSIGNED_INTEGER:
            value = 0
        else:
            lowerer.error(node, "maximum identity requires a numeric dtype")
    else:
        raise NotImplementedError(f"builtin identity for {operator}")
    return lowerer.emit_literal(value, node, ScalarType(dtype))


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
                lowerer.error(node, f"contract {purpose} axes must be compile-time integers")
            pair.append(value)
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


def _paired_relations(
    lowerer: FunctionLowerer,
    lhs: MlirValue,
    rhs: MlirValue,
    reduce: tuple[tuple[int, int], ...],
    batch: tuple[tuple[int, int], ...],
    node: ast.AST,
    *,
    logical_lhs_axis: int | None = None,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[object, ...]]:
    if not reduce:
        lowerer.error(node, "contract requires at least one reduction pair")
    lhs_reduce: set[int] = set()
    rhs_reduce: set[int] = set()
    lhs_batch: set[int] = set()
    rhs_batch: set[int] = set()
    normalized_reduce: list[tuple[int, int]] = []
    normalized_batch: list[tuple[int, int]] = []
    for left, right in reduce:
        left = normalize_axes(lowerer, (left,), lhs.type.rank, node)[0]
        right = normalize_axes(lowerer, (right,), rhs.type.rank, node)[0]
        if left in lhs_reduce or right in rhs_reduce:
            lowerer.error(node, "contract reduction axes must be unique")
        lhs_reduce.add(left)
        rhs_reduce.add(right)
        normalized_reduce.append((left, right))
    for left, right in batch:
        left = normalize_axes(lowerer, (left,), lhs.type.rank, node)[0]
        right = normalize_axes(lowerer, (right,), rhs.type.rank, node)[0]
        if left in lhs_reduce or right in rhs_reduce or left in lhs_batch or right in rhs_batch:
            lowerer.error(node, "contract batch axes must be unique and disjoint")
        if not dims_compatible(lhs.type.shape[left], rhs.type.shape[right]):
            lowerer.error(node, "contract paired batch extents are incompatible")
        lhs_batch.add(left)
        rhs_batch.add(right)
        normalized_batch.append((left, right))
    for left, right in normalized_reduce:
        if left != logical_lhs_axis and not dims_compatible(
            lhs.type.shape[left], rhs.type.shape[right]
        ):
            lowerer.error(node, "contract paired reduction extents are incompatible")
    result_shape = tuple(
        dimension
        for axis, dimension in enumerate(lhs.type.shape)
        if axis not in lhs_reduce
    ) + tuple(
        dimension
        for axis, dimension in enumerate(rhs.type.shape)
        if axis not in rhs_reduce and axis not in rhs_batch
    )
    return tuple(normalized_reduce), tuple(normalized_batch), result_shape


def _scaled_format(lowerer: FunctionLowerer, node: ast.AST) -> ScaledFormatKind:
    value = lowerer.lower_expression(node)
    if not isinstance(value, ScaledFormat):
        lowerer.error(node, "scaled-contract format must be I.e2m1/I.e4m3/I.e8m0")
    mapping = {
        "e2m1": ScaledFormatKind.E2M1,
        "e4m3": ScaledFormatKind.E4M3,
        "e8m0": ScaledFormatKind.E8M0,
    }
    try:
        return mapping[value.name]
    except KeyError:
        lowerer.error(node, "unknown scaled-contract format")
