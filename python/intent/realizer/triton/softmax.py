from __future__ import annotations

from intent.backend.triton.target import TritonTarget
from intent.ir import BinaryOperator
from intent.ir import FunctionKind
from intent.ir import Module
from intent.ir import OpCode
from intent.ir import Operation
from intent.ir import SymbolDim
from intent.ir import TensorType
from intent.ir import UnaryOperator
from intent.ir import verify
from intent.language import ViewKind
from intent.language import f32

from ..model import AccessMode
from ..model import BackendKind
from ..model import BoundaryBinding
from ..model import ExtentBinding
from ..model import GridPolicy
from ..model import LaunchSpec
from ..model import LayoutBinding
from ..model import LayoutKind
from ..model import OwnershipBinding
from ..model import PhysicalPlan
from ..model import PipelineSpec
from ..model import PrimitiveBinding
from ..model import PrimitiveKind
from ..model import StorageBinding
from ..model import StorageSpace
from ..model import TailKind
from ..model import TargetInfo
from ..model import TileKind
from ..model import TraversalKind
from ..model import WorkerKind
from ..verify import verify_plan


def realize_stable_softmax(module: Module, target: TritonTarget) -> PhysicalPlan:
    verify(module)
    entry = module.entry()
    if len(module.functions) != 1 or entry.kind is not FunctionKind.KERNEL:
        raise NotImplementedError("initial softmax realization does not lower helper functions")
    if len(entry.parameters) != 2:
        raise NotImplementedError("stable softmax realization requires input and output views")

    input_parameter, output_parameter = entry.parameters
    _require_softmax_view(input_parameter, ViewKind.IN)
    _require_softmax_view(output_parameter, ViewKind.OUT)
    if input_parameter.value.type != output_parameter.value.type:
        raise NotImplementedError("stable softmax input/output schemas must be identical")
    value_type = input_parameter.value.type
    if not isinstance(value_type, TensorType):
        raise NotImplementedError("stable softmax requires tensor views")
    row_extent, column_extent = value_type.shape
    if not isinstance(row_extent, SymbolDim) or not isinstance(column_extent, SymbolDim):
        raise NotImplementedError("stable softmax demo requires two symbolic dimensions")

    top_level = entry.body.blocks[0].operations
    row_loops = [operation for operation in top_level if operation.opcode is OpCode.PARALLEL]
    if len(row_loops) != 1:
        raise NotImplementedError("stable softmax requires one top-level parallel row loop")
    row_loop = row_loops[0]
    if len(row_loop.operands) != 1 or len(row_loop.regions) != 1:
        raise NotImplementedError("stable softmax row loop has an unsupported schema")
    row_domain = row_loop.operands[0].owner
    if not isinstance(row_domain, Operation) or row_domain.opcode is not OpCode.DOMAIN:
        raise NotImplementedError("stable softmax row ownership requires an explicit domain")

    body = row_loop.regions[0].blocks[0]
    loads = [operation for operation in body.operations if operation.opcode is OpCode.VIEW_LOAD]
    stores = [operation for operation in body.operations if operation.opcode is OpCode.VIEW_STORE]
    reductions = [operation for operation in body.operations if operation.opcode is OpCode.REDUCE]
    unary = [operation for operation in body.operations if operation.opcode is OpCode.UNARY]
    binary = [operation for operation in body.operations if operation.opcode is OpCode.BINARY]
    if len(loads) != 1 or len(stores) != 1 or len(reductions) != 2:
        raise NotImplementedError("stable softmax requires one load/store and two reductions")

    load = loads[0]
    store = stores[0]
    reduce_max = _one_operation(reductions, "combine", "maximum")
    reduce_sum = _one_operation(reductions, "combine", "add")
    subtract = _one_operation(binary, "operator", BinaryOperator.SUBTRACT)
    divide = _one_operation(binary, "operator", BinaryOperator.TRUE_DIVIDE)
    exponential = _one_operation(unary, "operator", UnaryOperator.EXP)

    loaded = load.results[0]
    maximum = reduce_max.results[0]
    numerator = exponential.results[0]
    denominator = reduce_sum.results[0]
    result = divide.results[0]
    if reduce_max.operands[0] is not loaded:
        raise NotImplementedError("stable softmax max must reduce the loaded row")
    maximum_broadcast = _broadcast_of(subtract.operands[1], maximum)
    if subtract.operands[0] is not loaded:
        raise NotImplementedError("stable softmax must subtract the full-row maximum")
    if exponential.operands != (subtract.results[0],):
        raise NotImplementedError("stable softmax exp must consume the shifted row")
    if reduce_sum.operands[0] is not numerator:
        raise NotImplementedError("stable softmax sum must reduce the numerator")
    denominator_broadcast = _broadcast_of(divide.operands[1], denominator)
    if divide.operands[0] is not numerator:
        raise NotImplementedError("stable softmax divide must use numerator and denominator")
    value_index = store.attributes.get("value_operand_index")
    if value_index != 1 or store.operands[value_index] is not result:
        raise NotImplementedError("stable softmax store must write the normalized row")
    if load.operands[0] is not input_parameter.value or store.operands[0] is not output_parameter.value:
        raise NotImplementedError("stable softmax memory flow must preserve the source ABI")

    column_domains = [
        operation
        for operation in top_level
        if operation.opcode is OpCode.DOMAIN and operation is not row_domain
    ]
    if len(column_domains) != 1:
        raise NotImplementedError("stable softmax requires one explicit column domain")
    column_domain = column_domains[0]

    plan = PhysicalPlan(
        entry_name=entry.name,
        target=_detect_target(target),
        extents=(
            ExtentBinding(row_loop.id, 0, row_extent.name, TileKind.ONE),
            ExtentBinding(column_domain.id, 1, column_extent.name, TileKind.NEXT_POWER_OF_TWO),
        ),
        ownership=(
            OwnershipBinding(
                row_loop.id,
                WorkerKind.PROGRAM,
                0,
                TraversalKind.PERSISTENT,
                TraversalKind.GRID_STRIDE,
            ),
        ),
        storage=(
            StorageBinding(input_parameter.value.id, StorageSpace.GLOBAL, AccessMode.READ),
            StorageBinding(output_parameter.value.id, StorageSpace.GLOBAL, AccessMode.WRITE),
        ),
        layouts=(
            LayoutBinding(input_parameter.value.id, LayoutKind.ROW_MAJOR, (0, 1)),
            LayoutBinding(output_parameter.value.id, LayoutKind.ROW_MAJOR, (0, 1)),
        ),
        primitives=(
            PrimitiveBinding(reduce_max.id, PrimitiveKind.REDUCTION, "maximum", 0, "negative_infinity"),
            PrimitiveBinding(maximum_broadcast.id, PrimitiveKind.POINTWISE, "broadcast"),
            PrimitiveBinding(subtract.id, PrimitiveKind.POINTWISE, "subtract"),
            PrimitiveBinding(exponential.id, PrimitiveKind.POINTWISE, "exp"),
            PrimitiveBinding(reduce_sum.id, PrimitiveKind.REDUCTION, "add", 0, "zero"),
            PrimitiveBinding(denominator_broadcast.id, PrimitiveKind.POINTWISE, "broadcast"),
            PrimitiveBinding(divide.id, PrimitiveKind.POINTWISE, "true_divide"),
        ),
        pipeline=PipelineSpec(
            low_stages=2,
            high_stages=4,
            smem_threshold=200_000,
            prefetch=False,
            async_copy=False,
        ),
        boundaries=(
            BoundaryBinding(
                column_domain.id,
                column_extent.name,
                TailKind.MASKED,
                "index_lt_extent",
                "negative_infinity",
            ),
        ),
        launch=LaunchSpec(row_loop.id, GridPolicy.PERSISTENT_OCCUPANCY, num_warps=8),
    )
    verify_plan(module, plan)
    return plan


def _require_softmax_view(parameter: object, expected_kind: ViewKind) -> None:
    if parameter.spec.view_kind is not expected_kind:
        raise NotImplementedError("stable softmax requires one In and one Out view")
    value_type = parameter.value.type
    if not isinstance(value_type, TensorType) or value_type.rank != 2 or value_type.dtype != f32:
        raise NotImplementedError("stable softmax demo accepts rank-two f32 views")
    constraints = parameter.spec.constraints
    if constraints.strides != (None, 1) or constraints.layout != "row_major" or not constraints.noalias:
        raise NotImplementedError(
            "stable softmax requires strides=(None, 1), layout='row_major', noalias=True"
        )
    if constraints.alignment is not None or constraints.alias is not None:
        raise NotImplementedError("stable softmax demo does not realize alignment/alias groups")


def _one_operation(operations: list[Operation], attribute: str, expected: object) -> Operation:
    matches = [operation for operation in operations if operation.attributes.get(attribute) == expected]
    if len(matches) != 1:
        raise NotImplementedError(
            f"stable softmax requires exactly one {attribute}={getattr(expected, 'value', expected)!r} operation"
        )
    return matches[0]


def _broadcast_of(value: object, source: object) -> Operation:
    owner = getattr(value, "owner", None)
    if (
        not isinstance(owner, Operation)
        or owner.opcode is not OpCode.BROADCAST
        or owner.operands != (source,)
    ):
        raise NotImplementedError("stable softmax scalar reduction must explicitly broadcast")
    return owner


def _detect_target(target: TritonTarget) -> TargetInfo:
    import torch

    if not torch.cuda.is_available() or target.device >= torch.cuda.device_count():
        raise RuntimeError("requested Triton CUDA device is unavailable")
    major, minor = torch.cuda.get_device_capability(target.device)
    return TargetInfo(
        backend=BackendKind.TRITON,
        architecture=f"sm_{major}{minor}",
        device=target.device,
        warp_size=32,
    )
