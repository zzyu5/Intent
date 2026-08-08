from __future__ import annotations

import math

from intent.errors import IntentError
from intent.ir import Module
from intent.ir import OpCode
from intent.ir import Operation
from intent.ir import ParameterKind
from intent.ir import SymbolDim
from intent.ir import TensorType
from intent.ir import Value
from intent.ir import verify
from intent.ir import walk_blocks
from intent.ir import walk_operations
from intent.language import ViewKind

from .model import AccessMode
from .model import BackendKind
from .model import GridPolicy
from .model import LayoutKind
from .model import PhysicalPlan
from .model import PrimitiveKind
from .model import StorageSpace
from .model import TailKind
from .model import TileKind
from .model import TraversalKind
from .model import WorkerKind


class PlanVerificationError(IntentError):
    pass


def verify_plan(module: Module, plan: PhysicalPlan) -> None:
    verify(module)
    errors: list[str] = []
    entry = module.entry()
    if plan.entry_name != entry.name:
        errors.append("Physical Plan entry does not match Kernel IR entry")
    if plan.target.backend is not BackendKind.TRITON:
        errors.append("initial Physical Plan requires a Triton target")
    if plan.target.device < 0 or plan.target.warp_size <= 0 or not plan.target.architecture:
        errors.append("target information is incomplete")

    operations = {
        operation.id: operation for operation in walk_operations(entry.body)
    }
    values: dict[int, Value] = {}
    for block in walk_blocks(entry.body):
        for argument in block.arguments:
            values[argument.id] = argument
        for operation in block.operations:
            for result in operation.results:
                values[result.id] = result

    extent_nodes: set[int] = set()
    abi_shape: tuple[str, str] | None = None
    if len(entry.parameters) == 2:
        input_type = entry.parameters[0].value.type
        output_type = entry.parameters[1].value.type
        if (
            isinstance(input_type, TensorType)
            and input_type == output_type
            and input_type.rank == 2
            and all(isinstance(dimension, SymbolDim) for dimension in input_type.shape)
        ):
            abi_shape = (input_type.shape[0].name, input_type.shape[1].name)
    for binding in plan.extents:
        operation = operations.get(binding.node_id)
        if operation is None or operation.opcode not in (OpCode.DOMAIN, OpCode.PARALLEL):
            errors.append("softmax extent must reference a domain or parallel loop")
        if binding.axis not in (0, 1) or not binding.logical_extent:
            errors.append("softmax extent requires row/column axis and symbolic extent")
        if binding.axis == 0 and binding.tile is not TileKind.ONE:
            errors.append("softmax row extent currently realizes one row at a time")
        if binding.axis == 1 and binding.tile is not TileKind.NEXT_POWER_OF_TWO:
            errors.append("softmax column tile must be next_power_of_two(logical extent)")
        if (
            abi_shape is None
            or binding.axis not in (0, 1)
            or binding.logical_extent != abi_shape[binding.axis]
        ):
            errors.append("softmax extent must preserve the source ABI symbolic shape")
        if binding.node_id in extent_nodes:
            errors.append("Kernel IR node has more than one extent binding")
        extent_nodes.add(binding.node_id)
    if len(plan.extents) != 2 or {binding.axis for binding in plan.extents} != {0, 1}:
        errors.append("softmax Plan requires exactly one row and one column extent")

    if len(plan.ownership) != 1:
        errors.append("softmax Plan requires exactly one ownership binding")
    else:
        ownership = plan.ownership[0]
        operation = operations.get(ownership.loop_node_id)
        if operation is None or operation.opcode is not OpCode.PARALLEL:
            errors.append("softmax ownership must reference the parallel row loop")
        if (
            ownership.worker is not WorkerKind.PROGRAM
            or ownership.worker_axis != 0
            or ownership.traversal is not TraversalKind.PERSISTENT
            or ownership.mapping is not TraversalKind.GRID_STRIDE
        ):
            errors.append("softmax ownership must be persistent program-axis-zero grid-stride")

    parameter_specs = {parameter.value.id: parameter.spec for parameter in entry.parameters}
    expected_access = {
        ViewKind.IN: AccessMode.READ,
        ViewKind.OUT: AccessMode.WRITE,
        ViewKind.INOUT: AccessMode.READ_WRITE,
    }
    storage_values: set[int] = set()
    for binding in plan.storage:
        spec = parameter_specs.get(binding.value_id)
        if (
            spec is None
            or spec.kind is not ParameterKind.VIEW
            or binding.space is not StorageSpace.GLOBAL
        ):
            errors.append("softmax storage must reference a global ABI view")
        elif binding.access is not expected_access[spec.view_kind]:
            errors.append("storage access mode changes the source View contract")
        elif (
            spec.constraints.strides != (None, 1)
            or spec.constraints.layout != "row_major"
            or not spec.constraints.noalias
            or spec.constraints.alignment is not None
            or spec.constraints.alias is not None
        ):
            errors.append("softmax Plan requires row-major noalias ABI views")
        if binding.value_id in storage_values:
            errors.append("SSA value has more than one storage binding")
        storage_values.add(binding.value_id)
    if storage_values != set(parameter_specs):
        errors.append("storage bindings must cover the complete softmax ABI")

    layout_values: set[int] = set()
    for binding in plan.layouts:
        value = values.get(binding.value_id)
        if value is None or not isinstance(value.type, TensorType):
            errors.append("layout binding must reference a tensor view")
        elif binding.kind is not LayoutKind.ROW_MAJOR or binding.order != (0, 1):
            errors.append("softmax layout must preserve row-major axes [0, 1]")
        if binding.value_id in layout_values:
            errors.append("SSA value has more than one layout binding")
        layout_values.add(binding.value_id)
    if layout_values != storage_values:
        errors.append("storage and layout bindings must cover the same ABI values")

    primitive_nodes: set[int] = set()
    expected_primitive_nodes: set[int] = set()
    for operation in operations.values():
        if operation.opcode in (OpCode.REDUCE, OpCode.BROADCAST, OpCode.UNARY, OpCode.BINARY):
            expected_primitive_nodes.add(operation.id)
    for binding in plan.primitives:
        operation = operations.get(binding.node_id)
        if operation is None:
            errors.append("primitive binding references a missing Kernel IR node")
        elif binding.kind is PrimitiveKind.REDUCTION and operation.opcode is not OpCode.REDUCE:
            errors.append("reduction primitive must reference intent.reduce")
        elif binding.kind is PrimitiveKind.POINTWISE and operation.opcode not in (
            OpCode.BROADCAST,
            OpCode.UNARY,
            OpCode.BINARY,
        ):
            errors.append("pointwise primitive references an incompatible Kernel IR node")
        elif binding.kind is PrimitiveKind.REDUCTION:
            combine = operation.attributes.get("combine")
            axes = operation.attributes.get("axes")
            required_identity = {
                "maximum": "negative_infinity",
                "add": "zero",
            }.get(combine)
            identity = operation.operands[1].owner
            identity_value = (
                identity.attributes.get("value")
                if isinstance(identity, Operation) and identity.opcode is OpCode.CONSTANT
                else None
            )
            identity_matches = (
                required_identity == "negative_infinity"
                and isinstance(identity_value, float)
                and math.isinf(identity_value)
                and identity_value < 0.0
            ) or (required_identity == "zero" and identity_value == 0.0)
            if (
                binding.operator != combine
                or tuple(axes or ()) != (binding.axis,)
                or binding.identity != required_identity
                or not identity_matches
            ):
                errors.append("reduction primitive changes axis/operator/identity")
        elif binding.kind is PrimitiveKind.POINTWISE:
            source_operator = (
                "broadcast"
                if operation.opcode is OpCode.BROADCAST
                else getattr(operation.attributes.get("operator"), "value", operation.attributes.get("operator"))
            )
            if (
                binding.operator != source_operator
                or binding.axis != -1
                or binding.identity != "none"
            ):
                errors.append("pointwise primitive does not match its Kernel IR node")
        if binding.node_id in primitive_nodes:
            errors.append("Kernel IR node has more than one primitive binding")
        primitive_nodes.add(binding.node_id)
    if primitive_nodes != expected_primitive_nodes:
        errors.append("primitive bindings must exactly cover softmax computations")

    if len(plan.boundaries) != 1:
        errors.append("softmax Plan requires exactly one column boundary")
    else:
        boundary = plan.boundaries[0]
        operation = operations.get(boundary.node_id)
        if operation is None or operation.opcode is not OpCode.DOMAIN:
            errors.append("softmax boundary must reference the column domain")
        if (
            boundary.tail is not TailKind.MASKED
            or boundary.predicate != "index_lt_extent"
            or boundary.load_fill != "negative_infinity"
        ):
            errors.append("softmax boundary must use masked col < N and -infinity fill")
        column_extent = next(
            (binding for binding in plan.extents if binding.axis == 1),
            None,
        )
        if (
            column_extent is None
            or boundary.node_id != column_extent.node_id
            or boundary.logical_extent != column_extent.logical_extent
        ):
            errors.append("softmax boundary must bind the planned column extent")

    pipeline = plan.pipeline
    if (
        pipeline.low_stages <= 0
        or pipeline.high_stages <= 0
        or pipeline.smem_threshold <= 0
        or pipeline.prefetch
        or pipeline.async_copy
    ):
        errors.append("softmax pipeline policy is invalid")

    launch = plan.launch
    if (
        launch.grid_policy is not GridPolicy.PERSISTENT_OCCUPANCY
        or launch.num_warps <= 0
    ):
        errors.append("softmax launch requires persistent occupancy policy")
    if len(plan.ownership) == 1 and launch.loop_node_id != plan.ownership[0].loop_node_id:
        errors.append("launch and ownership must reference the same row loop")

    if errors:
        raise PlanVerificationError("\n".join(errors))
