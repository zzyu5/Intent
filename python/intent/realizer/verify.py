from __future__ import annotations

from intent.errors import IntentError
from intent.ir import Module
from intent.ir import OpCode
from intent.ir import Operation
from intent.ir import ParameterKind
from intent.ir import TensorType
from intent.ir import Value
from intent.ir import verify
from intent.ir import walk_blocks
from intent.ir import walk_operations
from intent.language import ViewKind

from .model import AccessMode
from .model import BackendKind
from .model import LayoutKind
from .model import PhysicalPlan
from .model import PrimitiveKind
from .model import StorageSpace
from .model import TailKind
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
        errors.append("only Triton target plans are currently representable")
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

    parameter_specs = {
        parameter.value.id: parameter.spec for parameter in entry.parameters
    }

    extent_loops: set[int] = set()
    logical_extents: dict[int, int] = {}
    for binding in plan.extents:
        operation = operations.get(binding.loop_node_id)
        domain = values.get(binding.domain_value_id)
        if operation is None or operation.opcode not in (
            OpCode.FOR,
            OpCode.PARALLEL,
            OpCode.ORDERED,
            OpCode.STATE_STREAM,
        ):
            errors.append("extent binding must reference a structured iteration node")
        if domain is None or domain.owner is None or not hasattr(domain.owner, "opcode"):
            errors.append("extent binding must reference a logical domain value")
        elif domain.owner.opcode not in (
            OpCode.DOMAIN,
            OpCode.DOMAIN_PRODUCT,
            OpCode.PARTITION,
            OpCode.RAGGED_MEMBER,
        ):
            errors.append("extent binding source is not a logical domain/region producer")
        elif operation is not None and (
            not operation.operands or operation.operands[0] is not domain
        ):
            errors.append("extent binding does not match the structured loop source")
        elif domain.owner.opcode is OpCode.DOMAIN:
            bounds = domain.owner.operands
            start = _static_integer(bounds[0]) if len(bounds) >= 2 else None
            stop = _static_integer(bounds[1]) if len(bounds) >= 2 else None
            step = _static_integer(bounds[2]) if len(bounds) == 3 else 1
            if start != 0 or stop is None or stop <= 0 or step != 1:
                errors.append("initial Triton extent requires static domain(0, positive, 1)")
            else:
                logical_extents[binding.loop_node_id] = stop
        if binding.tile_size <= 0 or binding.tile_size & (binding.tile_size - 1):
            errors.append("Triton extent tile must be a positive power of two")
        if binding.loop_node_id in extent_loops:
            errors.append("structured loop has more than one extent binding")
        extent_loops.add(binding.loop_node_id)
    if len(extent_loops) != 1:
        errors.append("initial Triton artifact must realize exactly one logical loop")

    ownership_loops: set[int] = set()
    for binding in plan.ownership:
        if binding.loop_node_id not in extent_loops:
            errors.append("ownership binding has no matching extent")
        if binding.worker is not WorkerKind.PROGRAM or binding.worker_axis != 0:
            errors.append("initial Triton realization supports program axis zero only")
        if binding.traversal is not TraversalKind.STATIC:
            errors.append("initial Triton realization supports static traversal only")
        if binding.loop_node_id in ownership_loops:
            errors.append("structured loop has more than one ownership binding")
        ownership_loops.add(binding.loop_node_id)

    storage_values: set[int] = set()
    expected_access = {
        ViewKind.IN: AccessMode.READ,
        ViewKind.OUT: AccessMode.WRITE,
        ViewKind.INOUT: AccessMode.READ_WRITE,
    }
    for binding in plan.storage:
        spec = parameter_specs.get(binding.value_id)
        if (
            spec is None
            or spec.kind is not ParameterKind.VIEW
            or binding.space is not StorageSpace.GLOBAL
        ):
            errors.append("initial global storage bindings must reference ABI views")
        elif binding.access is not expected_access[spec.view_kind]:
            errors.append("storage access mode changes the source View contract")
        if binding.value_id in storage_values:
            errors.append("SSA value has more than one storage binding")
        storage_values.add(binding.value_id)
    expected_storage_values = {
        value_id
        for value_id, spec in parameter_specs.items()
        if spec.kind is ParameterKind.VIEW
    }
    if storage_values != expected_storage_values:
        errors.append("storage bindings must cover every kernel ABI View exactly once")

    layout_values: set[int] = set()
    for binding in plan.layouts:
        value = values.get(binding.value_id)
        if value is None or not isinstance(value.type, TensorType):
            errors.append("layout binding must reference a tensor/view value")
        elif binding.kind is not LayoutKind.CONTIGUOUS or sorted(binding.order) != list(
            range(value.type.rank)
        ):
            errors.append("layout binding does not preserve logical tensor axes")
        if binding.value_id in layout_values:
            errors.append("SSA value has more than one layout binding")
        layout_values.add(binding.value_id)
    if layout_values != storage_values:
        errors.append("every external storage binding requires one layout binding")

    primitive_nodes: set[int] = set()
    loop_primitive_nodes: set[int] = set()
    for loop_id in extent_loops:
        loop = operations.get(loop_id)
        if loop is not None:
            loop_primitive_nodes.update(
                operation.id
                for region in loop.regions
                for operation in walk_operations(region)
                if operation.opcode is OpCode.BINARY
            )
    for binding in plan.primitives:
        operation = operations.get(binding.node_id)
        if operation is None or binding.kind is not PrimitiveKind.POINTWISE:
            errors.append("primitive binding references an unsupported node")
        elif operation.opcode is not OpCode.BINARY:
            errors.append("pointwise primitive must reference a binary operation")
        elif getattr(operation.attributes.get("operator"), "value", None) != binding.operator:
            errors.append("primitive operator changes the Kernel IR operation")
        if binding.node_id in primitive_nodes:
            errors.append("Kernel IR node has more than one primitive binding")
        primitive_nodes.add(binding.node_id)
    if primitive_nodes != loop_primitive_nodes:
        errors.append("primitive bindings must exactly cover realized loop computations")

    if (
        plan.pipeline.stages != 1
        or plan.pipeline.prefetch
        or plan.pipeline.async_copy
    ):
        errors.append("initial Triton realization supports only a single unpipelined stage")

    boundary_loops: set[int] = set()
    for binding in plan.boundaries:
        if binding.loop_node_id not in extent_loops or binding.logical_extent <= 0:
            errors.append("boundary binding must cover a positive logical extent")
        elif logical_extents.get(binding.loop_node_id) != binding.logical_extent:
            errors.append("boundary logical extent differs from the Kernel IR domain")
        if binding.tail not in (TailKind.EXACT, TailKind.MASKED):
            errors.append("initial Triton realization supports exact/masked tails only")
        extent = next(
            (
                extent
                for extent in plan.extents
                if extent.loop_node_id == binding.loop_node_id
            ),
            None,
        )
        if (
            binding.tail is TailKind.EXACT
            and extent is not None
            and binding.logical_extent % extent.tile_size != 0
        ):
            errors.append("exact boundary requires an extent divisible by the tile")
        if binding.loop_node_id in boundary_loops:
            errors.append("structured loop has more than one boundary binding")
        boundary_loops.add(binding.loop_node_id)
    if boundary_loops != extent_loops or ownership_loops != extent_loops:
        errors.append("each physical extent requires ownership and boundary bindings")

    launch = plan.launch
    matching_extent = next(
        (binding for binding in plan.extents if binding.loop_node_id == launch.loop_node_id),
        None,
    )
    matching_boundary = next(
        (binding for binding in plan.boundaries if binding.loop_node_id == launch.loop_node_id),
        None,
    )
    if matching_extent is None or matching_boundary is None:
        errors.append("launch must reference a realized logical loop")
    else:
        if launch.block_size != matching_extent.tile_size:
            errors.append("launch block size differs from extent realization")
        expected_grid = (
            (matching_boundary.logical_extent + launch.block_size - 1)
            // launch.block_size
        )
        if launch.grid != (expected_grid,):
            errors.append("launch grid does not exactly cover the ownership space")
    if launch.num_warps <= 0:
        errors.append("launch must request a positive worker count")

    if errors:
        raise PlanVerificationError("\n".join(errors))


def _static_integer(value: Value) -> int | None:
    owner = value.owner
    if not isinstance(owner, Operation) or owner.opcode is not OpCode.CONSTANT:
        return None
    constant = owner.attributes.get("value")
    if isinstance(constant, bool) or not isinstance(constant, int):
        return None
    return constant
