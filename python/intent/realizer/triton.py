from __future__ import annotations

from intent.backend.triton.target import TritonTarget
from intent.ir import BinaryOperator
from intent.ir import FunctionKind
from intent.ir import IndexRelation
from intent.ir import IndexTermKind
from intent.ir import Module
from intent.ir import OpCode
from intent.ir import Operation
from intent.ir import ScalarType
from intent.ir import StaticDim
from intent.ir import TensorType
from intent.ir import Value
from intent.ir import verify
from intent.language import ViewKind
from intent.language import f32

from .model import AccessMode
from .model import BackendKind
from .model import BoundaryBinding
from .model import ExtentBinding
from .model import LaunchSpec
from .model import LayoutBinding
from .model import LayoutKind
from .model import OwnershipBinding
from .model import PhysicalPlan
from .model import PipelineSpec
from .model import PrimitiveBinding
from .model import PrimitiveKind
from .model import StorageBinding
from .model import StorageSpace
from .model import TailKind
from .model import TargetInfo
from .model import TraversalKind
from .model import WorkerKind
from .verify import verify_plan


def realize_triton(module: Module, target: TritonTarget) -> PhysicalPlan:
    verify(module)
    entry = module.entry()
    if len(module.functions) != 1 or entry.kind is not FunctionKind.KERNEL:
        raise NotImplementedError("initial Triton realization does not lower helper functions")

    parameter_by_value = {parameter.value: parameter for parameter in entry.parameters}
    view_parameters = tuple(entry.parameters)
    if not view_parameters or any(
        parameter.spec.view_kind is None for parameter in view_parameters
    ):
        raise NotImplementedError("initial Triton realization accepts ABI views only")
    for parameter in view_parameters:
        value_type = parameter.value.type
        if (
            not isinstance(value_type, TensorType)
            or value_type.rank != 1
            or value_type.dtype != f32
            or not isinstance(value_type.shape[0], StaticDim)
        ):
            raise NotImplementedError(
                "initial Triton realization accepts static rank-one f32 views"
            )
        if parameter.spec.constraints.layout != "contiguous":
            raise NotImplementedError(
                "initial Triton realization requires layout='contiguous' View constraints"
            )
        if (
            parameter.spec.constraints.strides is not None
            or parameter.spec.constraints.alignment is not None
            or parameter.spec.constraints.alias is not None
        ):
            raise NotImplementedError(
                "initial Triton realization does not implement stride/alignment/alias constraints"
            )

    top_level = entry.body.blocks[0].operations
    loops = [operation for operation in top_level if operation.opcode is OpCode.FOR]
    if len(loops) != 1:
        raise NotImplementedError("initial Triton realization requires one logical for loop")
    if any(
        operation.opcode
        not in (OpCode.CONSTANT, OpCode.DOMAIN, OpCode.FOR, OpCode.RETURN)
        for operation in top_level
    ):
        raise NotImplementedError("unsupported top-level operation in Triton pointwise kernel")

    loop = loops[0]
    if loop.result_types or len(loop.operands) != 1 or len(loop.regions) != 1:
        raise NotImplementedError("initial Triton loop cannot carry source state")
    domain_value = loop.operands[0]
    domain = domain_value.owner
    if not isinstance(domain, Operation) or domain.opcode is not OpCode.DOMAIN:
        raise NotImplementedError("Triton pointwise loop must iterate an explicit domain")
    if len(domain.operands) not in (2, 3):
        raise NotImplementedError("Triton pointwise domain requires start/stop[/step]")
    start = _integer_constant(domain.operands[0])
    stop = _integer_constant(domain.operands[1])
    step = _integer_constant(domain.operands[2]) if len(domain.operands) == 3 else 1
    if start != 0 or step != 1 or stop <= 0:
        raise NotImplementedError("initial Triton ownership requires domain(0, positive_extent, 1)")
    if any(parameter.value.type.shape != (StaticDim(stop),) for parameter in view_parameters):
        raise NotImplementedError("pointwise domain extent must match every ABI view")

    body = loop.regions[0].blocks[0]
    if len(body.arguments) != 1:
        raise NotImplementedError("pointwise loop requires one logical index")
    logical_index = body.arguments[0]
    allowed_body = {
        OpCode.CONSTANT,
        OpCode.VIEW_LOAD,
        OpCode.BINARY,
        OpCode.VIEW_STORE,
        OpCode.YIELD,
    }
    if any(operation.opcode not in allowed_body for operation in body.operations):
        raise NotImplementedError("unsupported operation in Triton pointwise loop")

    writes = 0
    primitive_bindings: list[PrimitiveBinding] = []
    for operation in body.operations:
        if operation.opcode is OpCode.VIEW_LOAD:
            _require_point_access(operation, logical_index, parameter_by_value)
            if operation.result_types != (ScalarType(f32),):
                raise NotImplementedError("pointwise View load must produce scalar f32")
        elif operation.opcode is OpCode.VIEW_STORE:
            _require_point_access(operation, logical_index, parameter_by_value)
            value_index = operation.attributes.get("value_operand_index")
            if value_index != 1 or operation.operands[value_index].type != ScalarType(f32):
                raise NotImplementedError("pointwise View store must consume scalar f32")
            writes += 1
        elif operation.opcode is OpCode.BINARY:
            operator = operation.attributes.get("operator")
            if operator not in (
                BinaryOperator.ADD,
                BinaryOperator.SUBTRACT,
                BinaryOperator.MULTIPLY,
                BinaryOperator.TRUE_DIVIDE,
            ):
                raise NotImplementedError("unsupported Triton pointwise binary operator")
            if any(value.type != ScalarType(f32) for value in operation.operands) or operation.result_types != (
                ScalarType(f32),
            ):
                raise NotImplementedError("Triton pointwise binary values must be scalar f32")
            primitive_bindings.append(
                PrimitiveBinding(operation.id, PrimitiveKind.POINTWISE, operator.value)
            )
    if writes == 0 or not primitive_bindings:
        raise NotImplementedError("pointwise Triton kernel must compute and write a value")

    target_info = _detect_target(target)
    tile_size = 256
    grid = ((stop + tile_size - 1) // tile_size,)
    access_modes = {
        ViewKind.IN: AccessMode.READ,
        ViewKind.OUT: AccessMode.WRITE,
        ViewKind.INOUT: AccessMode.READ_WRITE,
    }
    plan = PhysicalPlan(
        entry_name=entry.name,
        target=target_info,
        extents=(ExtentBinding(loop.id, domain_value.id, tile_size),),
        ownership=(
            OwnershipBinding(loop.id, WorkerKind.PROGRAM, 0, TraversalKind.STATIC),
        ),
        storage=tuple(
            StorageBinding(
                parameter.value.id,
                StorageSpace.GLOBAL,
                access_modes[parameter.spec.view_kind],
            )
            for parameter in view_parameters
        ),
        layouts=tuple(
            LayoutBinding(parameter.value.id, LayoutKind.CONTIGUOUS, (0,))
            for parameter in view_parameters
        ),
        primitives=tuple(primitive_bindings),
        pipeline=PipelineSpec(stages=1, prefetch=False, async_copy=False),
        boundaries=(
            BoundaryBinding(
                loop.id,
                stop,
                TailKind.EXACT if stop % tile_size == 0 else TailKind.MASKED,
            ),
        ),
        launch=LaunchSpec(loop.id, grid, tile_size, num_warps=4),
    )
    verify_plan(module, plan)
    return plan


def _integer_constant(value: Value) -> int:
    owner = value.owner
    if not isinstance(owner, Operation) or owner.opcode is not OpCode.CONSTANT:
        raise NotImplementedError("initial Triton domain bounds must be static constants")
    constant = owner.attributes.get("value")
    if isinstance(constant, bool) or not isinstance(constant, int):
        raise NotImplementedError("initial Triton domain bound must be an integer")
    return constant


def _require_point_access(
    operation: Operation,
    logical_index: Value,
    parameter_by_value: dict[Value, object],
) -> None:
    if not operation.operands or operation.operands[0] not in parameter_by_value:
        raise NotImplementedError("pointwise access must target an ABI View")
    relation = operation.attributes.get("index")
    if not isinstance(relation, IndexRelation) or len(relation.terms) != 1:
        raise NotImplementedError("pointwise access requires one explicit index term")
    term = relation.terms[0]
    if term.kind is not IndexTermKind.VALUE_INDEX or len(term.operand_positions) != 1:
        raise NotImplementedError("pointwise access requires a scalar value index")
    position = term.operand_positions[0]
    if position is None or operation.operands[position] is not logical_index:
        raise NotImplementedError("pointwise access must preserve the source logical index")


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
