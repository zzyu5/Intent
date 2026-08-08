from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from intent.diagnostics import IntentError
from intent.language import DType
from intent.language.dtypes import DTypeCategory
from intent.language.annotations import ViewKind
from intent.language import bool as intent_bool
from intent.language import index as intent_index

from .effects import EffectKind
from .effects import ResourceKind
from .locations import Location
from .module import Function
from .module import FunctionKind
from .module import Module
from .module import ParameterKind
from .ops import EFFECTFUL_OPS
from .ops import REGION_OPS
from .ops import TERMINATORS
from .ops import AutoExtent
from .ops import AtomicOrdering
from .ops import BinaryOperator
from .ops import ComparePredicate
from .ops import IndexRelation
from .ops import IndexTermKind
from .ops import MemoryScope
from .ops import OpCode
from .ops import UnaryOperator
from .types import BufferType
from .types import ConstexprType
from .types import DomainFlavor
from .types import DomainType
from .types import DynamicDim
from .types import IRType
from .types import LogicalIndexType
from .types import PartitionMode
from .types import PartitionType
from .types import RaggedType
from .types import RecordType
from .types import RegionType
from .types import ScalarType
from .types import StaticDim
from .types import TensorType
from .types import broadcast_shape
from .types import is_boolean
from .types import is_integer
from .types import static_numel
from .types import types_compatible
from .values import Block
from .values import Operation
from .values import Region
from .values import Value


@dataclass(frozen=True, slots=True)
class Diagnostic:
    message: str
    location: Location

    def format(self) -> str:
        return f"{self.location.format()}: error: {self.message}"


class VerificationError(IntentError):
    def __init__(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.diagnostics = tuple(diagnostics)
        super().__init__("\n".join(diagnostic.format() for diagnostic in self.diagnostics))


_FORBIDDEN_PHYSICAL_ATTRIBUTES = {
    "program_id",
    "block_id",
    "thread_id",
    "warp_id",
    "grid",
    "worker_mapping",
    "physical_tile",
    "num_warps",
    "num_stages",
    "address_space",
    "storage_address_space",
    "fragment_layout",
    "pipeline",
    "prefetch",
    "launch",
}


@dataclass(frozen=True, slots=True)
class LoopFrame:
    carried_types: tuple[IRType, ...]
    allow_break: bool
    allow_continue: bool


LoopStack = tuple[LoopFrame, ...]


class Verifier:
    def __init__(self) -> None:
        self.diagnostics: list[Diagnostic] = []
        self._operation_ids: set[int] = set()
        self._value_ids: set[int] = set()
        self._symbols: dict[str, Function] = {}

    def verify(self, module: Module) -> None:
        self.diagnostics.clear()
        self._operation_ids.clear()
        self._value_ids.clear()
        self._symbols.clear()

        self._verify_no_physical_attributes(module.attributes, module.location)

        names = [function.name for function in module.functions]
        if len(set(names)) != len(names):
            self._error(module.location, "function symbols must be unique")
        self._symbols = module.symbol_table()

        kernels = [
            function
            for function in module.functions
            if function.kind is FunctionKind.KERNEL
        ]
        if len(kernels) != 1:
            self._error(module.location, "module must contain exactly one kernel entry")

        for function in module.functions:
            self._verify_function(function)

        self._verify_combiner_purity(module)
        if self.diagnostics:
            raise VerificationError(self.diagnostics)

    def _error(self, location: Location, message: str) -> None:
        self.diagnostics.append(Diagnostic(message, location))

    def _register_value(self, value: Value) -> None:
        if value.id in self._value_ids:
            self._error(value.location, f"duplicate SSA value id {value.id}")
        self._value_ids.add(value.id)

    def _verify_function(self, function: Function) -> None:
        self._verify_no_physical_attributes(function.attributes, function.location)
        parameter_names = [parameter.spec.name for parameter in function.parameters]
        if len(set(parameter_names)) != len(parameter_names):
            self._error(function.location, "parameter names must be unique")
        if function.kind is FunctionKind.KERNEL and function.result_types:
            self._error(
                function.location,
                "kernel entry returns through Out/InOut views and cannot return SSA values",
            )
        if len(function.body.blocks) != 1:
            self._error(function.location, "function body must contain exactly one block")
            return

        entry = function.body.blocks[0]
        if len(entry.arguments) != len(function.parameters):
            self._error(function.location, "entry block arguments must match parameters")

        available: set[Value] = set()
        for argument in entry.arguments:
            if argument.owner is not entry:
                self._error(argument.location, "entry block argument owner is inconsistent")
            self._register_value(argument)
            available.add(argument)
        for index, parameter in enumerate(function.parameters):
            self._verify_parameter(function.kind, parameter.spec, parameter.value)
            if index < len(entry.arguments) and parameter.value is not entry.arguments[index]:
                self._error(parameter.spec.location, "parameter value is not its entry block argument")

        self._verify_block(
            entry,
            function,
            available,
            expected_terminator=OpCode.RETURN,
            yielded_types=function.result_types,
            loop_stack=(),
        )

    def _verify_no_physical_attributes(
        self,
        attributes: dict[str, object],
        location: Location,
    ) -> None:
        for key, value in attributes.items():
            if key in _FORBIDDEN_PHYSICAL_ATTRIBUTES:
                self._error(
                    location,
                    f"physical attribute {key!r} is not legal in Kernel IR",
                )
            if isinstance(value, dict):
                self._verify_no_physical_attributes(value, location)

    def _verify_parameter(
        self,
        function_kind: FunctionKind,
        spec: object,
        value: Value,
    ) -> None:
        from .module import ParameterSpec

        if not isinstance(spec, ParameterSpec):
            self._error(value.location, "invalid parameter specification")
            return
        if value.type != spec.type:
            self._error(spec.location, "parameter value type does not match signature")
        if function_kind is FunctionKind.KERNEL and isinstance(spec.type, BufferType):
            self._error(spec.location, "logical buffers cannot appear in the kernel ABI")
        if function_kind is FunctionKind.KERNEL and spec.kind is ParameterKind.VALUE:
            self._error(spec.location, "kernel entry parameters require explicit ABI kinds")
        if function_kind is FunctionKind.HELPER and spec.kind is ParameterKind.RUNTIME_SCALAR:
            self._error(spec.location, "helper SSA parameters use ParameterKind.VALUE")
        if spec.kind is ParameterKind.VIEW and not isinstance(spec.type, TensorType):
            self._error(spec.location, "view parameter must have tensor type")
        if (
            spec.kind is ParameterKind.VIEW
            and spec.constraints is not None
            and spec.constraints.strides is not None
            and isinstance(spec.type, TensorType)
            and len(spec.constraints.strides) != spec.type.rank
        ):
            self._error(spec.location, "view stride constraint rank must match tensor rank")
        if spec.kind is ParameterKind.CONSTEXPR and not isinstance(
            spec.type, ConstexprType
        ):
            self._error(spec.location, "constexpr parameter must have ConstexprType")
        if spec.kind is ParameterKind.RUNTIME_SCALAR and not isinstance(
            spec.type, (ScalarType, LogicalIndexType)
        ):
            self._error(spec.location, "runtime scalar parameter must be scalar")
        if spec.kind is ParameterKind.VALUE and isinstance(spec.type, ConstexprType):
            self._error(spec.location, "helper value parameter must be ordinary SSA data")

    def _verify_region(
        self,
        region: Region,
        function: Function,
        outer_available: set[Value],
        expected_terminator: OpCode,
        yielded_types: tuple[IRType, ...],
        loop_stack: LoopStack,
    ) -> None:
        if len(region.blocks) != 1:
            self._error(region.location, "structured region must contain exactly one block")
            return
        block = region.blocks[0]
        available = set(outer_available)
        for argument in block.arguments:
            if argument.owner is not block:
                self._error(argument.location, "region block argument owner is inconsistent")
            self._register_value(argument)
            available.add(argument)
        self._verify_block(
            block,
            function,
            available,
            expected_terminator,
            yielded_types,
            loop_stack,
        )

    def _verify_block(
        self,
        block: Block,
        function: Function,
        available: set[Value],
        expected_terminator: OpCode,
        yielded_types: tuple[IRType, ...],
        loop_stack: LoopStack,
    ) -> None:
        if not block.operations:
            self._error(block.location, "block must end with an explicit terminator")
            return

        for position, operation in enumerate(block.operations):
            if operation.owner is not block:
                self._error(operation.location, "operation owner does not match its block")
            for operand in operation.operands:
                if operand not in available:
                    self._error(
                        operation.location,
                        f"operand %{operand.id} is not available in this SSA scope",
                    )

            self._verify_operation(operation, function, available, loop_stack)
            for index, result in enumerate(operation.results):
                self._register_value(result)
                if result.owner is not operation or result.result_index != index:
                    self._error(result.location, "result ownership metadata is inconsistent")
                if index >= len(operation.result_types) or result.type != operation.result_types[index]:
                    self._error(result.location, "result type does not match operation signature")
                available.add(result)

            if operation.opcode in TERMINATORS and position != len(block.operations) - 1:
                self._error(operation.location, "terminator must be the final operation in a block")

        terminator = block.operations[-1]
        is_loop_exit = terminator.opcode in (OpCode.BREAK, OpCode.CONTINUE)
        if terminator.opcode is not expected_terminator and not (
            is_loop_exit and self._loop_exit_allowed(terminator.opcode, loop_stack)
        ):
            self._error(
                terminator.location,
                f"block requires {expected_terminator.value} terminator",
            )
        self._verify_terminator(
            terminator,
            function,
            block,
            expected_terminator,
            yielded_types,
            loop_stack,
        )

    def _verify_operation(
        self,
        operation: Operation,
        function: Function,
        available: set[Value],
        loop_stack: LoopStack,
    ) -> None:
        if operation.id in self._operation_ids:
            self._error(operation.location, f"duplicate operation id {operation.id}")
        self._operation_ids.add(operation.id)
        if len(operation.results) != len(operation.result_types):
            self._error(operation.location, "operation result count is inconsistent")

        self._verify_no_physical_attributes(operation.attributes, operation.location)
        self._verify_auto_extent(operation)
        self._verify_optional_memory_order(operation)
        self._verify_regions(operation, function, available, loop_stack)
        self._verify_effects(operation, function)

        verifier = getattr(self, f"_verify_{operation.opcode.value}", None)
        if verifier is not None:
            verifier(operation)

    def _verify_auto_extent(self, operation: Operation) -> None:
        found = [
            (key, value)
            for key, value in operation.attributes.items()
            if self._contains_auto_extent(value)
        ]
        if not found:
            return
        if operation.opcode not in (OpCode.PARTITION, OpCode.STATE_STREAM):
            self._error(
                operation.location,
                "I.auto is only legal as partition/state_stream extent",
            )
            return
        if any(key != "extent" for key, _ in found):
            self._error(operation.location, "I.auto may only occupy the extent attribute")

    def _contains_auto_extent(self, value: object) -> bool:
        if isinstance(value, AutoExtent):
            return True
        if isinstance(value, (tuple, list)):
            return any(self._contains_auto_extent(item) for item in value)
        if isinstance(value, dict):
            return any(self._contains_auto_extent(item) for item in value.values())
        return False

    def _verify_optional_memory_order(self, operation: Operation) -> None:
        has_ordering = "ordering" in operation.attributes
        has_scope = "scope" in operation.attributes
        if not has_ordering and not has_scope:
            return
        memory_ops = {
            OpCode.VIEW_LOAD,
            OpCode.VIEW_STORE,
            OpCode.GATHER,
            OpCode.SCATTER_UNIQUE,
            OpCode.SCATTER_REDUCE,
            OpCode.BUFFER_LOAD,
            OpCode.BUFFER_STORE,
            OpCode.ATOMIC_ADD,
            OpCode.ATOMIC_CAS,
            OpCode.FENCE,
        }
        if operation.opcode not in memory_ops:
            self._error(operation.location, "memory ordering is only legal on memory effects")
            return
        if not isinstance(operation.attributes.get("ordering"), AtomicOrdering):
            self._error(operation.location, "memory ordering must be AtomicOrdering")
        if not isinstance(operation.attributes.get("scope"), MemoryScope):
            self._error(operation.location, "memory scope must be MemoryScope")

    def _verify_regions(
        self,
        operation: Operation,
        function: Function,
        available: set[Value],
        loop_stack: LoopStack,
    ) -> None:
        if operation.opcode not in REGION_OPS and operation.regions:
            self._error(operation.location, "operation cannot own nested regions")
            return
        expected_counts = {
            OpCode.PARALLEL: (1,),
            OpCode.ORDERED: (1,),
            OpCode.STATE_STREAM: (1,),
            OpCode.IF: (1, 2),
            OpCode.FOR: (1,),
            OpCode.WHILE: (2,),
        }
        if operation.opcode in expected_counts and len(operation.regions) not in expected_counts[operation.opcode]:
            self._error(operation.location, "structured operation has wrong region count")
            return

        if operation.opcode is OpCode.IF:
            if operation.result_types and len(operation.regions) != 2:
                self._error(
                    operation.location,
                    "value-producing if requires both then and else regions",
                )
            region_yields = (tuple(operation.result_types),) * len(operation.regions)
            region_loop_stacks = (loop_stack,) * len(operation.regions)
        elif operation.opcode is OpCode.WHILE:
            state_types = tuple(operation.result_types)
            region_yields = (state_types, state_types)
            region_loop_stacks = (
                loop_stack + (LoopFrame((), False, False),),
                loop_stack + (LoopFrame(state_types, True, True),),
            )
        elif operation.opcode is OpCode.STATE_STREAM:
            state_types = self._state_stream_types(operation)
            region_yields = (state_types,)
            region_loop_stacks = (
                loop_stack + (LoopFrame(state_types, False, True),),
            )
        elif operation.opcode in (OpCode.ORDERED, OpCode.FOR):
            state_types = tuple(operation.result_types)
            region_yields = (state_types,)
            region_loop_stacks = (
                loop_stack + (LoopFrame(state_types, True, True),),
            )
        elif operation.opcode is OpCode.PARALLEL:
            region_yields = ((),)
            region_loop_stacks = (loop_stack + (LoopFrame((), False, False),),)
        else:
            region_yields = ((),) * len(operation.regions)
            region_loop_stacks = (loop_stack,) * len(operation.regions)

        for index, region in enumerate(operation.regions):
            if region.owner is not operation:
                self._error(region.location, "region owner does not match operation")
            terminator = OpCode.YIELD
            if operation.opcode is OpCode.WHILE and index == 0:
                terminator = OpCode.CONDITION
            self._verify_region_arguments(operation, index, region)
            self._verify_region(
                region,
                function,
                available,
                terminator,
                region_yields[index],
                region_loop_stacks[index],
            )

    def _verify_region_arguments(
        self,
        operation: Operation,
        region_index: int,
        region: Region,
    ) -> None:
        if not region.blocks:
            return
        arguments = tuple(region.blocks[0].arguments)
        expected: tuple[IRType, ...]
        if operation.opcode in (OpCode.PARALLEL, OpCode.ORDERED, OpCode.FOR):
            expected = self._iteration_argument_types(operation)
            if operation.opcode in (OpCode.ORDERED, OpCode.FOR):
                expected = (*expected, *operation.result_types)
        elif operation.opcode is OpCode.STATE_STREAM:
            state_types = self._state_stream_types(operation)
            segment_type: IRType = RegionType()
            if operation.operands:
                axis_type = operation.operands[0].type
                if isinstance(axis_type, RegionType):
                    segment_type = axis_type
                elif isinstance(axis_type, DomainType):
                    segment_type = RegionType(axis_type.rank, "state_stream")
            expected = (segment_type, *state_types)
        elif operation.opcode is OpCode.WHILE:
            expected = tuple(operation.result_types)
        else:
            expected = ()
        if len(arguments) != len(expected):
            self._error(
                region.location,
                f"{operation.opcode.value} region {region_index} has wrong argument count",
            )
            return
        for argument, expected_type in zip(arguments, expected):
            if not types_compatible(argument.type, expected_type):
                self._error(
                    argument.location,
                    f"{operation.opcode.value} region argument type mismatch",
                )

    def _iteration_argument_types(self, operation: Operation) -> tuple[IRType, ...]:
        if not operation.operands:
            return ()
        source = operation.operands[0].type
        if isinstance(source, PartitionType):
            if source.mode is PartitionMode.COUNT:
                return (LogicalIndexType("partition"), source.region_type)
            return (source.region_type,)
        if isinstance(source, DomainType):
            return tuple(
                LogicalIndexType(f"domain_axis_{axis}") for axis in range(source.rank)
            )
        if isinstance(source, RegionType):
            return tuple(
                LogicalIndexType(f"{source.relation}_axis_{axis}")
                for axis in range(source.rank)
            )
        return ()

    def _state_stream_types(self, operation: Operation) -> tuple[IRType, ...]:
        state_count = operation.attributes.get("state_count")
        if (
            isinstance(state_count, bool)
            or not isinstance(state_count, int)
            or state_count <= 0
        ):
            return ()
        return tuple(
            operand.type for operand in operation.operands[1 : 1 + state_count]
        )

    def _verify_effects(self, operation: Operation, function: Function) -> None:
        parameter_by_value = {
            parameter.value: parameter.spec for parameter in function.parameters
        }
        if operation.opcode in EFFECTFUL_OPS and not operation.effects:
            self._error(operation.location, "effectful operation requires effect metadata")
        if operation.opcode not in EFFECTFUL_OPS and operation.opcode not in (
            OpCode.CALL,
            OpCode.GATHER,
        ):
            if operation.effects:
                self._error(operation.location, "pure operation cannot carry effects")
        for effect in operation.effects:
            if effect.target is not None and effect.target not in operation.operands:
                self._error(operation.location, "effect target must be an operation operand")

        required: tuple[tuple[EffectKind, ResourceKind, Value | None], ...] = ()
        target = operation.operands[0] if operation.operands else None
        if operation.opcode is OpCode.VIEW_LOAD:
            required = ((EffectKind.READ, ResourceKind.EXTERNAL_VIEW, target),)
        elif operation.opcode is OpCode.GATHER:
            spec = parameter_by_value.get(target)
            if spec is not None and spec.kind is ParameterKind.VIEW:
                required = ((EffectKind.READ, ResourceKind.EXTERNAL_VIEW, target),)
        elif operation.opcode in (
            OpCode.VIEW_STORE,
            OpCode.SCATTER_UNIQUE,
            OpCode.SCATTER_REDUCE,
        ):
            required = ((EffectKind.WRITE, ResourceKind.EXTERNAL_VIEW, target),)
        elif operation.opcode is OpCode.BUFFER_LOAD:
            required = ((EffectKind.READ, ResourceKind.LOGICAL_BUFFER, target),)
        elif operation.opcode is OpCode.BUFFER_STORE:
            required = ((EffectKind.WRITE, ResourceKind.LOGICAL_BUFFER, target),)
        elif operation.opcode in (OpCode.ATOMIC_ADD, OpCode.ATOMIC_CAS):
            resource = (
                ResourceKind.LOGICAL_BUFFER
                if target is not None and isinstance(target.type, BufferType)
                else ResourceKind.EXTERNAL_VIEW
            )
            required = ((EffectKind.ATOMIC, resource, target),)
        elif operation.opcode is OpCode.FENCE:
            required = ((EffectKind.FENCE, ResourceKind.ORDERING, None),)
        elif operation.opcode is OpCode.RANDOM:
            required = ((EffectKind.RNG, ResourceKind.RNG_STATE, None),)

        actual = tuple(
            (effect.kind, effect.resource, effect.target) for effect in operation.effects
        )
        if (required or operation.opcode is OpCode.GATHER) and actual != required:
            self._error(
                operation.location,
                f"{operation.opcode.value} carries an invalid effect signature",
            )
        for effect in operation.effects:
            if effect.resource is not ResourceKind.EXTERNAL_VIEW:
                continue
            spec = parameter_by_value.get(effect.target)
            if spec is None or spec.kind is not ParameterKind.VIEW:
                self._error(
                    operation.location,
                    "external view effect must target a view parameter",
                )
                continue
            if effect.kind is EffectKind.READ and spec.view_kind not in (
                ViewKind.IN,
                ViewKind.INOUT,
            ):
                self._error(operation.location, "read effect cannot target an Out-only view")
            if effect.kind is EffectKind.WRITE and spec.view_kind not in (
                ViewKind.OUT,
                ViewKind.INOUT,
            ):
                self._error(operation.location, "write effect cannot target an In-only view")
            if effect.kind is EffectKind.ATOMIC and spec.view_kind is not ViewKind.INOUT:
                self._error(operation.location, "atomic effect requires an InOut view")

    def _verify_terminator(
        self,
        operation: Operation,
        function: Function,
        block: Block,
        expected_terminator: OpCode,
        yielded_types: tuple[IRType, ...],
        loop_stack: LoopStack,
    ) -> None:
        if operation.result_types or operation.results:
            self._error(operation.location, "terminators cannot produce SSA results")
        if operation.opcode is OpCode.RETURN:
            if block.owner is not function.body:
                self._error(operation.location, "return is only legal in a function body")
            if len(operation.operands) != len(function.result_types):
                self._error(operation.location, "return operand count does not match function results")
            for operand, expected in zip(operation.operands, function.result_types):
                if not types_compatible(operand.type, expected):
                    self._error(operation.location, "return operand type mismatch")
        elif operation.opcode is OpCode.YIELD:
            if expected_terminator is not OpCode.YIELD:
                self._error(operation.location, "yield is not legal in this region")
            self._verify_operand_types(operation, yielded_types, "yield")
        elif operation.opcode is OpCode.CONDITION:
            if expected_terminator is not OpCode.CONDITION:
                self._error(operation.location, "condition is only legal in a while-before region")
            if len(operation.operands) != 1 + len(yielded_types):
                self._error(
                    operation.location,
                    "condition must return predicate followed by complete while state",
                )
            elif not is_boolean(operation.operands[0].type):
                self._error(operation.location, "condition predicate must be scalar bool")
            else:
                self._verify_operand_types(
                    operation,
                    yielded_types,
                    "condition state",
                    offset=1,
                )
        elif operation.opcode in (OpCode.BREAK, OpCode.CONTINUE):
            if not self._loop_exit_allowed(operation.opcode, loop_stack):
                self._error(operation.location, "loop exit is only legal inside an iteration region")
                return
            self._verify_operand_types(
                operation,
                loop_stack[-1].carried_types,
                operation.opcode.value,
            )

    def _loop_exit_allowed(self, opcode: OpCode, loop_stack: LoopStack) -> bool:
        if not loop_stack:
            return False
        frame = loop_stack[-1]
        if opcode is OpCode.BREAK:
            return frame.allow_break
        if opcode is OpCode.CONTINUE:
            return frame.allow_continue
        return False

    def _verify_operand_types(
        self,
        operation: Operation,
        expected_types: tuple[IRType, ...],
        subject: str,
        *,
        offset: int = 0,
    ) -> None:
        operands = operation.operands[offset:]
        if len(operands) != len(expected_types):
            self._error(
                operation.location,
                f"{subject} operand count does not match carried values",
            )
            return
        for operand, expected in zip(operands, expected_types):
            if not types_compatible(operand.type, expected):
                self._error(operation.location, f"{subject} operand type mismatch")

    def _require_operands(self, operation: Operation, count: int) -> bool:
        if len(operation.operands) != count:
            self._error(
                operation.location,
                f"{operation.opcode.value} expects {count} operands",
            )
            return False
        return True

    def _require_results(self, operation: Operation, count: int) -> bool:
        if len(operation.result_types) != count:
            self._error(
                operation.location,
                f"{operation.opcode.value} expects {count} results",
            )
            return False
        return True

    def _value_dtype(self, value_type: IRType) -> DType | None:
        if isinstance(value_type, ConstexprType):
            return self._value_dtype(value_type.value_type)
        if isinstance(value_type, (ScalarType, TensorType)):
            return value_type.dtype
        return None

    def _value_shape(self, value_type: IRType) -> tuple[object, ...] | None:
        if isinstance(value_type, ConstexprType):
            return self._value_shape(value_type.value_type)
        if isinstance(value_type, ScalarType):
            return ()
        if isinstance(value_type, TensorType):
            return tuple(value_type.shape)
        return None

    def _is_boolean_value(self, value_type: IRType) -> bool:
        dtype = self._value_dtype(value_type)
        return dtype is not None and dtype.category is DTypeCategory.BOOL

    def _is_numeric_value(self, value_type: IRType) -> bool:
        dtype = self._value_dtype(value_type)
        return dtype is not None and dtype.category is not DTypeCategory.BOOL

    def _is_floating_value(self, value_type: IRType) -> bool:
        dtype = self._value_dtype(value_type)
        return dtype is not None and dtype.category in (
            DTypeCategory.FLOAT,
            DTypeCategory.BFLOAT,
        )

    def _broadcast_value_shape(
        self,
        value_types: tuple[IRType, ...],
        operation: Operation,
        subject: str,
    ) -> tuple[object, ...] | None:
        shapes = [self._value_shape(value_type) for value_type in value_types]
        if any(shape is None for shape in shapes):
            self._error(operation.location, f"{subject} requires scalar or tensor values")
            return None
        result: tuple[object, ...] = ()
        try:
            for shape in shapes:
                assert shape is not None
                result = tuple(broadcast_shape(result, shape))
        except ValueError:
            self._error(operation.location, f"{subject} shapes cannot broadcast")
            return None
        return result

    def _matches_dtype_shape(
        self,
        value_type: IRType,
        dtype: DType,
        shape: tuple[object, ...],
    ) -> bool:
        if shape:
            return (
                isinstance(value_type, TensorType)
                and value_type.dtype == dtype
                and tuple(value_type.shape) == shape
            )
        return isinstance(value_type, ScalarType) and value_type.dtype == dtype

    def _verify_callable_schema(
        self,
        operation: Operation,
        value: object,
        operand_types: tuple[IRType, ...],
        result_types: tuple[IRType, ...],
        subject: str,
    ) -> None:
        builtins = {
            "add",
            "subtract",
            "multiply",
            "maximum",
            "minimum",
            "logical_and",
            "logical_or",
        }
        builtin_name = value.value if isinstance(value, BinaryOperator) else value
        if isinstance(builtin_name, str) and builtin_name in builtins:
            if any(isinstance(value_type, RecordType) for value_type in operand_types):
                self._error(operation.location, f"{subject} for records requires a helper")
                return
            logical = builtin_name in ("logical_and", "logical_or")
            if logical and any(
                not self._is_boolean_value(value_type) for value_type in operand_types
            ):
                self._error(operation.location, f"{subject} logical builtin requires bool")
            if not logical and any(
                not self._is_numeric_value(value_type) for value_type in operand_types
            ):
                self._error(operation.location, f"{subject} arithmetic builtin requires numeric values")
            return
        helper = self._symbols.get(value) if isinstance(value, str) else None
        if helper is None or helper.kind is not FunctionKind.HELPER:
            self._error(operation.location, f"{subject} must name a builtin or helper")
            return
        parameter_types = tuple(parameter.spec.type for parameter in helper.parameters)
        if len(parameter_types) != len(operand_types) or any(
            not types_compatible(actual, expected)
            for actual, expected in zip(parameter_types, operand_types)
        ):
            self._error(operation.location, f"{subject} helper operand schema mismatch")
        if len(helper.result_types) != len(result_types) or any(
            not types_compatible(actual, expected)
            for actual, expected in zip(helper.result_types, result_types)
        ):
            self._error(operation.location, f"{subject} helper result schema mismatch")

    def _verify_constant(self, operation: Operation) -> None:
        self._require_operands(operation, 0)
        self._require_results(operation, 1)
        if "value" not in operation.attributes:
            self._error(operation.location, "constant requires value attribute")
        if operation.result_types and not isinstance(
            operation.result_types[0], (ScalarType, ConstexprType)
        ):
            self._error(operation.location, "constant result must be scalar or constexpr")

    def _verify_dim(self, operation: Operation) -> None:
        if not self._require_operands(operation, 1) or not self._require_results(operation, 1):
            return
        source = operation.operands[0].type
        if not isinstance(source, TensorType):
            self._error(operation.location, "dim source must be a tensor")
            return
        axis = operation.attributes.get("axis")
        if not isinstance(axis, int) or isinstance(axis, bool) or not -source.rank <= axis < source.rank:
            self._error(operation.location, "dim axis is outside tensor rank")
        if not is_integer(operation.result_types[0]):
            self._error(operation.location, "dim result must be integer/index")

    def _verify_domain(self, operation: Operation) -> None:
        if len(operation.operands) not in (2, 3):
            self._error(operation.location, "domain expects start, stop, and optional step")
        if not self._require_results(operation, 1):
            return
        if any(not is_integer(operand.type) for operand in operation.operands):
            self._error(operation.location, "domain bounds must be integer/index")
        result = operation.result_types[0]
        if not isinstance(result, DomainType):
            self._error(operation.location, "domain result must have DomainType")
        elif result.rank != 1 or result.flavor not in (
            DomainFlavor.DENSE,
            DomainFlavor.STRIDED,
            DomainFlavor.RUNTIME,
        ):
            self._error(operation.location, "domain result must be rank-one dense/strided/runtime")
        elif len(operation.operands) == 3 and result.flavor is DomainFlavor.DENSE:
            self._error(operation.location, "explicit domain step requires strided/runtime flavor")

    def _verify_domain_product(self, operation: Operation) -> None:
        if not operation.operands:
            self._error(operation.location, "domain_product requires at least one domain")
        if any(not isinstance(operand.type, DomainType) for operand in operation.operands):
            self._error(operation.location, "domain_product operands must be domains")
        if self._require_results(operation, 1):
            result = operation.result_types[0]
            if not isinstance(result, DomainType) or result.flavor is not DomainFlavor.PRODUCT:
                self._error(operation.location, "domain_product result must be product domain")
            elif result.rank != sum(
                operand.type.rank
                for operand in operation.operands
                if isinstance(operand.type, DomainType)
            ):
                self._error(operation.location, "product domain rank must equal component ranks")

    def _verify_partition(self, operation: Operation) -> None:
        if not operation.operands:
            self._error(operation.location, "partition requires a logical domain")
            return
        if not isinstance(operation.operands[0].type, (DomainType, RegionType)):
            self._error(operation.location, "partition source must be domain or region")
        if not self._require_results(operation, 1):
            return
        result = operation.result_types[0]
        if not isinstance(result, PartitionType):
            self._error(operation.location, "partition result must have PartitionType")
            return
        mode = operation.attributes.get("mode")
        if mode is not result.mode:
            self._error(operation.location, "partition mode attribute and result type differ")
        source = operation.operands[0].type
        source_rank = source.rank if isinstance(source, (DomainType, RegionType)) else None
        if source_rank is not None and result.region_type.rank != source_rank:
            self._error(operation.location, "partition result region rank must match its source")
        if (
            isinstance(source, RegionType)
            and result.region_type.relation != source.relation
        ):
            self._error(
                operation.location,
                "partitioning a region must preserve its logical index relation",
            )
        extent = operation.attributes.get("extent")
        if mode is PartitionMode.COUNT:
            if "extent" in operation.attributes:
                self._error(operation.location, "count partition cannot carry extent")
            if len(operation.operands) != 2:
                self._error(operation.location, "count partition requires a source-visible count operand")
            elif not is_integer(operation.operands[1].type):
                self._error(operation.location, "partition count must be integer/index")
        elif mode is PartitionMode.EXTENT:
            has_auto = isinstance(extent, AutoExtent)
            has_operand = "extent" not in operation.attributes and len(operation.operands) == 2
            if has_auto == has_operand:
                self._error(operation.location, "extent partition requires exactly one auto or value extent")
            if "extent" in operation.attributes and not has_auto:
                self._error(operation.location, "partition extent attribute must be I.auto")
            if len(operation.operands) not in (1, 2):
                self._error(operation.location, "extent partition has wrong operand count")
            if has_operand and not is_integer(operation.operands[1].type):
                self._error(operation.location, "partition extent must be integer/index")
        else:
            self._error(operation.location, "partition requires extent or count mode")

    def _verify_indices(self, operation: Operation) -> None:
        if not self._require_operands(operation, 1) or not self._require_results(operation, 1):
            return
        if not isinstance(operation.operands[0].type, RegionType):
            self._error(operation.location, "indices expects a logical region")
        result = operation.result_types[0]
        if not isinstance(result, TensorType) or result.dtype != intent_index:
            self._error(operation.location, "indices result must be an index tensor")
        elif isinstance(operation.operands[0].type, RegionType) and result.rank != operation.operands[0].type.rank:
            self._error(operation.location, "indices result rank must match logical region rank")

    def _verify_parallel(self, operation: Operation) -> None:
        self._verify_iteration(operation, allow_state=False)

    def _verify_ordered(self, operation: Operation) -> None:
        self._verify_iteration(operation, allow_state=True)

    def _verify_for(self, operation: Operation) -> None:
        self._verify_iteration(operation, allow_state=True)

    def _verify_iteration(self, operation: Operation, *, allow_state: bool) -> None:
        if not operation.operands:
            self._error(operation.location, "iteration requires a source domain")
            return
        if not isinstance(
            operation.operands[0].type,
            (DomainType, RegionType, PartitionType),
        ):
            self._error(operation.location, "iteration source must be domain, region, or partition")
        if not allow_state:
            if len(operation.operands) != 1:
                self._error(operation.location, "parallel cannot carry loop state")
            self._require_results(operation, 0)
        else:
            initial_state = operation.operands[1:]
            if len(initial_state) != len(operation.result_types) or any(
                not types_compatible(value.type, result_type)
                for value, result_type in zip(initial_state, operation.result_types)
            ):
                self._error(
                    operation.location,
                    f"{operation.opcode.value} initial and result state schemas must match",
                )
        if len(operation.regions) != 1:
            self._error(operation.location, "iteration requires exactly one body region")

    def _verify_state_stream(self, operation: Operation) -> None:
        if len(operation.operands) < 2:
            self._error(operation.location, "state_stream requires axis and initial state")
            return
        if not isinstance(operation.operands[0].type, (DomainType, RegionType)):
            self._error(operation.location, "state_stream axis must be domain or region")
        state_count = operation.attributes.get("state_count")
        if (
            isinstance(state_count, bool)
            or not isinstance(state_count, int)
            or state_count <= 0
        ):
            self._error(operation.location, "state_stream requires positive state_count")
            return
        state_types = tuple(
            operand.type for operand in operation.operands[1 : 1 + state_count]
        )
        if len(state_types) != state_count:
            self._error(operation.location, "state_stream is missing initial state operands")
        if len(operation.result_types) != state_count or any(
            not types_compatible(actual, expected)
            for actual, expected in zip(operation.result_types, state_types)
        ):
            self._error(operation.location, "state_stream results must match carried state")
        extent = operation.attributes.get("extent")
        has_auto = isinstance(extent, AutoExtent)
        extent_operand_index = operation.attributes.get("extent_operand_index")
        has_operand = extent_operand_index is not None
        if has_auto == has_operand:
            self._error(operation.location, "state_stream requires exactly one auto or value extent")
        expected_operands = 1 + state_count + (1 if has_operand else 0)
        if len(operation.operands) != expected_operands:
            self._error(operation.location, "state_stream has wrong operand count")
        if has_operand:
            if (
                isinstance(extent_operand_index, bool)
                or not isinstance(extent_operand_index, int)
            ):
                self._error(
                    operation.location,
                    "state_stream extent_operand_index must be an integer",
                )
            elif extent_operand_index != 1 + state_count:
                self._error(
                    operation.location,
                    "state_stream extent operand must follow all initial state operands",
                )
            elif extent_operand_index >= len(operation.operands) or not is_integer(
                operation.operands[extent_operand_index].type
            ):
                self._error(operation.location, "state_stream extent must be integer/index")
        if "extent" in operation.attributes and not has_auto:
            self._error(operation.location, "state_stream extent attribute must be I.auto")

    def _verify_if(self, operation: Operation) -> None:
        if len(operation.operands) != 1 or not is_boolean(operation.operands[0].type):
            self._error(operation.location, "if condition must be boolean")

    def _verify_while(self, operation: Operation) -> None:
        if len(operation.operands) != len(operation.result_types) or any(
            not types_compatible(value.type, result_type)
            for value, result_type in zip(operation.operands, operation.result_types)
        ):
            self._error(operation.location, "while operands and results must carry the same state types")

    def _verify_yield(self, operation: Operation) -> None:
        self._require_results(operation, 0)

    def _verify_condition(self, operation: Operation) -> None:
        self._require_results(operation, 0)
        if not operation.operands or not is_boolean(operation.operands[0].type):
            self._error(operation.location, "while condition terminator requires boolean first operand")

    def _verify_view_load(self, operation: Operation) -> None:
        if not operation.operands or not isinstance(operation.operands[0].type, TensorType):
            self._error(operation.location, "view_load requires tensor view source")
            return
        relation = operation.attributes.get("index")
        if not isinstance(relation, IndexRelation):
            self._error(operation.location, "view_load requires explicit IndexRelation")
            indexed_shape = None
        else:
            indexed_shape = self._verify_index_relation(
                operation, operation.operands[0].type, relation
            )
        if self._require_results(operation, 1):
            result = operation.result_types[0]
            if indexed_shape is not None and not self._matches_dtype_shape(
                result, operation.operands[0].type.dtype, indexed_shape
            ):
                self._error(
                    operation.location,
                    "view_load result must match indexed view dtype and shape",
                )

    def _verify_view_store(self, operation: Operation) -> None:
        if len(operation.operands) < 2 or not isinstance(operation.operands[0].type, TensorType):
            self._error(operation.location, "view_store requires tensor view and value")
            return
        relation = operation.attributes.get("index")
        if not isinstance(relation, IndexRelation):
            self._error(operation.location, "view_store requires explicit IndexRelation")
            indexed_shape = None
        else:
            indexed_shape = self._verify_index_relation(
                operation,
                operation.operands[0].type,
                relation,
                excluded_operand_positions=(1,),
            )
        if self._value_dtype(operation.operands[1].type) != operation.operands[0].type.dtype:
            self._error(operation.location, "view_store value must match view element dtype")
        if indexed_shape is not None:
            self._verify_indexed_value_shape(
                operation, operation.operands[1].type, indexed_shape, "view_store"
            )
        self._require_results(operation, 0)

    def _verify_index_relation(
        self,
        operation: Operation,
        source: TensorType | BufferType,
        relation: IndexRelation,
        *,
        excluded_operand_positions: tuple[int, ...] = (),
    ) -> tuple[object, ...] | None:
        valid = True
        consuming_terms = sum(
            term.kind is not IndexTermKind.NEW_AXIS for term in relation.terms
        )
        if consuming_terms != len(source.shape):
            self._error(
                operation.location,
                "index relation must consume every source axis exactly once",
            )
            valid = False
        result_shape: list[object] = []
        source_axis = 0
        for term in relation.terms:
            if term.kind is IndexTermKind.NEW_AXIS:
                result_shape.append(StaticDim(1))
                continue
            if source_axis >= len(source.shape):
                valid = False
                continue
            source_dimension = source.shape[source_axis]
            referenced_values: list[Value] = []
            for position in term.operand_positions:
                if position is None:
                    continue
                if (
                    position == 0
                    or position in excluded_operand_positions
                    or position >= len(operation.operands)
                ):
                    self._error(operation.location, "index relation references invalid operand")
                    valid = False
                    continue
                index_type = operation.operands[position].type
                referenced_values.append(operation.operands[position])
                if term.kind is IndexTermKind.REGION_INDEX:
                    if not isinstance(index_type, (DomainType, RegionType)):
                        self._error(
                            operation.location,
                            "logical region index term requires DomainType/RegionType operand",
                        )
                        valid = False
                elif term.kind is IndexTermKind.SLICE:
                    if not is_integer(index_type):
                        self._error(
                            operation.location,
                            "dynamic slice bounds must be scalar integer/index",
                        )
                        valid = False
                elif isinstance(index_type, TensorType):
                    if index_type.dtype.category not in (
                        DTypeCategory.SIGNED_INTEGER,
                        DTypeCategory.UNSIGNED_INTEGER,
                        DTypeCategory.INDEX,
                    ):
                        self._error(operation.location, "tensor index operand must be integer")
                        valid = False
                elif not is_integer(index_type):
                    self._error(operation.location, "index operand must be integer/index")
                    valid = False

            if term.kind is IndexTermKind.FULL_SLICE:
                result_shape.append(source_dimension)
            elif term.kind is IndexTermKind.STATIC_INDEX:
                static_index = term.static_values[0]
                if isinstance(source_dimension, StaticDim) and not (
                    -source_dimension.value <= static_index < source_dimension.value
                ):
                    self._error(
                        operation.location,
                        "static index is outside the source dimension",
                    )
                    valid = False
            elif term.kind is IndexTermKind.SLICE:
                result_shape.append(
                    DynamicDim(f"slice_{operation.operands[0].id}_{source_axis}")
                )
            elif term.kind is IndexTermKind.REGION_INDEX and referenced_values:
                region = referenced_values[0]
                rank = getattr(region.type, "rank", 1)
                result_shape.extend(
                    DynamicDim(f"region_{region.id}_{axis}") for axis in range(rank)
                )
            elif term.kind is IndexTermKind.VALUE_INDEX and referenced_values:
                index_type = referenced_values[0].type
                if isinstance(index_type, TensorType):
                    result_shape.extend(index_type.shape)
            source_axis += 1
        return tuple(result_shape) if valid else None

    def _verify_indexed_value_shape(
        self,
        operation: Operation,
        value_type: IRType,
        indexed_shape: tuple[object, ...],
        subject: str,
    ) -> None:
        value_shape = self._value_shape(value_type)
        if value_shape is None:
            self._error(operation.location, f"{subject} value must be scalar/tensor")
            return
        try:
            result = tuple(broadcast_shape(value_shape, indexed_shape))
        except ValueError:
            self._error(operation.location, f"{subject} value cannot broadcast to indexed shape")
            return
        if result != indexed_shape:
            self._error(operation.location, f"{subject} value cannot broadcast to indexed shape")

    def _verify_reshape(self, operation: Operation) -> None:
        if not self._require_operands(operation, 1) or not self._require_results(operation, 1):
            return
        source = operation.operands[0].type
        result = operation.result_types[0]
        if not isinstance(source, TensorType) or not isinstance(result, TensorType):
            self._error(operation.location, "reshape requires tensor input and result")
            return
        if source.dtype != result.dtype:
            self._error(operation.location, "reshape cannot change element dtype")
        source_numel = static_numel(source.shape)
        result_numel = static_numel(result.shape)
        if source_numel is not None and result_numel is not None and source_numel != result_numel:
            self._error(operation.location, "reshape changes static element count")

    def _verify_transpose(self, operation: Operation) -> None:
        if not self._require_operands(operation, 1) or not self._require_results(operation, 1):
            return
        source = operation.operands[0].type
        result = operation.result_types[0]
        permutation = operation.attributes.get("permutation")
        if not isinstance(source, TensorType) or not isinstance(result, TensorType):
            self._error(operation.location, "transpose requires tensor input and result")
            return
        if (
            not isinstance(permutation, tuple)
            or any(isinstance(axis, bool) or not isinstance(axis, int) for axis in permutation)
            or sorted(permutation) != list(range(source.rank))
        ):
            self._error(operation.location, "transpose permutation must cover every input axis")
            return
        expected = tuple(source.shape[axis] for axis in permutation)
        if result.shape != expected or result.dtype != source.dtype:
            self._error(operation.location, "transpose result type does not match permutation")

    def _verify_broadcast(self, operation: Operation) -> None:
        if not self._require_operands(operation, 1) or not self._require_results(operation, 1):
            return
        source = operation.operands[0].type
        result = operation.result_types[0]
        source_shape = self._value_shape(source)
        if source_shape is None or not isinstance(result, TensorType):
            self._error(operation.location, "broadcast requires scalar/tensor input and tensor result")
            return
        try:
            shape = broadcast_shape(source_shape, result.shape)
        except ValueError:
            self._error(operation.location, "value and result shapes are not broadcast-compatible")
            return
        if shape != result.shape:
            self._error(operation.location, "broadcast result shape is inconsistent")
        if self._value_dtype(source) != result.dtype:
            self._error(operation.location, "broadcast cannot change element dtype")

    def _verify_full(self, operation: Operation) -> None:
        if not self._require_operands(operation, 1) or not self._require_results(operation, 1):
            return
        result = operation.result_types[0]
        if not isinstance(result, TensorType):
            self._error(operation.location, "full result must be a tensor")
            return
        fill = operation.operands[0].type
        if not isinstance(fill, ScalarType) or fill.dtype != result.dtype:
            self._error(operation.location, "full fill value must match result element dtype")

    def _verify_zeros(self, operation: Operation) -> None:
        self._require_operands(operation, 0)
        if self._require_results(operation, 1) and not isinstance(operation.result_types[0], TensorType):
            self._error(operation.location, "zeros result must be a tensor")

    def _verify_make_record(self, operation: Operation) -> None:
        if not self._require_results(operation, 1):
            return
        result = operation.result_types[0]
        names = operation.attributes.get("fields")
        if not isinstance(result, RecordType) or not isinstance(names, tuple):
            self._error(operation.location, "record requires field names and RecordType")
            return
        if tuple(name for name, _ in result.fields) != names:
            self._error(operation.location, "record field order does not match result type")
        if len(operation.operands) != len(result.fields) or any(
            not types_compatible(operand.type, field_type)
            for operand, (_, field_type) in zip(operation.operands, result.fields)
        ):
            self._error(operation.location, "record operands do not match field types")

    def _verify_extract(self, operation: Operation) -> None:
        if not self._require_operands(operation, 1) or not self._require_results(operation, 1):
            return
        source = operation.operands[0].type
        key = operation.attributes.get("key")
        expected: IRType | None = None
        if isinstance(source, RecordType) and isinstance(key, str):
            expected = dict(source.fields).get(key)
        if expected is None or not types_compatible(operation.result_types[0], expected):
            self._error(operation.location, "extract key or result type is invalid")

    def _verify_unary(self, operation: Operation) -> None:
        if not self._require_operands(operation, 1) or not self._require_results(operation, 1):
            return
        operator = operation.attributes.get("operator")
        if not isinstance(operator, UnaryOperator):
            self._error(operation.location, "unary op requires UnaryOperator")
            return
        if not types_compatible(operation.operands[0].type, operation.result_types[0]):
            self._error(operation.location, "unary result type must match operand")
        operand_type = operation.operands[0].type
        if operator is UnaryOperator.NOT:
            if not self._is_boolean_value(operand_type):
                self._error(operation.location, "logical not requires bool values")
        elif operator in (UnaryOperator.EXP, UnaryOperator.EXP2, UnaryOperator.LOG, UnaryOperator.RSQRT):
            if not self._is_floating_value(operand_type):
                self._error(operation.location, f"{operator.value} requires floating-point values")
        elif not self._is_numeric_value(operand_type):
            self._error(operation.location, "negate requires numeric values")

    def _verify_binary(self, operation: Operation) -> None:
        if not self._require_operands(operation, 2) or not self._require_results(operation, 1):
            return
        operator = operation.attributes.get("operator")
        if not isinstance(operator, BinaryOperator):
            self._error(operation.location, "binary op requires BinaryOperator")
            return
        lhs, rhs = operation.operands
        result = operation.result_types[0]
        lhs_dtype = self._value_dtype(lhs.type)
        rhs_dtype = self._value_dtype(rhs.type)
        if lhs_dtype is None or rhs_dtype is None or lhs_dtype != rhs_dtype:
            self._error(operation.location, "binary operands must have one explicit element dtype")
            return
        logical = operator in (BinaryOperator.LOGICAL_AND, BinaryOperator.LOGICAL_OR)
        if logical and not self._is_boolean_value(lhs.type):
            self._error(operation.location, "logical binary operators require bool values")
        if not logical and not self._is_numeric_value(lhs.type):
            self._error(operation.location, "arithmetic binary operators require numeric values")
        expected_shape = self._broadcast_value_shape(
            (lhs.type, rhs.type), operation, "binary operands"
        )
        if expected_shape is not None and not self._matches_dtype_shape(
            result, lhs_dtype, expected_shape
        ):
            self._error(operation.location, "binary result type is inconsistent with broadcast")

    def _verify_compare(self, operation: Operation) -> None:
        if not self._require_operands(operation, 2) or not self._require_results(operation, 1):
            return
        predicate = operation.attributes.get("predicate")
        if not isinstance(predicate, ComparePredicate):
            self._error(operation.location, "compare requires ComparePredicate")
            return
        lhs_type, rhs_type = (value.type for value in operation.operands)
        lhs_dtype = self._value_dtype(lhs_type)
        rhs_dtype = self._value_dtype(rhs_type)
        if lhs_dtype is None or lhs_dtype != rhs_dtype:
            self._error(operation.location, "compare operands must have one explicit dtype")
            return
        if predicate not in (ComparePredicate.EQ, ComparePredicate.NE) and not self._is_numeric_value(lhs_type):
            self._error(operation.location, "ordered comparison requires numeric values")
        expected_shape = self._broadcast_value_shape(
            (lhs_type, rhs_type), operation, "compare operands"
        )
        result = operation.result_types[0]
        if expected_shape is not None and not self._matches_dtype_shape(
            result, intent_bool, expected_shape
        ):
            self._error(operation.location, "compare result must be broadcast-shaped bool")

    def _verify_select(self, operation: Operation) -> None:
        if not self._require_operands(operation, 3) or not self._require_results(operation, 1):
            return
        condition, true_value, false_value = operation.operands
        if not self._is_boolean_value(condition.type):
            self._error(operation.location, "select condition must be boolean")
        true_dtype = self._value_dtype(true_value.type)
        false_dtype = self._value_dtype(false_value.type)
        if true_dtype is None or true_dtype != false_dtype:
            self._error(operation.location, "select branches must have one explicit dtype")
            return
        expected_shape = self._broadcast_value_shape(
            (condition.type, true_value.type, false_value.type),
            operation,
            "select operands",
        )
        if expected_shape is not None and not self._matches_dtype_shape(
            operation.result_types[0], true_dtype, expected_shape
        ):
            self._error(operation.location, "select result type is inconsistent with broadcast")

    def _verify_cast(self, operation: Operation) -> None:
        if not self._require_operands(operation, 1) or not self._require_results(operation, 1):
            return
        source = operation.operands[0].type
        result = operation.result_types[0]
        source_shape = self._value_shape(source)
        result_shape = self._value_shape(result)
        if source_shape is None or result_shape is None:
            self._error(operation.location, "cast requires scalar or tensor source/result")
        elif source_shape != result_shape:
            self._error(operation.location, "cast cannot change value shape")

    def _verify_mask(self, operation: Operation) -> None:
        if not self._require_operands(operation, 3) or not self._require_results(operation, 1):
            return
        value, predicate, fill = operation.operands
        value_dtype = self._value_dtype(value.type)
        fill_dtype = self._value_dtype(fill.type)
        if value_dtype is None or value_dtype != fill_dtype:
            self._error(operation.location, "mask fill must have the value element dtype")
            return
        if not self._is_boolean_value(predicate.type):
            self._error(operation.location, "mask predicate must be boolean")
        expected_shape = self._broadcast_value_shape(
            (value.type, predicate.type, fill.type), operation, "mask operands"
        )
        if expected_shape is not None and not self._matches_dtype_shape(
            operation.result_types[0], value_dtype, expected_shape
        ):
            self._error(operation.location, "mask result type is inconsistent with broadcast")

    def _verify_reduce(self, operation: Operation) -> None:
        if not self._require_operands(operation, 2) or not self._require_results(operation, 1):
            self._error(operation.location, "reduce requires input and identity")
            return
        source = operation.operands[0].type
        if isinstance(source, RecordType):
            self._verify_record_reduction(operation, source, scan=False)
            return
        if not isinstance(source, TensorType):
            self._error(operation.location, "reduce input must be tensor or tensor record")
            return
        axes = self._normalize_axes(operation.attributes.get("axes"), source.rank, operation)
        if axes is None:
            return
        acc_dtype = operation.attributes.get("acc_dtype")
        if not isinstance(acc_dtype, DType):
            self._error(operation.location, "reduce requires explicit acc_dtype")
            return
        identity = operation.operands[1].type
        if not isinstance(identity, ScalarType) or identity.dtype != acc_dtype:
            self._error(operation.location, "reduce identity must have accumulator dtype")
        result = operation.result_types[0]
        expected_shape = tuple(dim for index, dim in enumerate(source.shape) if index not in axes)
        if not self._matches_dtype_shape(result, acc_dtype, tuple(expected_shape)):
            self._error(operation.location, "reduce result type is inconsistent with axes and accumulator")
        combine = operation.attributes.get("combine")
        accumulator_type = ScalarType(acc_dtype)
        self._verify_callable_schema(
            operation,
            combine,
            (accumulator_type, accumulator_type),
            (accumulator_type,),
            "reduce combiner",
        )

    def _verify_scan(self, operation: Operation) -> None:
        if not self._require_operands(operation, 2) or not self._require_results(operation, 1):
            self._error(operation.location, "scan requires input and identity")
            return
        source = operation.operands[0].type
        result = operation.result_types[0]
        if isinstance(source, RecordType):
            self._verify_record_reduction(operation, source, scan=True)
            return
        if not isinstance(source, TensorType) or not isinstance(result, TensorType):
            self._error(operation.location, "scan input/result must be tensors or tensor records")
            return
        axes = self._normalize_axes(operation.attributes.get("axis"), source.rank, operation)
        if axes is not None and len(axes) != 1:
            self._error(operation.location, "scan requires exactly one positional axis")
        acc_dtype = operation.attributes.get("acc_dtype")
        if not isinstance(acc_dtype, DType):
            self._error(operation.location, "scan requires explicit acc_dtype")
        else:
            identity = operation.operands[1].type
            if not isinstance(identity, ScalarType) or identity.dtype != acc_dtype:
                self._error(operation.location, "scan identity must have accumulator dtype")
            if result.shape != source.shape or result.dtype != acc_dtype:
                self._error(operation.location, "scan preserves shape and returns accumulator dtype")
        if not isinstance(operation.attributes.get("inclusive"), bool):
            self._error(operation.location, "scan requires inclusive boolean attribute")
        if isinstance(acc_dtype, DType):
            accumulator_type = ScalarType(acc_dtype)
            self._verify_callable_schema(
                operation,
                operation.attributes.get("combine"),
                (accumulator_type, accumulator_type),
                (accumulator_type,),
                "scan combiner",
            )

    def _verify_record_reduction(
        self,
        operation: Operation,
        source: RecordType,
        *,
        scan: bool,
    ) -> None:
        source_fields = tuple(source.fields)
        if not source_fields or any(
            not isinstance(field_type, TensorType) for _, field_type in source_fields
        ):
            self._error(
                operation.location,
                "record reduction source fields must all be tensors",
            )
            return
        tensor_fields = tuple(
            field_type for _, field_type in source_fields if isinstance(field_type, TensorType)
        )
        shape = tensor_fields[0].shape
        if any(field.shape != shape for field in tensor_fields[1:]):
            self._error(operation.location, "record reduction fields must share one shape")
            return
        axis_attribute = "axis" if scan else "axes"
        axes = self._normalize_axes(
            operation.attributes.get(axis_attribute),
            len(shape),
            operation,
        )
        if axes is None:
            return
        if scan and len(axes) != 1:
            self._error(operation.location, "record scan requires exactly one axis")
        identity = operation.operands[1].type
        if not isinstance(identity, RecordType):
            self._error(operation.location, "record reduction identity must be a record")
            return
        if tuple(name for name, _ in identity.fields) != tuple(
            name for name, _ in source_fields
        ) or any(not isinstance(field_type, ScalarType) for _, field_type in identity.fields):
            self._error(
                operation.location,
                "record reduction identity fields must be scalar and name-aligned",
            )
            return
        for (_, source_type), (_, identity_type) in zip(
            source_fields, identity.fields
        ):
            assert isinstance(source_type, TensorType)
            assert isinstance(identity_type, ScalarType)
            if source_type.dtype != identity_type.dtype:
                self._error(
                    operation.location,
                    "record reduction requires explicit per-field casts before accumulation",
                )
        if "acc_dtype" in operation.attributes:
            self._error(
                operation.location,
                "record reduction uses identity field dtypes, not one acc_dtype",
            )
        result = operation.result_types[0]
        if not isinstance(result, RecordType):
            self._error(operation.location, "record reduction result must be a record")
            return
        result_shape = shape if scan else tuple(
            dim for index, dim in enumerate(shape) if index not in axes
        )
        expected_fields: list[tuple[str, IRType]] = []
        for name, field_type in identity.fields:
            assert isinstance(field_type, ScalarType)
            expected_type: IRType = field_type
            if result_shape:
                expected_type = TensorType(field_type.dtype, result_shape)
            expected_fields.append((name, expected_type))
        expected_result = RecordType(tuple(expected_fields))
        if not types_compatible(result, expected_result):
            self._error(operation.location, "record reduction result schema is inconsistent")
        if scan and not isinstance(operation.attributes.get("inclusive"), bool):
            self._error(operation.location, "record scan requires inclusive boolean attribute")
        self._verify_callable_schema(
            operation,
            operation.attributes.get("combine"),
            (identity, identity),
            (identity,),
            "record reduction combiner",
        )

    def _verify_contract(self, operation: Operation) -> None:
        if not self._require_operands(operation, 2) or not self._require_results(operation, 1):
            return
        lhs = operation.operands[0].type
        rhs = operation.operands[1].type
        result = operation.result_types[0]
        if not all(isinstance(value, TensorType) for value in (lhs, rhs, result)):
            self._error(operation.location, "contract operands and result must be tensors")
            return
        assert isinstance(lhs, TensorType) and isinstance(rhs, TensorType) and isinstance(result, TensorType)
        if not self._is_numeric_value(lhs) or not self._is_numeric_value(rhs):
            self._error(operation.location, "contract operands must have numeric element dtypes")
        pairs = operation.attributes.get("reduce")
        if not isinstance(pairs, tuple) or not pairs:
            self._error(operation.location, "contract requires positional reduction-axis pairs")
            return
        lhs_axes: set[int] = set()
        rhs_axes: set[int] = set()
        for pair in pairs:
            if not (
                isinstance(pair, tuple)
                and len(pair) == 2
                and all(isinstance(axis, int) and not isinstance(axis, bool) for axis in pair)
            ):
                self._error(operation.location, "contract reduction pairs must be integer axis pairs")
                return
            lhs_axis, rhs_axis = pair
            if not 0 <= lhs_axis < lhs.rank or not 0 <= rhs_axis < rhs.rank:
                self._error(operation.location, "contract reduction axis is outside operand rank")
                return
            if lhs_axis in lhs_axes or rhs_axis in rhs_axes:
                self._error(operation.location, "contract reduction axes must be unique")
            lhs_axes.add(lhs_axis)
            rhs_axes.add(rhs_axis)
            if lhs.shape[lhs_axis] != rhs.shape[rhs_axis] and not (
                isinstance(lhs.shape[lhs_axis], DynamicDim)
                or isinstance(rhs.shape[rhs_axis], DynamicDim)
            ):
                self._error(operation.location, "contract paired dimensions are incompatible")
        expected = tuple(dim for index, dim in enumerate(lhs.shape) if index not in lhs_axes) + tuple(
            dim for index, dim in enumerate(rhs.shape) if index not in rhs_axes
        )
        if result.shape != expected:
            self._error(operation.location, "contract result shape does not match free axes")
        acc_dtype = operation.attributes.get("acc_dtype")
        if not isinstance(acc_dtype, DType):
            self._error(operation.location, "contract requires acc_dtype")
        elif acc_dtype.category in (DTypeCategory.BOOL, DTypeCategory.INDEX):
            self._error(operation.location, "contract accumulator must be numeric")
        elif result.dtype != acc_dtype:
            self._error(operation.location, "contract result must use acc_dtype")
        if isinstance(acc_dtype, DType):
            accumulator_type = ScalarType(acc_dtype)
            self._verify_callable_schema(
                operation,
                operation.attributes.get("multiply"),
                (ScalarType(lhs.dtype), ScalarType(rhs.dtype)),
                (accumulator_type,),
                "contract multiply",
            )
            self._verify_callable_schema(
                operation,
                operation.attributes.get("combine"),
                (accumulator_type, accumulator_type),
                (accumulator_type,),
                "contract combiner",
            )

    def _verify_ragged(self, operation: Operation) -> None:
        if not self._require_operands(operation, 3) or not self._require_results(operation, 1):
            return
        outer, offsets, indices = operation.operands
        if not isinstance(outer.type, DomainType):
            self._error(operation.location, "ragged outer operand must be a domain")
        if not all(isinstance(value.type, TensorType) for value in (offsets, indices)):
            self._error(operation.location, "ragged offsets and indices must be tensors")
        else:
            assert isinstance(offsets.type, TensorType) and isinstance(indices.type, TensorType)
            if offsets.type.rank != 1 or indices.type.rank != 1:
                self._error(operation.location, "ragged offsets and indices must be rank-one")
            if offsets.type.dtype.category not in (
                DTypeCategory.SIGNED_INTEGER,
                DTypeCategory.UNSIGNED_INTEGER,
                DTypeCategory.INDEX,
            ) or indices.type.dtype.category not in (
                DTypeCategory.SIGNED_INTEGER,
                DTypeCategory.UNSIGNED_INTEGER,
                DTypeCategory.INDEX,
            ):
                self._error(operation.location, "ragged offsets and indices must be integer tensors")
        result = operation.result_types[0]
        if not isinstance(result, RaggedType):
            self._error(operation.location, "ragged result must have RaggedType")
        elif isinstance(outer.type, DomainType):
            if result.outer.flavor is not DomainFlavor.RAGGED_OUTER:
                self._error(operation.location, "ragged outer result domain has wrong flavor")
            if result.member.flavor is not DomainFlavor.RAGGED_MEMBER:
                self._error(operation.location, "ragged member result domain has wrong flavor")
            if result.outer.rank != outer.type.rank:
                self._error(operation.location, "ragged outer rank must match source domain")

    def _verify_ragged_outer(self, operation: Operation) -> None:
        if not self._require_operands(operation, 1) or not self._require_results(operation, 1):
            return
        source = operation.operands[0].type
        if not isinstance(source, RaggedType):
            self._error(operation.location, "ragged_outer expects RaggedType")
        elif operation.result_types[0] != source.outer:
            self._error(operation.location, "ragged_outer result must be descriptor outer domain")

    def _verify_ragged_member(self, operation: Operation) -> None:
        if not self._require_operands(operation, 2) or not self._require_results(operation, 1):
            return
        source = operation.operands[0].type
        if not isinstance(source, RaggedType):
            self._error(operation.location, "ragged_member expects RaggedType")
        else:
            if not is_integer(operation.operands[1].type):
                self._error(operation.location, "ragged member selector must be integer/index")
            if operation.result_types[0] != source.member:
                self._error(operation.location, "ragged_member result must be descriptor member domain")

    def _verify_members(self, operation: Operation) -> None:
        if not self._require_operands(operation, 1) or not self._require_results(operation, 1):
            return
        source = operation.operands[0].type
        if not isinstance(source, (RegionType, DomainType)):
            self._error(operation.location, "members expects ragged member region")
        elif isinstance(source, RegionType) and source.relation != "ragged_member":
            self._error(operation.location, "members expects ragged_member index relation")
        elif isinstance(source, DomainType) and source.flavor is not DomainFlavor.RAGGED_MEMBER:
            self._error(operation.location, "members expects ragged member domain")
        result = operation.result_types[0]
        if not isinstance(result, TensorType) or result.dtype != intent_index:
            self._error(operation.location, "members result must be an index tensor")

    def _verify_gather(self, operation: Operation) -> None:
        if len(operation.operands) < 3 or not self._require_results(operation, 1):
            self._error(operation.location, "gather requires source and index")
            return
        source = operation.operands[0].type
        if not isinstance(source, TensorType):
            self._error(operation.location, "gather source must be a tensor view")
            return
        relation = operation.attributes.get("index")
        if not isinstance(relation, IndexRelation):
            self._error(operation.location, "gather requires explicit index relation")
            indexed_shape = None
        valid_index = operation.attributes.get("valid_operand_index")
        fill_index = operation.attributes.get("fill_operand_index")
        if not self._valid_operand_index(operation, valid_index) or not self._valid_operand_index(
            operation, fill_index
        ):
            self._error(operation.location, "gather requires valid/fill operand positions")
            return
        assert isinstance(valid_index, int) and isinstance(fill_index, int)
        if valid_index == fill_index or valid_index == 0 or fill_index == 0:
            self._error(operation.location, "gather valid/fill operands must be distinct")
            return
        if isinstance(relation, IndexRelation):
            indexed_shape = self._verify_index_relation(
                operation,
                source,
                relation,
                excluded_operand_positions=(valid_index, fill_index),
            )
        valid_type = operation.operands[valid_index].type
        fill_type = operation.operands[fill_index].type
        result = operation.result_types[0]
        if not self._is_boolean_value(valid_type):
            self._error(operation.location, "gather valid operand must be bool")
        if self._value_dtype(fill_type) != source.dtype:
            self._error(operation.location, "gather fill must match source element dtype")
        if self._value_dtype(result) != source.dtype:
            self._error(operation.location, "gather result must match source element dtype")
        if indexed_shape is not None:
            if not self._matches_dtype_shape(result, source.dtype, indexed_shape):
                self._error(operation.location, "gather result must match indexed source shape")
            self._verify_indexed_value_shape(
                operation, valid_type, indexed_shape, "gather valid"
            )
            self._verify_indexed_value_shape(
                operation, fill_type, indexed_shape, "gather fill"
            )

    def _verify_scatter_unique(self, operation: Operation) -> None:
        self._verify_scatter(operation, require_combine=False)

    def _verify_scatter_reduce(self, operation: Operation) -> None:
        self._verify_scatter(operation, require_combine=True)

    def _verify_scatter(self, operation: Operation, require_combine: bool) -> None:
        if len(operation.operands) < 2:
            self._error(operation.location, "scatter requires destination, index, and value")
            return
        self._require_results(operation, 0)
        destination = operation.operands[0].type
        if not isinstance(destination, TensorType):
            self._error(operation.location, "scatter destination must be a tensor view")
            return
        relation = operation.attributes.get("index")
        if not isinstance(relation, IndexRelation):
            self._error(operation.location, "scatter requires explicit index relation")
            indexed_shape = None
        value_index = operation.attributes.get("value_operand_index")
        if not self._valid_operand_index(operation, value_index) or value_index == 0:
            self._error(operation.location, "scatter requires value operand position")
            return
        assert isinstance(value_index, int)
        if isinstance(relation, IndexRelation):
            indexed_shape = self._verify_index_relation(
                operation,
                destination,
                relation,
                excluded_operand_positions=(value_index,),
            )
        if self._value_dtype(operation.operands[value_index].type) != destination.dtype:
            self._error(operation.location, "scatter value must match destination element dtype")
        if indexed_shape is not None:
            self._verify_indexed_value_shape(
                operation,
                operation.operands[value_index].type,
                indexed_shape,
                "scatter",
            )
        if require_combine:
            element_type = ScalarType(destination.dtype)
            self._verify_callable_schema(
                operation,
                operation.attributes.get("combine"),
                (element_type, element_type),
                (element_type,),
                "scatter_reduce combiner",
            )

    def _valid_operand_index(self, operation: Operation, value: object) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < len(operation.operands)
        )

    def _verify_buffer(self, operation: Operation) -> None:
        if not self._require_results(operation, 1):
            return
        if len(operation.operands) not in (0, 1):
            self._error(operation.location, "buffer accepts at most one explicit initializer")
        result = operation.result_types[0]
        if not isinstance(result, BufferType):
            self._error(operation.location, "buffer result must have BufferType")
            return
        if operation.operands:
            initializer = operation.operands[0].type
            initializer_dtype = self._value_dtype(initializer)
            initializer_shape = self._value_shape(initializer)
            if initializer_dtype != result.dtype or initializer_shape not in (
                (),
                tuple(result.shape),
            ):
                self._error(
                    operation.location,
                    "buffer initializer must be scalar or full-shaped with buffer dtype",
                )

    def _verify_buffer_load(self, operation: Operation) -> None:
        if not operation.operands or not isinstance(operation.operands[0].type, BufferType):
            self._error(operation.location, "buffer_load requires logical buffer")
            return
        source = operation.operands[0].type
        relation = operation.attributes.get("index")
        if not isinstance(relation, IndexRelation):
            self._error(operation.location, "buffer_load requires explicit IndexRelation")
            indexed_shape = None
        else:
            indexed_shape = self._verify_index_relation(operation, source, relation)
        if self._require_results(operation, 1):
            result = operation.result_types[0]
            if indexed_shape is not None and not self._matches_dtype_shape(
                result, source.dtype, indexed_shape
            ):
                self._error(
                    operation.location,
                    "buffer_load result must match indexed buffer dtype and shape",
                )

    def _verify_buffer_store(self, operation: Operation) -> None:
        if len(operation.operands) < 2 or not isinstance(operation.operands[0].type, BufferType):
            self._error(operation.location, "buffer_store requires logical buffer and value")
            return
        destination = operation.operands[0].type
        value_index = operation.attributes.get("value_operand_index")
        if not self._valid_operand_index(operation, value_index) or value_index == 0:
            self._error(operation.location, "buffer_store requires value operand position")
            return
        assert isinstance(value_index, int)
        relation = operation.attributes.get("index")
        if not isinstance(relation, IndexRelation):
            self._error(operation.location, "buffer_store requires explicit IndexRelation")
            indexed_shape = None
        else:
            indexed_shape = self._verify_index_relation(
                operation,
                destination,
                relation,
                excluded_operand_positions=(value_index,),
            )
        if self._value_dtype(operation.operands[value_index].type) != destination.dtype:
            self._error(operation.location, "buffer_store value must match buffer dtype")
        if indexed_shape is not None:
            self._verify_indexed_value_shape(
                operation,
                operation.operands[value_index].type,
                indexed_shape,
                "buffer_store",
            )
        self._require_results(operation, 0)

    def _verify_atomic_add(self, operation: Operation) -> None:
        if len(operation.operands) < 2:
            self._error(operation.location, "atomic_add requires target and value")
            return
        self._verify_atomic_common(operation, compare_and_swap=False)
        self._require_results(operation, 0)

    def _verify_atomic_cas(self, operation: Operation) -> None:
        if len(operation.operands) < 3:
            self._error(operation.location, "atomic_cas requires target, compare, and value")
            return
        self._verify_atomic_common(operation, compare_and_swap=True)
        if self._require_results(operation, 1):
            value_index = operation.attributes.get("value_operand_index")
            if self._valid_operand_index(operation, value_index):
                assert isinstance(value_index, int)
                if not types_compatible(
                    operation.result_types[0], operation.operands[value_index].type
                ):
                    self._error(operation.location, "atomic_cas result must match stored value")

    def _verify_atomic_common(
        self,
        operation: Operation,
        *,
        compare_and_swap: bool,
    ) -> None:
        target = operation.operands[0].type
        if not isinstance(target, (TensorType, BufferType)):
            self._error(operation.location, "atomic target must be a view or logical buffer")
            return
        value_index = operation.attributes.get("value_operand_index")
        if not self._valid_operand_index(operation, value_index) or value_index == 0:
            self._error(operation.location, "atomic operation requires value operand position")
            return
        assert isinstance(value_index, int)
        excluded = [value_index]
        if compare_and_swap:
            compare_index = operation.attributes.get("compare_operand_index")
            if (
                not self._valid_operand_index(operation, compare_index)
                or compare_index == 0
                or compare_index == value_index
            ):
                self._error(operation.location, "atomic_cas requires compare operand position")
                return
            assert isinstance(compare_index, int)
            excluded.append(compare_index)
            if not types_compatible(
                operation.operands[compare_index].type,
                operation.operands[value_index].type,
            ):
                self._error(operation.location, "atomic_cas compare/value types must match")
        relation = operation.attributes.get("index")
        if not isinstance(relation, IndexRelation):
            self._error(operation.location, "atomic operation requires explicit IndexRelation")
            indexed_shape = None
        else:
            indexed_shape = self._verify_index_relation(
                operation,
                target,
                relation,
                excluded_operand_positions=tuple(excluded),
            )
        if self._value_dtype(operation.operands[value_index].type) != target.dtype:
            self._error(operation.location, "atomic value must match target element dtype")
        if indexed_shape is not None:
            self._verify_indexed_value_shape(
                operation,
                operation.operands[value_index].type,
                indexed_shape,
                "atomic",
            )
            if compare_and_swap:
                self._verify_indexed_value_shape(
                    operation,
                    operation.operands[compare_index].type,
                    indexed_shape,
                    "atomic_cas compare",
                )
        if not isinstance(operation.attributes.get("ordering"), AtomicOrdering):
            self._error(operation.location, "atomic operation requires AtomicOrdering")
        if not isinstance(operation.attributes.get("scope"), MemoryScope):
            self._error(operation.location, "atomic operation requires MemoryScope")

    def _verify_fence(self, operation: Operation) -> None:
        self._require_operands(operation, 0)
        self._require_results(operation, 0)
        if not isinstance(operation.attributes.get("ordering"), AtomicOrdering):
            self._error(operation.location, "fence requires AtomicOrdering")
        if not isinstance(operation.attributes.get("scope"), MemoryScope):
            self._error(operation.location, "fence requires MemoryScope")

    def _verify_random(self, operation: Operation) -> None:
        if not self._require_operands(operation, 2) or not self._require_results(operation, 1):
            return
        if not is_integer(operation.operands[0].type):
            self._error(operation.location, "random seed must be integer")
        index_type = operation.operands[1].type
        if not (
            isinstance(index_type, LogicalIndexType)
            or (
                isinstance(index_type, TensorType)
                and index_type.dtype == intent_index
            )
        ):
            self._error(operation.location, "random identity must be a source logical index")
        result = operation.result_types[0]
        if not self._is_numeric_value(result):
            self._error(operation.location, "random result must be numeric scalar/tensor")

    def _verify_call(self, operation: Operation) -> None:
        callee_name = operation.attributes.get("callee")
        callee = self._symbols.get(callee_name) if isinstance(callee_name, str) else None
        if callee is None:
            self._error(operation.location, "call references unknown helper")
            return
        if callee.kind is not FunctionKind.HELPER:
            self._error(operation.location, "kernel entries cannot be called from Kernel IR")
        if len(operation.operands) != len(callee.parameters):
            self._error(operation.location, "call operand count does not match helper")
        else:
            for index, (operand, parameter) in enumerate(
                zip(operation.operands, callee.parameters)
            ):
                if not types_compatible(operand.type, parameter.spec.type):
                    self._error(
                        operation.location,
                        f"call operand {index} type does not match helper parameter",
                    )
        if len(operation.result_types) != len(callee.result_types) or any(
            not types_compatible(actual, expected)
            for actual, expected in zip(operation.result_types, callee.result_types)
        ):
            self._error(operation.location, "call result types do not match helper")
        expected_effects: set[tuple[EffectKind, ResourceKind, Value | None]] = set()
        parameter_values = [parameter.value for parameter in callee.parameters]
        for nested in self._walk_region(callee.body):
            for effect in nested.effects:
                target: Value | None = None
                if effect.target is not None:
                    try:
                        parameter_index = parameter_values.index(effect.target)
                    except ValueError:
                        if effect.resource is not ResourceKind.LOGICAL_BUFFER:
                            self._error(
                                nested.location,
                                "helper external effect must target a helper parameter",
                            )
                        continue
                    if parameter_index < len(operation.operands):
                        target = operation.operands[parameter_index]
                expected_effects.add((effect.kind, effect.resource, target))
        actual_effects = {
            (effect.kind, effect.resource, effect.target) for effect in operation.effects
        }
        if actual_effects != expected_effects or len(actual_effects) != len(
            operation.effects
        ):
            self._error(operation.location, "call effect summary does not match helper body")

    def _normalize_axes(
        self,
        value: object,
        rank: int,
        operation: Operation,
    ) -> set[int] | None:
        axes = (value,) if isinstance(value, int) and not isinstance(value, bool) else value
        if not isinstance(axes, tuple) or not axes:
            self._error(operation.location, "operation requires one or more positional axes")
            return None
        normalized: set[int] = set()
        for axis in axes:
            if not isinstance(axis, int) or isinstance(axis, bool) or not -rank <= axis < rank:
                self._error(operation.location, "axis is outside tensor rank")
                return None
            normalized.add(axis % rank)
        if len(normalized) != len(axes):
            self._error(operation.location, "axes must be unique")
            return None
        return normalized

    def _verify_combiner_purity(self, module: Module) -> None:
        effectful_functions = {
            function.name
            for function in module.functions
            if self._function_has_effects(function)
        }
        for function in module.functions:
            for operation in self._walk_region(function.body):
                if operation.opcode not in (OpCode.REDUCE, OpCode.SCAN, OpCode.SCATTER_REDUCE):
                    continue
                combine = operation.attributes.get("combine")
                if isinstance(combine, str) and combine in effectful_functions:
                    self._error(
                        operation.location,
                        "structured combiner helper must be pure",
                    )

    def _function_has_effects(self, function: Function) -> bool:
        return any(operation.effects for operation in self._walk_region(function.body))

    def _walk_region(self, region: Region) -> Iterable[Operation]:
        for block in region.blocks:
            for operation in block.operations:
                yield operation
                for nested in operation.regions:
                    yield from self._walk_region(nested)


def verify(module: Module) -> None:
    Verifier().verify(module)
