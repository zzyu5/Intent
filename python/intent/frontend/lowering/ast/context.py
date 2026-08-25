from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from intent.api import Definition
from intent.api import DefinitionKind
from intent.frontend.mlir import BlockState
from intent.frontend.semantics import ConstexprType
from intent.frontend.semantics import BufferType
from intent.frontend.semantics import DynamicDim
from intent.frontend.semantics import Effect
from intent.frontend.mlir import FunctionState
from intent.frontend.semantics import ValueType
from intent.frontend.diagnostics import Location
from intent.frontend.semantics import LogicalIndexType
from intent.frontend.semantics import IndexRelation
from intent.frontend.semantics import IndexTerm
from intent.frontend.semantics import IndexTermKind
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import ShapeExpr
from intent.frontend.semantics import ShapeExprKind
from intent.frontend.semantics import ShapeRelation
from intent.frontend.mlir import EmittedOperation
from intent.frontend.mlir import RegionState
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import StaticDim
from intent.frontend.semantics import TensorType
from intent.frontend.semantics import TupleType
from intent.frontend.mlir import MlirValue
from intent.frontend.semantics import broadcast_shape
from intent.frontend.semantics import dims_compatible
from intent.language import ViewKind
from intent.language import DType
from intent.language import DTypeCategory
from intent.language import bool as intent_bool
from intent.language import f64
from intent.language import i64
from intent.language import index as intent_index
from intent.frontend.semantics import EffectKind
from intent.frontend.semantics import ResourceKind
from intent.frontend.semantics import RegionType

from ...diagnostics.errors import FrontendError
from .model import ConstexprBinding
from .model import Expression
from .model import Literal
from .model import LoopContext
from .model import ShapeDimension
from .model import ShapeValue
from .model import StaticTuple
from ...source.unit import SourceUnit

if TYPE_CHECKING:
    from ...compilation.compiler import FrontendCompiler


@dataclass(slots=True)
class InlineHelperFrame:
    entry_block: BlockState
    returned: tuple[MlirValue, ...] | None = None


class FunctionLowerer:
    def __init__(
        self,
        *,
        compiler: FrontendCompiler,
        definition: Definition[object, object],
        source: SourceUnit,
        function: FunctionState,
        constexpr_values: dict[str, object],
    ) -> None:
        self.compiler = compiler
        self.definition = definition
        self.source = source
        self.function = function
        self.environment: dict[str, Expression] = {}
        self.current_block = function.body.blocks[0]
        self.loop_stack: list[LoopContext] = []
        self.view_kinds: dict[MlirValue, ViewKind] = {}
        self.inline_helpers: list[InlineHelperFrame] = []
        self.ragged_mappings: dict[MlirValue, MlirValue] = {}
        self.iteration_shapes: dict[MlirValue, tuple[object, ...]] = {}
        self.dimension_values: dict[MlirValue, object] = {}
        self.dimension_origins: dict[object, list[ShapeDimension]] = {}
        self.value_blocks: dict[MlirValue, BlockState] = {}
        self.operation_blocks: dict[int, BlockState] = {}
        self.region_parent_blocks: dict[RegionState, BlockState] = {}
        self._dynamic_dimension_counter = 0
        self._initialize_parameters(constexpr_values)

    def _initialize_parameters(self, constexpr_values: dict[str, object]) -> None:
        for parameter in self.function.parameters:
            name = parameter.spec.name
            value = parameter.value
            if isinstance(parameter.spec.type, ConstexprType):
                self.environment[name] = ConstexprBinding(constexpr_values[name], value)
            else:
                self.environment[name] = value
            if parameter.spec.view_kind is not None:
                self.view_kinds[value] = parameter.spec.view_kind
            self.register_value_shape(value)

    def lower(self) -> None:
        self.lower_statements(self.source.function.body)
        if not self.is_terminated(self.current_block):
            self.emit(OperationKind.RETURN, self.source.location(self.source.function))

    def lower_inline_helper(
        self,
        *,
        definition: Definition[object, object],
        source: SourceUnit,
        parameters: tuple[object, ...],
        arguments: tuple[object, ...],
    ) -> tuple[MlirValue, ...]:
        if len(parameters) != len(arguments):
            self.error(source.function, "helper argument count does not match call")
        saved_definition = self.definition
        saved_source = self.source
        saved_environment = self.environment
        saved_loop_stack = self.loop_stack
        entry_block = self.current_block
        frame = InlineHelperFrame(entry_block)
        self.definition = definition
        self.source = source
        self.environment = {
            parameter.name: argument
            for parameter, argument in zip(parameters, arguments)
        }
        self.loop_stack = []
        self.inline_helpers.append(frame)
        self.lower_statements(source.function.body)
        if frame.returned is None:
            self.error(source.function, "@intent.fn must end with an explicit return")
        results = frame.returned
        self.inline_helpers.pop()
        self.definition = saved_definition
        self.source = saved_source
        self.environment = saved_environment
        self.loop_stack = saved_loop_stack
        self.current_block = entry_block
        return results

    def lower_statements(self, statements: list[ast.stmt]) -> None:
        from .statements import lower_statement

        for statement in statements:
            if self.inline_helpers and self.inline_helpers[-1].returned is not None:
                self.error(statement, "statement is unreachable after helper return")
            if self.is_terminated(self.current_block):
                self.error(statement, "statement is unreachable after a terminator")
            lower_statement(self, statement)

    def lower_expression(self, node: ast.AST) -> Expression:
        from .expressions import lower_expression

        return lower_expression(self, node)

    def emit(
        self,
        opcode: OperationKind,
        location: Location,
        *,
        operands: tuple[MlirValue, ...] = (),
        result_types: tuple[ValueType, ...] = (),
        attributes: dict[str, object] | None = None,
        regions: tuple[RegionState, ...] = (),
        effects: tuple[Effect, ...] = (),
        result_names: tuple[str | None, ...] = (),
    ) -> EmittedOperation:
        operation = self.compiler.builder.emit(
            self.current_block,
            opcode,
            location,
            operands=operands,
            result_types=result_types,
            attributes=attributes,
            regions=regions,
            effects=effects,
            result_names=result_names,
        )
        self.operation_blocks[operation.id] = self.current_block
        for result in operation.results:
            self.value_blocks[result] = self.current_block
            self.register_value_shape(result)
        return operation

    def make_region(
        self,
        location: Location,
        argument_types: tuple[ValueType, ...] = (),
        argument_names: tuple[str | None, ...] = (),
    ) -> RegionState:
        region = self.compiler.builder.region(
            location, argument_types, argument_names
        )
        self.region_parent_blocks[region] = self.current_block
        for argument in region.blocks[0].arguments:
            self.register_value_shape(argument, region.blocks[0])
        return region

    def materialize(
        self,
        expression: Expression,
        node: ast.AST,
        expected_type: ValueType | None = None,
    ) -> MlirValue:
        if isinstance(expression, MlirValue):
            if expected_type is not None and not self.types_compatible_for_literal(
                expression.type, expected_type
            ):
                self.error(node, f"value type {expression.type} does not match {expected_type}")
            return expression
        if isinstance(expression, ConstexprBinding):
            constexpr_type = expression.ir_value.type
            return self.emit_literal(
                expression.python_value,
                node,
                expected_type
                or (
                    constexpr_type.value_type
                    if isinstance(constexpr_type, ConstexprType)
                    else constexpr_type
                ),
            )
        if isinstance(expression, ShapeDimension):
            return self.materialize_dimension(expression, node)
        if isinstance(expression, Literal):
            return self.emit_literal(expression.value, node, expected_type)
        if isinstance(expression, StaticTuple):
            expected_components: tuple[ValueType, ...] | None = None
            if expected_type is not None:
                if not isinstance(expected_type, TupleType):
                    self.error(node, f"tuple value does not match {expected_type}")
                if len(expression.elements) != len(expected_type.components):
                    self.error(node, "tuple value/expected type arity mismatch")
                expected_components = expected_type.components
            components = tuple(
                self.materialize(
                    element,
                    node,
                    expected_components[index] if expected_components is not None else None,
                )
                for index, element in enumerate(expression.elements)
            )
            result_type = TupleType(tuple(component.type for component in components))
            return self.emit(
                OperationKind.MAKE_TUPLE,
                self.location(node),
                operands=components,
                result_types=(result_type,),
            ).results[0]
        self.error(node, "expression is compile-time metadata, not an SSA value")

    def emit_literal(
        self,
        value: bool | int | float,
        node: ast.AST,
        expected_type: ValueType | None = None,
    ) -> MlirValue:
        if expected_type is None:
            if isinstance(value, bool):
                expected_type = ScalarType(intent_bool)
            elif isinstance(value, int):
                expected_type = ScalarType(i64)
            else:
                expected_type = ScalarType(f64)
        if isinstance(expected_type, ConstexprType):
            expected_type = expected_type.value_type
        if not isinstance(expected_type, ScalarType):
            self.error(node, "literal context must require a scalar type")
        category = expected_type.dtype.category
        if category is DTypeCategory.BOOL and not isinstance(value, bool):
            self.error(node, "bool literal context requires a Python bool")
        if category in (
            DTypeCategory.SIGNED_INTEGER,
            DTypeCategory.UNSIGNED_INTEGER,
            DTypeCategory.INDEX,
        ) and (isinstance(value, bool) or not isinstance(value, int)):
            self.error(node, "integer literal context requires a Python int")
        if category in (DTypeCategory.FLOAT, DTypeCategory.BFLOAT) and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            self.error(node, "floating literal context requires int/float")
        if category in (DTypeCategory.FLOAT, DTypeCategory.BFLOAT) and isinstance(
            value, int
        ):
            value = float(value)
        operation = self.emit(
            OperationKind.CONSTANT,
            self.location(node),
            result_types=(expected_type,),
            attributes={"value": value},
        )
        return operation.results[0]

    def materialize_dimension(self, dimension: ShapeDimension, node: ast.AST) -> MlirValue:
        if isinstance(dimension.dimension, StaticDim):
            return self.emit_literal(
                dimension.dimension.value, node, ScalarType(intent_index)
            )
        operation = self.emit(
            OperationKind.DIM,
            self.location(node),
            operands=(dimension.source,),
            result_types=(ScalarType(intent_index),),
            attributes={
                "axis": dimension.axis,
                "dimension": self.compiler.builder.dimension_id(dimension.dimension),
            },
        )
        result = operation.results[0]
        self.dimension_values[result] = dimension.dimension
        self._remember_dimension_origin(dimension)
        return result

    def materialize_shape_extent(self, dimension: object, node: ast.AST) -> MlirValue:
        candidates = self.dimension_origins.get(dimension, [])
        origin = next(
            (
                candidate
                for candidate in candidates
                if self._block_dominates(
                    self.value_blocks.get(candidate.source), self.current_block
                )
            ),
            None,
        )
        if origin is None:
            self.error(node, f"dynamic shape extent {dimension} has no SSA source")
        return self.materialize_dimension(origin, node)

    def read_value(self, expression: Expression, node: ast.AST) -> MlirValue:
        value = self.materialize(expression, node)
        if value not in self.view_kinds:
            return value
        self.require_readable_view(value, node)
        if not isinstance(value.type, TensorType):
            self.error(node, "view parameter must have tensor type")
        relation = IndexRelation(
            len(value.type.shape),
            len(value.type.shape),
            tuple(
                self.compiler.builder.dimension_id(dimension)
                for dimension in value.type.shape
            ),
            tuple(IndexTerm(IndexTermKind.FULL_SLICE) for _ in value.type.shape)
        )
        operation = self.emit(
            OperationKind.VIEW_LOAD,
            self.location(node),
            operands=(value,),
            result_types=(value.type,),
            attributes={"index": relation},
            effects=(Effect(EffectKind.READ, ResourceKind.EXTERNAL_VIEW, value),),
        )
        return operation.results[0]

    def require_readable_view(self, value: MlirValue, node: ast.AST) -> None:
        kind = self.view_kinds.get(value)
        if kind not in (ViewKind.IN, ViewKind.INOUT):
            self.error(node, "Out-only view cannot be read")

    def require_writable_view(self, value: MlirValue, node: ast.AST) -> None:
        kind = self.view_kinds.get(value)
        if kind not in (ViewKind.OUT, ViewKind.INOUT):
            self.error(node, "view write requires Out or InOut ABI")

    def require_atomic_view(self, value: MlirValue, node: ast.AST) -> None:
        if self.view_kinds.get(value) is not ViewKind.INOUT:
            self.error(node, "external atomic target requires InOut ABI")

    def shape_value(self, value: MlirValue, node: ast.AST) -> ShapeValue:
        if not isinstance(value.type, TensorType):
            self.error(node, ".shape is only valid on a tensor/view")
        return ShapeValue(
            tuple(
                ShapeDimension(dimension, value, axis)
                for axis, dimension in enumerate(value.type.shape)
            )
        )

    def value_result_type(
        self,
        dtype: DType,
        shape: tuple[object, ...],
    ) -> ValueType:
        return TensorType(dtype, shape) if shape else ScalarType(dtype)

    def broadcast_result_type(self, lhs: ValueType, rhs: ValueType, node: ast.AST) -> ValueType:
        lhs_dtype, lhs_shape = self.dtype_and_shape(lhs, node)
        rhs_dtype, rhs_shape = self.dtype_and_shape(rhs, node)
        if lhs_dtype != rhs_dtype:
            self.error(node, "mixed dtypes require an explicit I.cast")
        try:
            shape = broadcast_shape(lhs_shape, rhs_shape)
        except ValueError as error:
            self.error(node, str(error))
        return self.value_result_type(lhs_dtype, shape)

    def broadcast_value(
        self,
        value: MlirValue,
        result_shape: tuple[object, ...],
        node: ast.AST,
    ) -> MlirValue:
        dtype, source_shape = self.dtype_and_shape(value.type, node)
        target_shape = tuple(result_shape)
        if tuple(source_shape) == target_shape:
            return value
        if not target_shape:
            self.error(node, "tensor value cannot broadcast to a scalar")
        try:
            broadcasted = tuple(broadcast_shape(source_shape, target_shape))
        except ValueError as error:
            self.error(node, str(error))
        if len(broadcasted) != len(target_shape) or not all(
            dims_compatible(source, destination)
            for source, destination in zip(broadcasted, target_shape)
        ):
            self.error(node, "value cannot broadcast to the required result shape")
        shape_operands: list[MlirValue] = []
        shape_relation: list[ShapeExpr] = []
        for dimension in target_shape:
            if isinstance(dimension, StaticDim):
                shape_relation.append(
                    ShapeExpr(ShapeExprKind.STATIC, 0, dimension.value)
                )
            else:
                shape_relation.append(
                    ShapeExpr(
                        ShapeExprKind.SSA_EXTENT,
                        self.compiler.builder.dimension_id(dimension),
                        1 + len(shape_operands),
                    )
                )
                shape_operands.append(self.materialize_shape_extent(dimension, node))
        operation = self.emit(
            OperationKind.BROADCAST,
            self.location(node),
            operands=(value, *shape_operands),
            result_types=(TensorType(dtype, target_shape),),
            attributes={"shape": ShapeRelation(tuple(shape_relation))},
        )
        return operation.results[0]

    def dtype_and_shape(
        self,
        value_type: ValueType,
        node: ast.AST,
    ) -> tuple[DType, tuple[object, ...]]:
        if isinstance(value_type, ScalarType):
            return value_type.dtype, ()
        if isinstance(value_type, LogicalIndexType):
            return intent_index, ()
        if isinstance(value_type, TensorType):
            return value_type.dtype, tuple(value_type.shape)
        self.error(node, f"expected scalar/tensor value, got {value_type}")

    def coerce_pair(
        self,
        lhs: Expression,
        rhs: Expression,
        node: ast.AST,
    ) -> tuple[MlirValue, MlirValue]:
        from .expressions import compile_time_value

        lhs_known, lhs_static = compile_time_value(lhs)
        rhs_known, rhs_static = compile_time_value(rhs)
        if lhs_known and not rhs_known:
            lhs = Literal(lhs_static)
        if rhs_known and not lhs_known:
            rhs = Literal(rhs_static)
        if isinstance(lhs, Literal) and not isinstance(rhs, Literal):
            rhs_value = self.materialize(rhs, node)
            dtype, _ = self.dtype_and_shape(rhs_value.type, node)
            return self.materialize(lhs, node, ScalarType(dtype)), rhs_value
        if isinstance(rhs, Literal) and not isinstance(lhs, Literal):
            lhs_value = self.materialize(lhs, node)
            dtype, _ = self.dtype_and_shape(lhs_value.type, node)
            return lhs_value, self.materialize(rhs, node, ScalarType(dtype))
        return self.materialize(lhs, node), self.materialize(rhs, node)

    def types_compatible_for_literal(self, actual: ValueType, expected: ValueType) -> bool:
        from intent.frontend.semantics.types import types_compatible

        return types_compatible(actual, expected)

    def is_terminated(self, block: BlockState) -> bool:
        from intent.frontend.semantics import TERMINATORS

        return block.last_operation in TERMINATORS

    def location(self, node: ast.AST) -> Location:
        return self.source.location(node)

    def error(self, node: ast.AST, message: str) -> None:
        raise FrontendError(message, self.location(node))

    def dynamic_shape_for_region(self, value: MlirValue) -> tuple[DynamicDim, ...]:
        known = self.iteration_shapes.get(value)
        if known is not None:
            dimensions = tuple(known)
        else:
            rank = getattr(value.type, "rank", 1)
            dimensions = tuple(
                DynamicDim(f"region_{value.id}_{axis}") for axis in range(rank)
            )
        for axis, dimension in enumerate(dimensions):
            self._remember_dimension_origin(ShapeDimension(dimension, value, axis))
        return dimensions

    def fresh_dynamic_dimension(self, role: str) -> DynamicDim:
        identity = self._dynamic_dimension_counter
        self._dynamic_dimension_counter += 1
        return DynamicDim(f"{role}_{identity}")

    def logical_index_type(self, source: MlirValue, axis: int) -> LogicalIndexType:
        source_id = (
            source.type.source_id
            if isinstance(source.type, RegionType)
            else getattr(source.type, "origin_id", None)
        )
        if source_id is None:
            raise TypeError("logical index source requires stable domain provenance")
        return LogicalIndexType(source_id, axis)

    def register_value_shape(
        self, value: MlirValue, block: BlockState | None = None
    ) -> None:
        self.value_blocks.setdefault(value, block or self.current_block)
        value_type = value.type
        if isinstance(value_type, TensorType):
            shape = value_type.shape
        elif isinstance(value_type, BufferType):
            shape = value_type.shape
        else:
            return
        for axis, dimension in enumerate(shape):
            if not isinstance(dimension, StaticDim):
                self._remember_dimension_origin(
                    ShapeDimension(dimension, value, axis)
                )

    def _remember_dimension_origin(self, origin: ShapeDimension) -> None:
        candidates = self.dimension_origins.setdefault(origin.dimension, [])
        if all(candidate.source is not origin.source or candidate.axis != origin.axis
               for candidate in candidates):
            candidates.append(origin)

    def _block_dominates(
        self, candidate: BlockState | None, current: BlockState
    ) -> bool:
        block: BlockState | None = current
        while block is not None:
            if block is candidate:
                return True
            region = block.owner
            owner = region.owner_operation if region is not None else None
            block = (
                self.operation_blocks.get(owner)
                if owner is not None
                else self.region_parent_blocks.get(region)
                if region is not None
                else None
            )
        return False
