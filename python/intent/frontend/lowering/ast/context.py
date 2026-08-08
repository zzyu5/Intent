from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from intent.api import Definition
from intent.api import DefinitionKind
from intent.ir import Block
from intent.ir import ConstexprType
from intent.ir import DynamicDim
from intent.ir import Effect
from intent.ir import Function
from intent.ir import IRType
from intent.ir import Location
from intent.ir import LogicalIndexType
from intent.ir import IndexRelation
from intent.ir import IndexTerm
from intent.ir import IndexTermKind
from intent.ir import OpCode
from intent.ir import Operation
from intent.ir import Region
from intent.ir import ScalarType
from intent.ir import TensorType
from intent.ir import Value
from intent.ir import broadcast_shape
from intent.language import ViewKind
from intent.language import DType
from intent.language import DTypeCategory
from intent.language import bool as intent_bool
from intent.language import f64
from intent.language import i64
from intent.language import index as intent_index
from intent.ir import EffectKind
from intent.ir import ResourceKind

from ...diagnostics.errors import FrontendError
from .model import ConstexprBinding
from .model import Expression
from .model import Literal
from .model import LoopContext
from .model import ShapeDimension
from .model import ShapeValue
from ...source.unit import SourceUnit

if TYPE_CHECKING:
    from ...compilation.compiler import FrontendCompiler


class FunctionLowerer:
    def __init__(
        self,
        *,
        compiler: FrontendCompiler,
        definition: Definition[object, object],
        source: SourceUnit,
        function: Function,
        constexpr_values: dict[str, object],
    ) -> None:
        self.compiler = compiler
        self.definition = definition
        self.source = source
        self.function = function
        self.environment: dict[str, Expression] = {}
        self.current_block = function.body.blocks[0]
        self.loop_stack: list[LoopContext] = []
        self.view_kinds: dict[Value, ViewKind] = {}
        self.return_types: tuple[IRType, ...] | None = None
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

    def lower(self) -> None:
        self.lower_statements(self.source.function.body)
        if not self.is_terminated(self.current_block):
            if self.definition.kind is DefinitionKind.KERNEL:
                self.emit(OpCode.RETURN, self.source.location(self.source.function))
            else:
                self.error(self.source.function, "@intent.fn must end with an explicit return")
        if self.definition.kind is DefinitionKind.HELPER:
            if self.return_types is None:
                self.error(self.source.function, "@intent.fn did not produce a return schema")
            self.function.result_types = self.return_types

    def lower_statements(self, statements: list[ast.stmt]) -> None:
        from .statements import lower_statement

        for statement in statements:
            if self.is_terminated(self.current_block):
                self.error(statement, "statement is unreachable after a terminator")
            lower_statement(self, statement)

    def lower_expression(self, node: ast.AST) -> Expression:
        from .expressions import lower_expression

        return lower_expression(self, node)

    def emit(
        self,
        opcode: OpCode,
        location: Location,
        *,
        operands: tuple[Value, ...] = (),
        result_types: tuple[IRType, ...] = (),
        attributes: dict[str, object] | None = None,
        regions: tuple[Region, ...] = (),
        effects: tuple[Effect, ...] = (),
        result_names: tuple[str | None, ...] = (),
    ) -> Operation:
        return self.compiler.builder.emit(
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

    def materialize(
        self,
        expression: Expression,
        node: ast.AST,
        expected_type: IRType | None = None,
    ) -> Value:
        if isinstance(expression, Value):
            if expected_type is not None and not self.types_compatible_for_literal(
                expression.type, expected_type
            ):
                self.error(node, f"value type {expression.type} does not match {expected_type}")
            return expression
        if isinstance(expression, ConstexprBinding):
            return expression.ir_value
        if isinstance(expression, ShapeDimension):
            return self.materialize_dimension(expression, node)
        if isinstance(expression, Literal):
            return self.emit_literal(expression.value, node, expected_type)
        self.error(node, "expression is compile-time metadata, not an SSA value")

    def emit_literal(
        self,
        value: bool | int | float,
        node: ast.AST,
        expected_type: IRType | None = None,
    ) -> Value:
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
        operation = self.emit(
            OpCode.CONSTANT,
            self.location(node),
            result_types=(expected_type,),
            attributes={"value": value},
        )
        return operation.results[0]

    def materialize_dimension(self, dimension: ShapeDimension, node: ast.AST) -> Value:
        operation = self.emit(
            OpCode.DIM,
            self.location(node),
            operands=(dimension.source,),
            result_types=(ScalarType(intent_index),),
            attributes={"axis": dimension.axis},
        )
        return operation.results[0]

    def read_value(self, expression: Expression, node: ast.AST) -> Value:
        value = self.materialize(expression, node)
        if value not in self.view_kinds:
            return value
        self.require_readable_view(value, node)
        if not isinstance(value.type, TensorType):
            self.error(node, "view parameter must have tensor type")
        relation = IndexRelation(
            tuple(IndexTerm(IndexTermKind.FULL_SLICE) for _ in value.type.shape)
        )
        operation = self.emit(
            OpCode.VIEW_LOAD,
            self.location(node),
            operands=(value,),
            result_types=(value.type,),
            attributes={"index": relation},
            effects=(Effect(EffectKind.READ, ResourceKind.EXTERNAL_VIEW, value),),
        )
        return operation.results[0]

    def require_readable_view(self, value: Value, node: ast.AST) -> None:
        kind = self.view_kinds.get(value)
        if kind not in (ViewKind.IN, ViewKind.INOUT):
            self.error(node, "Out-only view cannot be read")

    def require_writable_view(self, value: Value, node: ast.AST) -> None:
        kind = self.view_kinds.get(value)
        if kind not in (ViewKind.OUT, ViewKind.INOUT):
            self.error(node, "view write requires Out or InOut ABI")

    def require_atomic_view(self, value: Value, node: ast.AST) -> None:
        if self.view_kinds.get(value) is not ViewKind.INOUT:
            self.error(node, "external atomic target requires InOut ABI")

    def shape_value(self, value: Value, node: ast.AST) -> ShapeValue:
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
    ) -> IRType:
        return TensorType(dtype, shape) if shape else ScalarType(dtype)

    def broadcast_result_type(self, lhs: IRType, rhs: IRType, node: ast.AST) -> IRType:
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
        value: Value,
        result_shape: tuple[object, ...],
        node: ast.AST,
    ) -> Value:
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
        if broadcasted != target_shape:
            self.error(node, "value cannot broadcast to the required result shape")
        operation = self.emit(
            OpCode.BROADCAST,
            self.location(node),
            operands=(value,),
            result_types=(TensorType(dtype, target_shape),),
        )
        return operation.results[0]

    def dtype_and_shape(
        self,
        value_type: IRType,
        node: ast.AST,
    ) -> tuple[DType, tuple[object, ...]]:
        if isinstance(value_type, ScalarType):
            return value_type.dtype, ()
        if isinstance(value_type, TensorType):
            return value_type.dtype, tuple(value_type.shape)
        self.error(node, f"expected scalar/tensor value, got {value_type}")

    def coerce_pair(
        self,
        lhs: Expression,
        rhs: Expression,
        node: ast.AST,
    ) -> tuple[Value, Value]:
        if isinstance(lhs, Literal) and not isinstance(rhs, Literal):
            rhs_value = self.materialize(rhs, node)
            dtype, _ = self.dtype_and_shape(rhs_value.type, node)
            return self.materialize(lhs, node, ScalarType(dtype)), rhs_value
        if isinstance(rhs, Literal) and not isinstance(lhs, Literal):
            lhs_value = self.materialize(lhs, node)
            dtype, _ = self.dtype_and_shape(lhs_value.type, node)
            return lhs_value, self.materialize(rhs, node, ScalarType(dtype))
        return self.materialize(lhs, node), self.materialize(rhs, node)

    def types_compatible_for_literal(self, actual: IRType, expected: IRType) -> bool:
        from intent.ir.types import types_compatible

        return types_compatible(actual, expected)

    def is_terminated(self, block: Block) -> bool:
        from intent.ir.ops import TERMINATORS

        return bool(block.operations and block.operations[-1].opcode in TERMINATORS)

    def location(self, node: ast.AST) -> Location:
        return self.source.location(node)

    def error(self, node: ast.AST, message: str) -> None:
        raise FrontendError(message, self.location(node))

    def dynamic_shape_for_region(self, value: Value) -> tuple[DynamicDim, ...]:
        rank = getattr(value.type, "rank", 1)
        return tuple(DynamicDim(f"region_{value.id}_{axis}") for axis in range(rank))

    def logical_index_type(self, relation: str) -> LogicalIndexType:
        return LogicalIndexType(relation)

    def helper_call_effects(
        self,
        helper: Function,
        arguments: tuple[Value, ...],
    ) -> tuple[Effect, ...]:
        parameter_values = tuple(parameter.value for parameter in helper.parameters)
        effects: list[Effect] = []
        seen: set[tuple[object, object, Value | None]] = set()

        def walk(region: Region) -> None:
            for block in region.blocks:
                for operation in block.operations:
                    for effect in operation.effects:
                        target: Value | None = None
                        if effect.target is not None:
                            if effect.target not in parameter_values:
                                continue
                            target = arguments[parameter_values.index(effect.target)]
                        key = (effect.kind, effect.resource, target)
                        if key not in seen:
                            effects.append(Effect(effect.kind, effect.resource, target))
                            seen.add(key)
                    for nested in operation.regions:
                        walk(nested)

        walk(helper.body)
        return tuple(effects)
