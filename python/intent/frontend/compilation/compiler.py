from __future__ import annotations

from dataclasses import dataclass

from intent.api import HelperDefinition
from intent.api import KernelDefinition
from intent.frontend.mlir import FunctionKind
from intent.frontend.mlir import MlirBuilder
from intent.frontend.mlir import MlirValue
from intent.frontend.mlir import ParameterKind
from intent.frontend.mlir import ParameterSpec
from intent.frontend.mlir import canonicalize_mlir
from intent.frontend.mlir.attributes import SymbolRef
from intent.frontend.semantics import BinaryOperator
from intent.frontend.semantics import ComparePredicate
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import RecordType
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import ValueType
from intent.language import bool as intent_bool
from intent.language import i32

from ..diagnostics.errors import FrontendError
from ..lowering.ast.context import FunctionLowerer
from ..source.signature import lower_helper_parameters
from ..source.signature import lower_kernel_signature
from ..source.unit import SourceUnit


@dataclass(frozen=True, slots=True)
class HelperKey:
    definition: HelperDefinition[object, object]
    argument_types: tuple[ValueType, ...]


class FrontendCompiler:
    def __init__(
        self,
        definition: KernelDefinition[object, object],
        constexprs: dict[str, object],
    ) -> None:
        self.definition = definition
        self.source = SourceUnit.from_definition(definition)
        self.signature = lower_kernel_signature(definition, self.source, constexprs)
        self.builder = MlirBuilder(definition.__name__, self.source.location(self.source.function))
        self.active_helpers: set[HelperKey] = set()
        self.combiner_helpers: dict[HelperKey, SymbolRef] = {}
        self.argmax_helpers: dict[tuple[ValueType, ValueType], SymbolRef] = {}

    def lower(self) -> str:
        function = self.builder.function(
            self.definition.__name__,
            FunctionKind.KERNEL,
            self.signature.parameters,
            (),
            self.source.location(self.source.function),
        )
        lowerer = FunctionLowerer(
            compiler=self,
            definition=self.definition,
            source=self.source,
            function=function,
            constexpr_values=self.signature.constexpr_values,
        )
        lowerer.lower()
        return canonicalize_mlir(self.builder.emit_module())

    def lower_helper_inline(
        self,
        caller: FunctionLowerer,
        definition: HelperDefinition[object, object],
        arguments: tuple[MlirValue, ...],
        call_location: object,
    ) -> tuple[MlirValue, ...]:
        argument_types = tuple(argument.type for argument in arguments)
        key = HelperKey(definition, argument_types)
        if key in self.active_helpers:
            raise FrontendError(
                "recursive @intent.fn requires an explicit recursive IR contract and is unsupported",
                call_location,
            )
        source = SourceUnit.from_definition(definition)
        parameters = lower_helper_parameters(source, argument_types)
        self.active_helpers.add(key)
        results = caller.lower_inline_helper(
            definition=definition,
            source=source,
            parameters=parameters,
            arguments=arguments,
        )
        self.active_helpers.remove(key)
        return results

    def lower_combiner(
        self,
        caller: FunctionLowerer,
        definition: HelperDefinition[object, object],
        accumulator_type: ValueType,
        component_names: tuple[str, ...],
        component_types: tuple[ValueType, ...],
        captures: tuple[MlirValue, ...],
        call_location: object,
    ) -> SymbolRef:
        logical_argument_types = (accumulator_type, accumulator_type) + tuple(
            value.type for value in captures
        )
        key = HelperKey(definition, logical_argument_types)
        cached = self.combiner_helpers.get(key)
        if cached is not None:
            return cached

        source = SourceUnit.from_definition(definition)
        name = f"__intent_combine_{len(self.combiner_helpers)}"
        location = source.location(source.function)
        physical_types = component_types + component_types + tuple(
            value.type for value in captures
        )
        parameters = tuple(
            ParameterSpec(
                name=f"arg_{index}",
                type=value_type,
                kind=ParameterKind.VALUE,
                location=location,
            )
            for index, value_type in enumerate(physical_types)
        )
        function = self.builder.function(
            name,
            FunctionKind.HELPER,
            parameters,
            component_types,
            location,
            attributes={
                "role": "combiner",
                "component_count": len(component_types),
                "capture_count": len(captures),
            },
        )
        lowerer = FunctionLowerer(
            compiler=self,
            definition=definition,
            source=source,
            function=function,
            constexpr_values={},
        )
        arguments = tuple(parameter.value for parameter in function.parameters)
        lhs_components = arguments[: len(component_types)]
        rhs_components = arguments[len(component_types) : 2 * len(component_types)]
        capture_arguments = arguments[2 * len(component_types) :]

        def group(components: tuple[MlirValue, ...]) -> MlirValue:
            if len(components) == 1 and not isinstance(accumulator_type, RecordType):
                return components[0]
            operation = lowerer.emit(
                OperationKind.MAKE_RECORD,
                location,
                operands=components,
                result_types=(accumulator_type,),
                attributes={"fields": component_names},
            )
            return operation.results[0]

        logical_arguments = (
            group(lhs_components),
            group(rhs_components),
            *capture_arguments,
        )
        results = self.lower_helper_inline(
            lowerer,
            definition,
            tuple(logical_arguments),
            call_location,
        )
        flattened: tuple[MlirValue, ...]
        if len(component_types) == 1 and len(results) == 1 and not isinstance(
            results[0].type, RecordType
        ):
            flattened = results
        elif len(results) == len(component_types) and all(
            result.type == expected
            for result, expected in zip(results, component_types)
        ):
            flattened = results
        elif len(results) == 1 and isinstance(results[0].type, RecordType):
            record = results[0]
            record_type = record.type
            if tuple(field for field, _ in record_type.fields) != component_names:
                raise FrontendError(
                    "combiner result record fields must match the accumulator schema",
                    call_location,
                )
            values: list[MlirValue] = []
            for (field, field_type), expected in zip(record_type.fields, component_types):
                if field_type != expected:
                    raise FrontendError(
                        "combiner result record field types must match the accumulator schema",
                        call_location,
                    )
                values.append(
                    lowerer.emit(
                        OperationKind.EXTRACT,
                        location,
                        operands=(record,),
                        result_types=(field_type,),
                        attributes={"key": field},
                    ).results[0]
                )
            flattened = tuple(values)
        else:
            raise FrontendError(
                "combiner result schema must match the accumulator schema",
                call_location,
            )
        if tuple(value.type for value in flattened) != component_types:
            raise FrontendError(
                "combiner result types must match the accumulator schema",
                call_location,
            )
        lowerer.emit(OperationKind.RETURN, location, operands=flattened)
        symbol = SymbolRef(name)
        self.combiner_helpers[key] = symbol
        return symbol

    def lower_argmax_combiner(
        self,
        value_type: ValueType,
        index_type: ValueType,
        location: object,
    ) -> SymbolRef:
        key = (value_type, index_type)
        cached = self.argmax_helpers.get(key)
        if cached is not None:
            return cached
        name = f"__intent_argmax_{len(self.argmax_helpers)}"
        parameters = tuple(
            ParameterSpec(
                name=name_hint,
                type=parameter_type,
                kind=ParameterKind.VALUE,
                location=location,
            )
            for name_hint, parameter_type in (
                ("lhs_value", value_type),
                ("lhs_index", index_type),
                ("rhs_value", value_type),
                ("rhs_index", index_type),
            )
        )
        function = self.builder.function(
            name,
            FunctionKind.HELPER,
            parameters,
            (value_type, index_type),
            location,
            attributes={
                "role": "combiner",
                "builtin": "argmax_lowest",
                "component_count": 2,
                "capture_count": 0,
            },
        )
        lowerer = FunctionLowerer(
            compiler=self,
            definition=self.definition,
            source=self.source,
            function=function,
            constexpr_values={},
        )
        lhs_value, lhs_index, rhs_value, rhs_index = (
            parameter.value for parameter in function.parameters
        )

        def compare(lhs: MlirValue, rhs: MlirValue, predicate: ComparePredicate) -> MlirValue:
            return lowerer.emit(
                OperationKind.COMPARE,
                location,
                operands=(lhs, rhs),
                result_types=(ScalarType(intent_bool),),
                attributes={"predicate": predicate},
            ).results[0]

        def binary(lhs: MlirValue, rhs: MlirValue, operator: BinaryOperator) -> MlirValue:
            return lowerer.emit(
                OperationKind.BINARY,
                location,
                operands=(lhs, rhs),
                result_types=(lhs.type,),
                attributes={"operator": operator},
            ).results[0]

        greater = compare(lhs_value, rhs_value, ComparePredicate.GT)
        equal = compare(lhs_value, rhs_value, ComparePredicate.EQ)
        lower_index = compare(lhs_index, rhs_index, ComparePredicate.LE)
        tied = binary(equal, lower_index, BinaryOperator.LOGICAL_AND)
        choose_lhs = binary(greater, tied, BinaryOperator.LOGICAL_OR)
        selected_value = lowerer.emit(
            OperationKind.SELECT,
            location,
            operands=(choose_lhs, lhs_value, rhs_value),
            result_types=(value_type,),
        ).results[0]
        selected_index = lowerer.emit(
            OperationKind.SELECT,
            location,
            operands=(choose_lhs, lhs_index, rhs_index),
            result_types=(index_type,),
        ).results[0]
        lowerer.emit(
            OperationKind.RETURN,
            location,
            operands=(selected_value, selected_index),
        )
        symbol = SymbolRef(name)
        self.argmax_helpers[key] = symbol
        return symbol


def lower_to_mlir(
    definition: KernelDefinition[object, object],
    *,
    constexprs: dict[str, object] | None = None,
) -> str:
    if not isinstance(definition, KernelDefinition):
        raise TypeError("lower_to_mlir expects an @intent.kernel definition")
    return FrontendCompiler(definition, dict(constexprs or {})).lower()
