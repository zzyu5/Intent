from __future__ import annotations

from collections.abc import Iterable

from ..diagnostics.locations import Location
from ..semantics.effects import Effect
from ..semantics.operations import OperationKind
from ..semantics.operations import REGION_OPS
from ..semantics.operations import TERMINATORS
from ..semantics.types import ValueType
from .attributes import emit_dictionary
from .attributes import emit_effect
from .state import BlockState
from .state import EmittedOperation
from .state import FunctionKind
from .state import FunctionState
from .state import MlirValue
from .state import ParameterKind
from .state import ParameterSpec
from .state import ParameterState
from .state import RegionState
from .types import emit_shape_metadata
from .types import emit_type
from .types import emit_type_metadata
from .types import emit_view_type
from .types import quote


class MlirBuilder:
    def __init__(self, module_name: str, location: Location) -> None:
        self.module_name = module_name
        self.location = location
        self.functions: list[FunctionState] = []
        self._values: list[MlirValue] = []
        self._next_value_id = 0
        self._next_operation_id = 0

    def _value(
        self,
        value_type: ValueType,
        location: Location,
        *,
        name_hint: str | None = None,
        view_access: str | None = None,
    ) -> MlirValue:
        value = MlirValue(
            id=self._next_value_id,
            type=value_type,
            location=location,
            name_hint=name_hint,
            view_access=view_access,
        )
        self._next_value_id += 1
        self._values.append(value)
        return value

    def region(
        self,
        location: Location,
        argument_types: Iterable[ValueType] = (),
        argument_names: Iterable[str | None] = (),
    ) -> RegionState:
        types = tuple(argument_types)
        names = tuple(argument_names)
        if names and len(names) != len(types):
            raise ValueError("region argument names and types must have equal length")
        if not names:
            names = (None,) * len(types)
        region = RegionState(location)
        block = BlockState(location, owner=region)
        region.blocks.append(block)
        block.arguments.extend(
            self._value(value_type, location, name_hint=name)
            for value_type, name in zip(types, names)
        )
        return region

    def function(
        self,
        name: str,
        kind: FunctionKind,
        parameters: Iterable[ParameterSpec],
        result_types: Iterable[ValueType],
        location: Location,
        *,
        attributes: dict[str, object] | None = None,
    ) -> FunctionState:
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise ValueError(f"invalid function name: {name!r}")
        parameter_specs = tuple(parameters)
        body = self.region(
            location,
            (parameter.type for parameter in parameter_specs),
            (parameter.name for parameter in parameter_specs),
        )
        values = body.blocks[0].arguments
        for spec, value in zip(parameter_specs, values):
            if spec.kind is ParameterKind.VIEW:
                value.view_access = spec.view_kind.name.lower()
        function = FunctionState(
            name=name,
            kind=kind,
            parameters=tuple(
                ParameterState(spec, value) for spec, value in zip(parameter_specs, values)
            ),
            result_types=tuple(result_types),
            body=body,
            location=location,
            attributes=dict(attributes or {}),
        )
        self.functions.append(function)
        return function

    def emit(
        self,
        block: BlockState,
        operation_kind: OperationKind,
        location: Location,
        *,
        operands: Iterable[MlirValue] = (),
        result_types: Iterable[ValueType] = (),
        attributes: dict[str, object] | None = None,
        regions: Iterable[RegionState] = (),
        effects: Iterable[Effect] = (),
        result_names: Iterable[str | None] = (),
    ) -> EmittedOperation:
        if not isinstance(block, BlockState):
            raise TypeError("MLIR emission requires a BlockState")
        if not isinstance(operation_kind, OperationKind):
            raise TypeError("MLIR emission requires an OperationKind")
        if block.last_operation in TERMINATORS:
            raise ValueError("cannot emit after a block terminator")
        operands = tuple(operands)
        result_types = tuple(result_types)
        regions = tuple(regions)
        effects = tuple(effects)
        names = tuple(result_names)
        if any(not isinstance(value, MlirValue) for value in operands):
            raise TypeError("operation operands must be MLIR values")
        if any(not isinstance(value_type, ValueType) for value_type in result_types):
            raise TypeError("operation result types must be frontend ValueType values")
        if any(not isinstance(effect, Effect) for effect in effects):
            raise TypeError("operation effects must be Effect values")
        if any(effect.target is not None and effect.target not in operands for effect in effects):
            raise ValueError("effect target must be one of the operation operands")
        expected_regions = 2 if operation_kind in (OperationKind.IF, OperationKind.WHILE) else 1
        if operation_kind in REGION_OPS and len(regions) != expected_regions:
            raise ValueError(
                f"intent.{operation_kind.value} requires {expected_regions} region(s)"
            )
        if operation_kind not in REGION_OPS and regions:
            raise ValueError(f"intent.{operation_kind.value} does not own regions")
        for region in regions:
            if region.owner_operation is not None:
                raise ValueError("region is already attached to an operation")
            if len(region.blocks) != 1:
                raise NotImplementedError("Intent structured regions require one block")
            if region.blocks[0].last_operation not in TERMINATORS:
                raise ValueError("Intent structured region must end in a terminator")
        if names and len(names) != len(result_types):
            raise ValueError("result names and result types must have equal length")
        if not names:
            names = (None,) * len(result_types)

        operation_id = self._next_operation_id
        self._next_operation_id += 1
        results = tuple(
            self._value(value_type, location, name_hint=name)
            for value_type, name in zip(result_types, names)
        )
        operation_attributes = {
            f"intent.{key}": value for key, value in dict(attributes or {}).items()
        }
        operation_attributes["intent.node"] = operation_id
        operation_attributes["intent.result_nodes"] = [value.id for value in results]
        operation_attributes["intent.result_names"] = [
            self._name_marker(value) for value in results
        ]
        operation_attributes["intent.result_types"] = [
            emit_type_metadata(value_type) for value_type in result_types
        ]
        shapes = [emit_shape_metadata(value_type) for value_type in result_types]
        if any(shape is not None for shape in shapes):
            operation_attributes["intent.result_shapes"] = [
                shape if shape is not None else [] for shape in shapes
            ]
        if regions:
            operation_attributes["intent.region_argument_nodes"] = [
                [[value.id for value in nested.arguments] for nested in region.blocks]
                for region in regions
            ]
            operation_attributes["intent.region_argument_names"] = [
                [
                    [self._name_marker(value) for value in nested.arguments]
                    for nested in region.blocks
                ]
                for region in regions
            ]
        if effects:
            operation_attributes["intent.effects"] = [
                emit_effect(effect, operands) for effect in effects
            ]

        text = self._operation_text(
            operation_kind,
            location,
            operands,
            results,
            result_types,
            operation_attributes,
            regions,
        )
        nested_effects = tuple(effect for region in regions for effect in region.effects)
        block.lines.append(text)
        block.effects.extend(effects)
        block.effects.extend(nested_effects)
        block.effect_markers.append(bool(effects or nested_effects))
        block.last_operation = operation_kind
        for region in regions:
            region.owner_operation = operation_id
        return EmittedOperation(operation_id, results)

    def emit_module(self) -> str:
        attributes = emit_dictionary({"intent.source_module": self.module_name})
        lines = [f"module attributes {attributes} {{"]
        for function in self.functions:
            lines.extend(self._function_lines(function, 1))
        lines.append("}")
        assembly = "\n".join(lines) + "\n"
        for value in self._values:
            assembly = assembly.replace(
                quote(self._name_marker(value)),
                quote(value.name_hint or f"v{value.id}"),
            )
        return assembly

    def _function_lines(self, function: FunctionState, indent: int) -> list[str]:
        prefix = "  " * indent
        arguments = ", ".join(
            f"{self._value_name(parameter.value)}: {self._value_type(parameter.value)}"
            for parameter in function.parameters
        )
        attributes = {
            "intent.kind": function.kind.value,
            "intent.parameters": [
                self._parameter_metadata(parameter.spec) for parameter in function.parameters
            ],
            "intent.parameter_nodes": [parameter.value.id for parameter in function.parameters],
            "intent.results": [self._type_metadata(value) for value in function.result_types],
            "intent.source": function.location.format(),
        }
        attributes.update({f"intent.{key}": value for key, value in function.attributes.items()})
        result_types = self._function_result_types(function.result_types)
        lines = [
            f"{prefix}func.func @{function.name}({arguments}){result_types} "
            f"attributes {emit_dictionary(attributes)} {{"
        ]
        if len(function.body.blocks) != 1:
            raise NotImplementedError("Intent functions require one entry block")
        for operation in function.body.blocks[0].lines:
            lines.extend(self._indent_text(operation, indent + 1))
        lines.append(f"{prefix}}}")
        return lines

    def _operation_text(
        self,
        operation_kind: OperationKind,
        location: Location,
        operands: tuple[MlirValue, ...],
        results: tuple[MlirValue, ...],
        result_types: tuple[ValueType, ...],
        attributes: dict[str, object],
        regions: tuple[RegionState, ...],
    ) -> str:
        result_prefix = ""
        if results:
            result_prefix = ", ".join(self._value_name(value) for value in results) + " = "
        operand_names = ", ".join(self._value_name(value) for value in operands)
        lines = [f'{result_prefix}"intent.{operation_kind.value}"({operand_names})']
        if regions:
            lines[0] += " ("
            for index, region in enumerate(regions):
                if index:
                    lines[-1] += ","
                lines.extend(self._region_lines(region, 1))
            lines.append(")")
        if attributes:
            lines[-1] += " " + emit_dictionary(attributes)
        operand_types = ", ".join(self._value_type(value) for value in operands)
        lines[-1] += (
            f" : ({operand_types}) -> {self._result_types(result_types)} "
            f"{self._location(location)}"
        )
        return "\n".join(lines)

    def _region_lines(self, region: RegionState, indent: int) -> list[str]:
        prefix = "  " * indent
        lines = [f"{prefix}{{"]
        for block_index, block in enumerate(region.blocks):
            arguments = ", ".join(
                f"{self._value_name(value)}: {self._value_type(value)}"
                for value in block.arguments
            )
            signature = f"({arguments})" if arguments else ""
            lines.append(f"{prefix}  ^bb{block_index}{signature}:")
            for operation in block.lines:
                lines.extend(self._indent_text(operation, indent + 2))
        lines.append(f"{prefix}}}")
        return lines

    def _parameter_metadata(self, spec: ParameterSpec) -> dict[str, object]:
        metadata: dict[str, object] = {
            "name": spec.name,
            "kind": spec.kind.value,
            "type": emit_type_metadata(spec.type),
        }
        shape = emit_shape_metadata(spec.type)
        if shape is not None:
            metadata["shape"] = shape
        if spec.kind is ParameterKind.VIEW:
            metadata["view_kind"] = spec.view_kind.name.lower()
            constraints = spec.constraints
            metadata["constraints"] = {
                "strides": list(constraints.strides) if constraints.strides is not None else None,
                "layout": constraints.layout,
                "alignment": constraints.alignment,
                "alias": constraints.alias,
                "noalias": constraints.noalias,
            }
        return metadata

    def _type_metadata(self, value_type: ValueType) -> dict[str, object]:
        metadata: dict[str, object] = {"type": emit_type_metadata(value_type)}
        shape = emit_shape_metadata(value_type)
        if shape is not None:
            metadata["shape"] = shape
        return metadata

    def _value_type(self, value: MlirValue) -> str:
        if value.view_access is not None:
            return emit_view_type(value.type, value.view_access)
        return emit_type(value.type)

    def _value_name(self, value: MlirValue) -> str:
        return f"%v{value.id}"

    def _name_marker(self, value: MlirValue) -> str:
        return f"__intent_value_name_{value.id}__"

    def _result_types(self, result_types: tuple[ValueType, ...]) -> str:
        if not result_types:
            return "()"
        if len(result_types) == 1:
            return emit_type(result_types[0])
        return "(" + ", ".join(emit_type(value) for value in result_types) + ")"

    def _function_result_types(self, result_types: tuple[ValueType, ...]) -> str:
        if not result_types:
            return ""
        if len(result_types) == 1:
            return " -> " + emit_type(result_types[0])
        return " -> (" + ", ".join(emit_type(value) for value in result_types) + ")"

    def _location(self, location: Location) -> str:
        span = location.primary
        return f"loc({quote(span.filename)}:{span.start_line}:{span.start_column + 1})"

    def _indent_text(self, text: str, indent: int) -> list[str]:
        prefix = "  " * indent
        return [prefix + line for line in text.splitlines()]


def canonicalize_mlir(assembly: str) -> str:
    from mlir.dialects import func as _func
    from mlir.ir import Context
    from mlir.ir import Module

    del _func
    with Context() as context:
        context.allow_unregistered_dialects = True
        module = Module.parse(assembly)
        return str(module) + "\n"
