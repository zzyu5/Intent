from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from intent.ir import Block
from intent.ir import Function
from intent.ir import FunctionKind
from intent.ir import Module
from intent.ir import Operation
from intent.ir import ParameterKind
from intent.ir import Region
from intent.ir import Value
from intent.ir import verify

from .attributes import emit_attribute
from .attributes import emit_dictionary
from .attributes import emit_effect
from .types import emit_shape_metadata
from .types import emit_type
from .types import emit_view_type
from .types import quote


@dataclass(slots=True)
class MlirEmitter:
    module: Module
    _block_names: dict[Block, str] = field(default_factory=dict, init=False)
    _next_block: int = field(default=0, init=False)
    _view_types: dict[Value, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._block_names.clear()
        self._view_types.clear()

    def emit(self) -> str:
        attributes = {"intent.source_module": self.module.name}
        attributes.update(
            {f"intent.{key}": value for key, value in self.module.attributes.items()}
        )
        lines = ["module attributes " + emit_dictionary(attributes) + " {"]
        for function in self.module.functions:
            lines.extend(self._emit_function(function, 1))
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _emit_function(self, function: Function, indent: int) -> list[str]:
        prefix = "  " * indent
        for parameter in function.parameters:
            if parameter.spec.kind is ParameterKind.VIEW:
                self._view_types[parameter.value] = parameter.spec.view_kind.name.lower()
        arguments = ", ".join(
            f"{self._value(parameter.value)}: {self._value_type(parameter.value)}"
            for parameter in function.parameters
        )
        result_types = self._function_results(function)
        attributes = {
            "intent.kind": function.kind.value,
            "intent.parameters": [
                self._parameter_metadata(parameter.spec) for parameter in function.parameters
            ],
            "intent.source": function.location.format(),
        }
        attributes.update({f"intent.{key}": value for key, value in function.attributes.items()})
        header = (
            f"{prefix}func.func @{function.name}({arguments}){result_types} "
            f"attributes {emit_dictionary(attributes)} {{"
        )
        lines = [header]
        if len(function.body.blocks) != 1:
            raise NotImplementedError("Intent functions require a single structured entry block")
        entry = function.body.blocks[0]
        for operation in entry.operations:
            lines.extend(self._emit_operation(operation, indent + 1))
        lines.append(f"{prefix}}}")
        return lines

    def _emit_operation(self, operation: Operation, indent: int) -> list[str]:
        prefix = "  " * indent
        results = ""
        if operation.results:
            results = ", ".join(self._value(value) for value in operation.results) + " = "
        operands = ", ".join(self._value(value) for value in operation.operands)
        line = f'{prefix}{results}"intent.{operation.opcode.value}"({operands})'
        lines = [line]
        if operation.regions:
            lines[-1] += " ("
            for index, region in enumerate(operation.regions):
                region_lines = self._emit_region(region, indent + 1)
                if index:
                    lines[-1] += ","
                lines.extend(region_lines)
            lines.append(f"{prefix})")
        attributes = self._operation_attributes(operation)
        if attributes:
            lines[-1] += " " + emit_dictionary(attributes)
        operand_types = ", ".join(self._value_type(value) for value in operation.operands)
        result_types = self._operation_results(operation)
        lines[-1] += f" : ({operand_types}) -> {result_types} {self._location(operation)}"
        return lines

    def _emit_region(self, region: Region, indent: int) -> list[str]:
        prefix = "  " * indent
        lines = [f"{prefix}{{"]
        for block in region.blocks:
            name = self._block_name(block)
            arguments = ", ".join(
                f"{self._value(value)}: {self._value_type(value)}" for value in block.arguments
            )
            signature = f"({arguments})" if arguments else ""
            lines.append(f"{prefix}  ^{name}{signature}:")
            for operation in block.operations:
                lines.extend(self._emit_operation(operation, indent + 2))
        lines.append(f"{prefix}}}")
        return lines

    def _operation_attributes(self, operation: Operation) -> dict[str, object]:
        attributes = {
            f"intent.{key}": value for key, value in operation.attributes.items()
        }
        if operation.effects:
            attributes["intent.effects"] = [
                emit_effect(effect, operation.operands) for effect in operation.effects
            ]
        attributes["intent.result_types"] = [
            value_type.format() for value_type in operation.result_types
        ]
        shapes = [emit_shape_metadata(value_type) for value_type in operation.result_types]
        if any(shape is not None for shape in shapes):
            attributes["intent.result_shapes"] = [
                shape if shape is not None else [] for shape in shapes
            ]
        return attributes

    def _parameter_metadata(self, spec: object) -> dict[str, object]:
        metadata: dict[str, object] = {
            "name": spec.name,
            "kind": spec.kind.value,
            "type": spec.type.format(),
        }
        shape = emit_shape_metadata(spec.type)
        if shape is not None:
            metadata["shape"] = shape
        if spec.kind is ParameterKind.VIEW:
            metadata["view_kind"] = spec.view_kind.name.lower()
            constraints = spec.constraints
            metadata["constraints"] = {
                "strides": list(constraints.strides) if constraints.strides is not None else [],
                "layout": constraints.layout,
                "alignment": constraints.alignment if constraints.alignment is not None else 0,
                "alias": constraints.alias,
                "noalias": constraints.noalias,
            }
        return metadata

    def _function_results(self, function: Function) -> str:
        if not function.result_types:
            return ""
        if len(function.result_types) == 1:
            return " -> " + emit_type(function.result_types[0])
        return " -> (" + ", ".join(emit_type(value) for value in function.result_types) + ")"

    def _operation_results(self, operation: Operation) -> str:
        if not operation.result_types:
            return "()"
        if len(operation.result_types) == 1:
            return emit_type(operation.result_types[0])
        return "(" + ", ".join(emit_type(value) for value in operation.result_types) + ")"

    def _value(self, value: Value) -> str:
        return f"%v{value.id}"

    def _value_type(self, value: Value) -> str:
        access = self._view_types.get(value)
        if access is not None:
            return emit_view_type(value.type, access)
        return emit_type(value.type)

    def _block_name(self, block: Block) -> str:
        if block not in self._block_names:
            self._block_names[block] = f"bb{self._next_block}"
            self._next_block += 1
        return self._block_names[block]

    def _location(self, operation: Operation) -> str:
        span = operation.location.primary
        return (
            "loc("
            + quote(span.filename)
            + f":{span.start_line}:{span.start_column + 1})"
        )


def emit_mlir(module: Module) -> str:
    verify(module)
    return MlirEmitter(module).emit()
