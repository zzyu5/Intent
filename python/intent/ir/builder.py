from __future__ import annotations

from collections.abc import Iterable

from .effects import Effect
from .locations import Location
from .module import Function
from .module import FunctionKind
from .module import Module
from .module import Parameter
from .module import ParameterSpec
from .ops import OpCode
from .types import IRType
from .values import Block
from .values import Operation
from .values import Region
from .values import Value


class IRBuilder:
    def __init__(self, module_name: str, location: Location) -> None:
        self.module = Module(module_name, [], location)
        self._next_value_id = 0

    def _value(
        self,
        value_type: IRType,
        location: Location,
        *,
        name_hint: str | None = None,
        owner: Operation | Block | None = None,
        result_index: int | None = None,
    ) -> Value:
        value = Value(
            id=self._next_value_id,
            type=value_type,
            location=location,
            name_hint=name_hint,
            owner=owner,
            result_index=result_index,
        )
        self._next_value_id += 1
        return value

    def region(
        self,
        location: Location,
        argument_types: Iterable[IRType] = (),
        argument_names: Iterable[str | None] = (),
    ) -> Region:
        region = Region(location)
        block = Block(location, owner=region)
        region.blocks.append(block)
        names = tuple(argument_names)
        types = tuple(argument_types)
        if names and len(names) != len(types):
            raise ValueError("region argument names and types must have equal length")
        if not names:
            names = (None,) * len(types)
        block.arguments.extend(
            self._value(
                value_type,
                location,
                name_hint=name,
                owner=block,
                result_index=index,
            )
            for index, (value_type, name) in enumerate(zip(types, names))
        )
        return region

    def function(
        self,
        name: str,
        kind: FunctionKind,
        parameters: Iterable[ParameterSpec],
        result_types: Iterable[IRType],
        location: Location,
        *,
        attributes: dict[str, object] | None = None,
    ) -> Function:
        parameter_specs = tuple(parameters)
        body = self.region(
            location,
            (parameter.type for parameter in parameter_specs),
            (parameter.name for parameter in parameter_specs),
        )
        block = body.blocks[0]
        function = Function(
            name=name,
            kind=kind,
            parameters=tuple(
                Parameter(spec, value)
                for spec, value in zip(parameter_specs, block.arguments)
            ),
            result_types=tuple(result_types),
            body=body,
            location=location,
            attributes=dict(attributes or {}),
        )
        self.module.functions.append(function)
        return function

    def emit(
        self,
        block: Block,
        opcode: OpCode,
        location: Location,
        *,
        operands: Iterable[Value] = (),
        result_types: Iterable[IRType] = (),
        attributes: dict[str, object] | None = None,
        regions: Iterable[Region] = (),
        effects: Iterable[Effect] = (),
        result_names: Iterable[str | None] = (),
    ) -> Operation:
        operation = Operation(
            opcode=opcode,
            location=location,
            operands=tuple(operands),
            result_types=tuple(result_types),
            attributes=dict(attributes or {}),
            regions=list(regions),
            effects=tuple(effects),
            owner=block,
        )
        names = tuple(result_names)
        if names and len(names) != len(operation.result_types):
            raise ValueError("result names and result types must have equal length")
        if not names:
            names = (None,) * len(operation.result_types)
        operation.results = tuple(
            self._value(
                result_type,
                location,
                name_hint=name,
                owner=operation,
                result_index=index,
            )
            for index, (result_type, name) in enumerate(
                zip(operation.result_types, names)
            )
        )
        for region in operation.regions:
            if region.owner is not operation:
                raise ValueError("operation region ownership changed during construction")
            region.owner = operation
        block.operations.append(operation)
        return operation
