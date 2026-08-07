from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum

from intent.language.annotations import ViewConstraints
from intent.language.annotations import ViewKind

from .locations import Location
from .types import IRType
from .values import Region
from .values import Value


class ParameterKind(Enum):
    VIEW = "view"
    RUNTIME_SCALAR = "runtime_scalar"
    CONSTEXPR = "constexpr"
    VALUE = "value"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    type: IRType
    kind: ParameterKind
    location: Location
    view_kind: ViewKind | None = None
    constraints: ViewConstraints | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid parameter name: {self.name!r}")
        if not isinstance(self.type, IRType):
            raise TypeError("parameter type must be an IRType")
        if not isinstance(self.kind, ParameterKind):
            raise TypeError("parameter kind must be a ParameterKind")
        if not isinstance(self.location, Location):
            raise TypeError("parameter location must be a Location")
        if self.kind is ParameterKind.VIEW:
            if not isinstance(self.view_kind, ViewKind) or not isinstance(
                self.constraints, ViewConstraints
            ):
                raise ValueError("view parameter requires view kind and constraints")
        elif self.view_kind is not None or self.constraints is not None:
            raise ValueError("only view parameters may carry view metadata")


@dataclass(frozen=True, slots=True)
class Parameter:
    spec: ParameterSpec
    value: Value

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ParameterSpec):
            raise TypeError("parameter spec must be a ParameterSpec")
        if not isinstance(self.value, Value):
            raise TypeError("parameter value must be an SSA value")


class FunctionKind(Enum):
    KERNEL = "kernel"
    HELPER = "helper"


@dataclass(eq=False, slots=True)
class Function:
    name: str
    kind: FunctionKind
    parameters: tuple[Parameter, ...]
    result_types: tuple[IRType, ...]
    body: Region
    location: Location
    attributes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parameters = tuple(self.parameters)
        self.result_types = tuple(self.result_types)
        if any(not isinstance(parameter, Parameter) for parameter in self.parameters):
            raise TypeError("function parameters must be Parameter values")
        if any(not isinstance(result_type, IRType) for result_type in self.result_types):
            raise TypeError("function result types must be IRType values")
        if not isinstance(self.kind, FunctionKind):
            raise TypeError("function kind must be a FunctionKind")
        if not isinstance(self.body, Region):
            raise TypeError("function body must be a Region")
        if not isinstance(self.location, Location):
            raise TypeError("function location must be a Location")
        if not isinstance(self.attributes, dict):
            raise TypeError("function attributes must be a dictionary")
        if self.body.owner is not None and self.body.owner is not self:
            raise ValueError("function body region already belongs to another owner")
        self.body.owner = self
        if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid function name: {self.name!r}")


@dataclass(eq=False, slots=True)
class Module:
    name: str
    functions: list[Function]
    location: Location
    attributes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid module name: {self.name!r}")
        if not isinstance(self.location, Location):
            raise TypeError("module location must be a Location")
        if not isinstance(self.attributes, dict):
            raise TypeError("module attributes must be a dictionary")
        if any(not isinstance(function, Function) for function in self.functions):
            raise TypeError("module functions must be Function values")

    def symbol_table(self) -> dict[str, Function]:
        return {function.name: function for function in self.functions}

    def entry(self) -> Function:
        entries = [function for function in self.functions if function.kind is FunctionKind.KERNEL]
        if len(entries) != 1:
            raise ValueError("Intent module must contain exactly one kernel entry")
        return entries[0]
