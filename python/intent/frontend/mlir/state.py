from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum

from intent.language.annotations import ViewConstraints
from intent.language.annotations import ViewKind

from ..diagnostics.locations import Location
from ..semantics.effects import Effect
from ..semantics.operations import OperationKind
from ..semantics.types import ValueType


class ParameterKind(Enum):
    VIEW = "view"
    RUNTIME_SCALAR = "runtime_scalar"
    CONSTEXPR = "constexpr"
    VALUE = "value"


class FunctionKind(Enum):
    KERNEL = "kernel"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    type: ValueType
    kind: ParameterKind
    location: Location
    view_kind: ViewKind | None = None
    constraints: ViewConstraints | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid parameter name: {self.name!r}")
        if not isinstance(self.type, ValueType):
            raise TypeError("parameter type must be a frontend ValueType")
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


@dataclass(eq=False, slots=True)
class MlirValue:
    id: int
    type: ValueType
    location: Location
    name_hint: str | None = None
    view_access: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.id, bool) or not isinstance(self.id, int) or self.id < 0:
            raise ValueError("MLIR value id must be a non-negative integer")
        if not isinstance(self.type, ValueType):
            raise TypeError("MLIR value requires a frontend ValueType")
        if not isinstance(self.location, Location):
            raise TypeError("MLIR value location must be a Location")

    def __hash__(self) -> int:
        return self.id


@dataclass(frozen=True, slots=True)
class ParameterState:
    spec: ParameterSpec
    value: MlirValue


@dataclass(eq=False, slots=True)
class RegionState:
    location: Location
    blocks: list[BlockState] = field(default_factory=list)
    owner_operation: int | None = None

    @property
    def effects(self) -> tuple[Effect, ...]:
        return tuple(effect for block in self.blocks for effect in block.effects)


@dataclass(eq=False, slots=True)
class BlockState:
    location: Location
    arguments: list[MlirValue] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    effects: list[Effect] = field(default_factory=list)
    effect_markers: list[bool] = field(default_factory=list)
    last_operation: OperationKind | None = None
    owner: RegionState | None = None

    @property
    def operation_count(self) -> int:
        return len(self.effect_markers)

    def has_effect_since(self, operation_count: int) -> bool:
        return any(self.effect_markers[operation_count:])


@dataclass(eq=False, slots=True)
class FunctionState:
    name: str
    kind: FunctionKind
    parameters: tuple[ParameterState, ...]
    result_types: tuple[ValueType, ...]
    body: RegionState
    location: Location
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmittedOperation:
    id: int
    results: tuple[MlirValue, ...]
