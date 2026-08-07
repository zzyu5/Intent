from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from .effects import Effect
from .locations import Location
from .ops import OpCode
from .types import IRType


@dataclass(eq=False, slots=True)
class Value:
    id: int
    type: IRType
    location: Location
    name_hint: str | None = None
    owner: Operation | Block | None = field(default=None, repr=False)
    result_index: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.id, bool) or not isinstance(self.id, int) or self.id < 0:
            raise ValueError("SSA value id must be a non-negative integer")
        if not isinstance(self.type, IRType):
            raise TypeError("SSA value type must be an IRType")
        if not isinstance(self.location, Location):
            raise TypeError("SSA value location must be a Location")
        if self.name_hint is not None and not isinstance(self.name_hint, str):
            raise TypeError("SSA value name hint must be a string")
        if self.result_index is not None and (
            isinstance(self.result_index, bool)
            or not isinstance(self.result_index, int)
            or self.result_index < 0
        ):
            raise ValueError("SSA result index must be a non-negative integer")

    def __hash__(self) -> int:
        return self.id


@dataclass(eq=False, slots=True)
class Region:
    location: Location
    blocks: list[Block] = field(default_factory=list)
    owner: Operation | FunctionBodyOwner | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.location, Location):
            raise TypeError("region location must be a Location")
        if any(not isinstance(block, Block) for block in self.blocks):
            raise TypeError("region blocks must be Block values")
        for block in self.blocks:
            if block.owner is not None and block.owner is not self:
                raise ValueError("block already belongs to another region")
            block.owner = self


@dataclass(eq=False, slots=True)
class Block:
    location: Location
    arguments: list[Value] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    owner: Region | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.location, Location):
            raise TypeError("block location must be a Location")
        if any(not isinstance(argument, Value) for argument in self.arguments):
            raise TypeError("block arguments must be SSA values")
        if any(not isinstance(operation, Operation) for operation in self.operations):
            raise TypeError("block operations must be Operation values")


@dataclass(eq=False, slots=True)
class Operation:
    opcode: OpCode
    location: Location
    operands: tuple[Value, ...] = ()
    result_types: tuple[IRType, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    regions: list[Region] = field(default_factory=list)
    effects: tuple[Effect, ...] = ()
    results: tuple[Value, ...] = field(default=(), init=False)
    owner: Block | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.opcode, OpCode):
            raise TypeError("operation opcode must be an OpCode")
        if not isinstance(self.location, Location):
            raise TypeError("operation location must be a Location")
        if not isinstance(self.attributes, dict):
            raise TypeError("operation attributes must be a dictionary")
        self.regions = list(self.regions)
        self.operands = tuple(self.operands)
        self.result_types = tuple(self.result_types)
        self.effects = tuple(self.effects)
        if any(not isinstance(operand, Value) for operand in self.operands):
            raise TypeError("operation operands must be SSA values")
        if any(not isinstance(result_type, IRType) for result_type in self.result_types):
            raise TypeError("operation result types must be IRType values")
        if any(not isinstance(effect, Effect) for effect in self.effects):
            raise TypeError("operation effects must be Effect values")
        if any(not isinstance(region, Region) for region in self.regions):
            raise TypeError("operation regions must be Region values")
        for region in self.regions:
            if region.owner is not None and region.owner is not self:
                raise ValueError("region already belongs to another operation")
            region.owner = self


class FunctionBodyOwner:
    """Marker protocol implemented structurally by Function."""
