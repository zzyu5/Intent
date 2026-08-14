from __future__ import annotations

import struct
import re
from dataclasses import dataclass
from enum import Enum

from intent.language import DType

from ..semantics.effects import Effect
from ..semantics.operations import AutoExtent
from ..semantics.operations import IndexRelation
from ..semantics.operations import IndexTerm
from .state import MlirValue
from .types import quote


@dataclass(frozen=True, slots=True)
class SymbolRef:
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
            raise ValueError("MLIR symbol reference requires an identifier")


def emit_attribute(value: object) -> str:
    if isinstance(value, SymbolRef):
        return f"@{value.name}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value} : i64"
    if isinstance(value, float):
        bits = int.from_bytes(struct.pack(">d", value), byteorder="big")
        return f"0x{bits:016X} : f64"
    if isinstance(value, str):
        return quote(value)
    if value is None:
        return "unit"
    if isinstance(value, DType):
        return quote(value.name)
    if isinstance(value, Enum):
        return quote(str(value.value))
    if isinstance(value, AutoExtent):
        return "{name = " + quote(value.name) + "}"
    if isinstance(value, IndexRelation):
        return "[" + ", ".join(_emit_index_term(term) for term in value.terms) + "]"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(emit_attribute(element) for element in value) + "]"
    if isinstance(value, dict):
        return emit_dictionary(value)
    raise TypeError(f"unsupported MLIR attribute value {value!r}")


def emit_dictionary(values: dict[str, object]) -> str:
    return "{" + ", ".join(
        f"{_emit_key(key)} = {emit_attribute(value)}"
        for key, value in sorted(values.items())
    ) + "}"


def _emit_key(key: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$.]*", key):
        return key
    return quote(key)


def emit_effect(effect: Effect, operands: tuple[MlirValue, ...]) -> dict[str, object]:
    target = -1
    if effect.target is not None:
        target = operands.index(effect.target)
    return {
        "kind": effect.kind.value,
        "resource": effect.resource.value,
        "target": target,
    }


def _emit_index_term(term: IndexTerm) -> str:
    return emit_dictionary(
        {
            "kind": term.kind.value,
            "operands": term.operand_positions,
            "static": term.static_values,
        }
    )
