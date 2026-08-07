from __future__ import annotations

import struct
import re
from enum import Enum

from intent.ir import AutoExtent
from intent.ir import Effect
from intent.ir import IndexRelation
from intent.ir import IndexTerm
from intent.ir import Value
from intent.language import DType

from .types import quote


def emit_attribute(value: object) -> str:
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


def emit_effect(effect: Effect, operands: tuple[Value, ...]) -> dict[str, object]:
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
