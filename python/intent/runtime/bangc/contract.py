"""Pure interpretation of a generated BANG C entry and extent obligations."""
from __future__ import annotations

from dataclasses import dataclass

from ..contract import integer_field, name_field, sequence_field
from ..interface import ViewParameter
from ..native import NativeABI


@dataclass(frozen=True, slots=True)
class BangCFacts:
    entry: str
    full_extent_dimensions: tuple[int, ...]

    @classmethod
    def read(cls, source: str, metadata: dict, abi: NativeABI) -> BangCFacts:
        entry = name_field(metadata["entry"], "BANG C native entry")
        dimensions = tuple(integer_field(value, "BANG C full-extent dimension", minimum=1)
                           for value in sequence_field(metadata["full_extent_dimensions"],
                                                       "BANG C full-extent dimensions"))
        known = {identity for parameter in abi.interface.parameters if isinstance(parameter, ViewParameter)
                 for identity in parameter.dimensions if identity > 0}
        if len(set(dimensions)) != len(dimensions) or not set(dimensions).issubset(known):
            raise ValueError("BANG C full-extent obligations must name distinct public dimensions")
        return cls(entry, dimensions)
