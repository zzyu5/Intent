from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeneratedProgram:
    """Provider source, compiler IR and interface; no native callable or runtime."""

    source: str
    ir: str
    metadata: dict[str, object]
