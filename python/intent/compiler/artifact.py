from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GeneratedProgram:
    """Provider source, compiler IR and interface; no native callable or runtime."""

    source: str
    ir: str
    metadata: dict[str, object]
    cache_directory: Path
