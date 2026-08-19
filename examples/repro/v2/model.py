from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch


TensorTree: TypeAlias = torch.Tensor | tuple["TensorTree", ...]


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float = 0.0


@dataclass(frozen=True)
class PreparedLaunch:
    launch: Callable[[], object]
    outputs: Callable[[], TensorTree]
    prepare: Callable[[], object] | None = None


@dataclass(frozen=True)
class PreparedComparison:
    generated: PreparedLaunch
    source: PreparedLaunch
    tolerance: Tolerance | tuple[Tolerance, ...]
    cuda_graph: bool


@dataclass(frozen=True)
class Context:
    compiler: str
    project_root: Path
    target: object
    provider: str


@dataclass(frozen=True)
class ResultRow:
    kernel: str
    case: str
    generated_p50_ms: float | None
    source_p50_ms: float | None
    ratio: float | None
    status: str


CaseFactory: TypeAlias = Callable[[Context], PreparedComparison]
