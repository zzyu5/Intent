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
    native_benchmark: Callable[[], float] | None = None


@dataclass(frozen=True)
class PreparedComparison:
    generated: PreparedLaunch
    source: PreparedLaunch
    tolerance: Tolerance | tuple[Tolerance, ...]
    cuda_graph: bool
    status: str = "pass"
    note: str = ""
    device_type: str = "cuda"


@dataclass(frozen=True)
class Context:
    compiler: str
    project_root: Path
    target: object
    provider: str
    compiler_timeout_seconds: int = 15
    tuning_config: Path | None = None


@dataclass(frozen=True)
class ResultRow:
    kernel: str
    case: str
    generated_p50_ms: float | None
    source_p50_ms: float | None
    ratio: float | None
    status: str
    note: str = ""


CaseFactory: TypeAlias = Callable[[Context], PreparedComparison]


class ComparisonUnavailable(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
