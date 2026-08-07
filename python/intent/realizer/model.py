from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BackendKind(Enum):
    TRITON = "triton"


class WorkerKind(Enum):
    PROGRAM = "program"


class TraversalKind(Enum):
    STATIC = "static"
    GRID_STRIDE = "grid_stride"
    PERSISTENT = "persistent"
    SWIZZLED = "swizzled"


class StorageSpace(Enum):
    GLOBAL = "global"
    REGISTER = "register"
    SHARED = "shared"
    LOCAL = "local"
    SCRATCH = "scratch"


class AccessMode(Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class LayoutKind(Enum):
    CONTIGUOUS = "contiguous"


class PrimitiveKind(Enum):
    POINTWISE = "pointwise"


class TailKind(Enum):
    EXACT = "exact"
    MASKED = "masked"
    LOOP = "tail_loop"


@dataclass(frozen=True, slots=True)
class TargetInfo:
    backend: BackendKind
    architecture: str
    device: int
    warp_size: int


@dataclass(frozen=True, slots=True)
class ExtentBinding:
    loop_node_id: int
    domain_value_id: int
    tile_size: int


@dataclass(frozen=True, slots=True)
class OwnershipBinding:
    loop_node_id: int
    worker: WorkerKind
    worker_axis: int
    traversal: TraversalKind


@dataclass(frozen=True, slots=True)
class StorageBinding:
    value_id: int
    space: StorageSpace
    access: AccessMode


@dataclass(frozen=True, slots=True)
class LayoutBinding:
    value_id: int
    kind: LayoutKind
    order: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrimitiveBinding:
    node_id: int
    kind: PrimitiveKind
    operator: str


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    stages: int
    prefetch: bool
    async_copy: bool


@dataclass(frozen=True, slots=True)
class BoundaryBinding:
    loop_node_id: int
    logical_extent: int
    tail: TailKind


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    loop_node_id: int
    grid: tuple[int, ...]
    block_size: int
    num_warps: int


@dataclass(frozen=True, slots=True)
class PhysicalPlan:
    entry_name: str
    target: TargetInfo
    extents: tuple[ExtentBinding, ...]
    ownership: tuple[OwnershipBinding, ...]
    storage: tuple[StorageBinding, ...]
    layouts: tuple[LayoutBinding, ...]
    primitives: tuple[PrimitiveBinding, ...]
    pipeline: PipelineSpec
    boundaries: tuple[BoundaryBinding, ...]
    launch: LaunchSpec
