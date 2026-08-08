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


class TileKind(Enum):
    ONE = "one"
    NEXT_POWER_OF_TWO = "next_power_of_two"


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
    ROW_MAJOR = "row_major"


class PrimitiveKind(Enum):
    POINTWISE = "pointwise"
    REDUCTION = "reduction"


class TailKind(Enum):
    EXACT = "exact"
    MASKED = "masked"
    LOOP = "tail_loop"


class GridPolicy(Enum):
    STATIC = "static"
    PERSISTENT_OCCUPANCY = "persistent_occupancy"


@dataclass(frozen=True, slots=True)
class TargetInfo:
    backend: BackendKind
    architecture: str
    device: int
    warp_size: int


@dataclass(frozen=True, slots=True)
class ExtentBinding:
    node_id: int
    axis: int
    logical_extent: str
    tile: TileKind


@dataclass(frozen=True, slots=True)
class OwnershipBinding:
    loop_node_id: int
    worker: WorkerKind
    worker_axis: int
    traversal: TraversalKind
    mapping: TraversalKind


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
    axis: int = -1
    identity: str = "none"


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    low_stages: int
    high_stages: int
    smem_threshold: int
    prefetch: bool
    async_copy: bool


@dataclass(frozen=True, slots=True)
class BoundaryBinding:
    node_id: int
    logical_extent: str
    tail: TailKind
    predicate: str
    load_fill: str


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    loop_node_id: int
    grid_policy: GridPolicy
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
