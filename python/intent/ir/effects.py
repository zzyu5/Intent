from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .values import Value


class EffectKind(Enum):
    READ = "read"
    WRITE = "write"
    ATOMIC = "atomic"
    FENCE = "fence"
    RNG = "rng"


class ResourceKind(Enum):
    EXTERNAL_VIEW = "external_view"
    LOGICAL_BUFFER = "logical_buffer"
    RNG_STATE = "rng_state"
    ORDERING = "ordering"


@dataclass(frozen=True, slots=True)
class Effect:
    kind: EffectKind
    resource: ResourceKind
    target: Value | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EffectKind):
            raise TypeError("effect kind must be an EffectKind")
        if not isinstance(self.resource, ResourceKind):
            raise TypeError("effect resource must be a ResourceKind")
        if self.target is not None:
            from .values import Value

            if not isinstance(self.target, Value):
                raise TypeError("effect target must be an SSA Value or None")


PURE: tuple[Effect, ...] = ()
