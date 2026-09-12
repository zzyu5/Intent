from __future__ import annotations

from dataclasses import dataclass

from intent.runtime.weft.target import matrix_capability


@dataclass(frozen=True, slots=True)
class ResolvedWeftTarget:
    vector_bits: int
    workers: int
    matrix_i8_i32: bool

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return ("--target=weft", f"--cpu-vector-bits={self.vector_bits}",
                f"--cpu-workers={self.workers}", f"--cpu-matrix-i8-i32={str(self.matrix_i8_i32).lower()}")

    @property
    def compiler_role(self) -> str:
        return "Intent CPU to canonical Weft compiler"


@dataclass(frozen=True, slots=True)
class WeftTarget:
    # These are explicit CPU program construction budgets, not ISA discovery.
    vector_bits: int
    workers: int
    matrix_extension: str | None = None

    def resolve(self) -> ResolvedWeftTarget:
        if self.vector_bits <= 0 or self.workers <= 0:
            raise ValueError("Weft generation requires positive CPU construction budgets")
        capability = matrix_capability(self.matrix_extension, self.vector_bits)
        return ResolvedWeftTarget(self.vector_bits, self.workers, capability is not None)
