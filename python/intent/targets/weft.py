from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolvedWeftTarget:
    vector_bits: int
    workers: int

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return ("--target=weft", f"--cpu-vector-bits={self.vector_bits}",
                f"--cpu-workers={self.workers}")

    @property
    def compiler_role(self) -> str:
        return "Intent CPU to canonical Weft compiler"


@dataclass(frozen=True, slots=True)
class WeftTarget:
    # These are explicit CPU program construction budgets, not ISA discovery.
    vector_bits: int
    workers: int

    def resolve(self) -> ResolvedWeftTarget:
        if self.vector_bits <= 0 or self.workers <= 0:
            raise ValueError("Weft generation requires positive CPU construction budgets")
        return ResolvedWeftTarget(self.vector_bits, self.workers)
