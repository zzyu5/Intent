from __future__ import annotations

from dataclasses import dataclass

from .provider import Provider
from .specification import CPUCompilationTarget


@dataclass(frozen=True, slots=True)
class WeftTarget:
    # These are explicit CPU program construction budgets, not ISA discovery.
    vector_bits: int
    workers: int
    matrix_extension: str | None = None
    private_bytes: int = 262144

    def resolve(self) -> CPUCompilationTarget:
        from intent.runtime.weft.target import matrix_capability
        if self.vector_bits <= 0 or self.workers <= 0:
            raise ValueError("Weft generation requires positive CPU construction budgets")
        capability = matrix_capability(self.matrix_extension, self.vector_bits)
        return CPUCompilationTarget("weft", self.vector_bits, self.workers,
                                    capability is not None, self.private_bytes)


def _read_facts(source, metadata, abi):
    from intent.runtime.weft.contract import WeftFacts
    return WeftFacts.read(source, metadata, abi)


def _environment_checks(target):
    return ()


PROVIDER = Provider("cpu", WeftTarget, _read_facts, None, _environment_checks,
                    "Weft source generation prerequisites only; the external AOT toolchain and RISC-V device were not checked")
