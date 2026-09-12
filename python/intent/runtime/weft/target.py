from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixCapability:
    vlen_bits: int
    isa_feature: str
    vendor_id: int
    architecture_id: int


# Native matching facts from Weft's registered IME fragment contract.
MATRIX_CAPABILITIES = {
    "spacemit-ime1": MatrixCapability(256, "ime", 0x710, 0x8000000058000001),
}


def matrix_capability(name: str | None, vlen_bits: int) -> MatrixCapability | None:
    if name is None:
        return None
    if name not in MATRIX_CAPABILITIES:
        raise NotImplementedError(f"unsupported Weft matrix extension: {name}")
    capability = MATRIX_CAPABILITIES[name]
    if vlen_bits != capability.vlen_bits:
        raise NotImplementedError(f"{name} requires VLEN {capability.vlen_bits}")
    return capability


@dataclass(frozen=True)
class TargetProfile:
    march: str
    abi: str
    vlen_bits: int
    cpus: tuple[int, ...]
    matrix_extension: str | None = None
    required_extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cpus", tuple(self.cpus))
        object.__setattr__(self, "required_extensions", tuple(sorted(set(self.required_extensions))))
        if not self.march.startswith("rv64") or self.abi != "lp64d":
            raise NotImplementedError("native Weft currently requires RV64/lp64d")
        if self.vlen_bits <= 0 or not self.cpus or any(cpu < 0 for cpu in self.cpus):
            raise ValueError("native Weft requires an explicit VLEN and CPU execution set")
        matrix_capability(self.matrix_extension, self.vlen_bits)
        if any(extension != self.matrix_extension for extension in self.required_extensions):
            raise ValueError("required matrix extension is absent from the target profile")

    @classmethod
    def from_deployment(cls, deployment: dict) -> TargetProfile:
        return cls(**{name: deployment[name] for name in cls.__dataclass_fields__})
