from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuDeviceCapabilities:
    device: int
    compute_units: int
    shared_memory_per_unit: int
    registers_per_unit: int
    matrix_units: bool
    dynamic_vector_width: bool

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (
            f"--device={self.device}",
            f"--compute-units={self.compute_units}",
            f"--shared-memory-per-unit={self.shared_memory_per_unit}",
            f"--registers-per-unit={self.registers_per_unit}",
            f"--matrix-units={'true' if self.matrix_units else 'false'}",
            f"--dynamic-vector-width={'true' if self.dynamic_vector_width else 'false'}",
        )


def resolve_gpu_device(device: int) -> GpuDeviceCapabilities:
    import torch

    if not torch.cuda.is_available() or device >= torch.cuda.device_count():
        raise RuntimeError("requested CUDA device is unavailable")
    properties = torch.cuda.get_device_properties(device)
    return GpuDeviceCapabilities(
        device=device,
        compute_units=properties.multi_processor_count,
        shared_memory_per_unit=properties.shared_memory_per_multiprocessor,
        registers_per_unit=properties.regs_per_multiprocessor,
        matrix_units=properties.major >= 7,
        dynamic_vector_width=False,
    )
