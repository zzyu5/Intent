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
    from cuda.bindings import driver
    import torch

    if not torch.cuda.is_available() or device >= torch.cuda.device_count():
        raise RuntimeError("requested CUDA device is unavailable")

    def require_success(result: tuple[object, ...], operation: str) -> object:
        status, *values = result
        if status != driver.CUresult.CUDA_SUCCESS:
            _, name = driver.cuGetErrorName(status)
            raise RuntimeError(f"{operation} failed: {name.decode()}")
        return values[0] if values else None

    require_success(driver.cuInit(0), "cuInit")
    cuda_device = require_success(driver.cuDeviceGet(device), "cuDeviceGet")

    def attribute(name: object) -> int:
        value = require_success(
            driver.cuDeviceGetAttribute(name, cuda_device),
            f"cuDeviceGetAttribute({name.name})",
        )
        return int(value)

    major = attribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    )
    return GpuDeviceCapabilities(
        device=device,
        compute_units=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        ),
        shared_memory_per_unit=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        ),
        registers_per_unit=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
        ),
        matrix_units=major >= 7,
        dynamic_vector_width=False,
    )
