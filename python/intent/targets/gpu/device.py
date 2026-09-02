from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuDeviceCapabilities:
    device: int
    compute_units: int
    shared_memory_per_unit: int
    max_dynamic_shared_memory_per_block: int
    registers_per_unit: int
    max_threads_per_block: int
    compute_capability_major: int
    compute_capability_minor: int
    matrix_units: bool
    dynamic_vector_width: bool

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (
            f"--compute-units={self.compute_units}",
            f"--shared-memory-per-unit={self.shared_memory_per_unit}",
            f"--max-dynamic-shared-memory-per-block={self.max_dynamic_shared_memory_per_block}",
            f"--registers-per-unit={self.registers_per_unit}",
            f"--max-threads-per-block={self.max_threads_per_block}",
            f"--compute-capability-major={self.compute_capability_major}",
            f"--compute-capability-minor={self.compute_capability_minor}",
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
    minor = attribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    )
    return GpuDeviceCapabilities(
        device=device,
        compute_units=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        ),
        shared_memory_per_unit=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        ),
        max_dynamic_shared_memory_per_block=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        ),
        registers_per_unit=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
        ),
        max_threads_per_block=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        ),
        compute_capability_major=major,
        compute_capability_minor=minor,
        matrix_units=major >= 7,
        dynamic_vector_width=False,
    )
