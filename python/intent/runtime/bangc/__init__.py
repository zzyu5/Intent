from __future__ import annotations

from pathlib import Path

from intent.runtime.artifact import CompiledArtifact
from .buffer import DeviceBuffer, DeviceView
from .compilation import compile_library
from .program import NativeProgram


def materialize_bangc_artifact(source, module_text, metadata, target) -> CompiledArtifact:
    program = NativeProgram(source, metadata, target)

    return CompiledArtifact(source=source, mlir=module_text, device=target.device,
                            runtime=program, _backend_ir_collector=None,
                            device_type="mlu")


def export_artifact(program, directory: str | Path) -> Path:
    return program.save(directory)


def load_artifact(directory: str | Path, *, target) -> CompiledArtifact:
    from intent.compiler.artifact import GeneratedProgram

    return GeneratedProgram.load(directory).materialize(target=target)


__all__ = ["DeviceBuffer", "DeviceView", "NativeProgram", "compile_library", "export_artifact", "load_artifact"]
