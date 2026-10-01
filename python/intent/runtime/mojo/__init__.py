from __future__ import annotations

from intent.runtime.artifact import CompiledArtifact

from .program import NativeProgram


def materialize_mojo_artifact(source: str, module_text: str,
                             metadata: dict[str, object], target) -> CompiledArtifact:
    program = NativeProgram(source, metadata, target)
    return CompiledArtifact(
        source=source, mlir=module_text, device=0, device_type="cpu",
        runtime=program,
        _backend_ir_collector=None,
    )
