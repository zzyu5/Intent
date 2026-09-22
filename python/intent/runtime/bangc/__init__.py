from __future__ import annotations

import json
from pathlib import Path

from intent.runtime.artifact import CompiledArtifact
from .buffer import DeviceBuffer, DeviceView
from .compilation import compile_library
from .program import NativeProgram


def materialize_bangc_artifact(source, module_text, metadata, target) -> CompiledArtifact:
    program = NativeProgram(source, metadata, target)

    def launch(*arguments):
        program.prepare(arguments, explicit_outputs=True).launch()

    def run(*arguments):
        call = program.prepare(arguments)
        call.launch()
        return call.result()

    return CompiledArtifact(source, module_text, target.device, launch, run, None,
                            {"program": program}, device_type="mlu")


def export_artifact(program, directory: str | Path) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "kernel.mlu").write_text(program.source)
    (path / "program.mlir").write_text(program.ir)
    (path / "artifact.json").write_text(json.dumps(program.metadata, indent=2))
    return path


def load_artifact(directory: str | Path, target=None) -> CompiledArtifact:
    from intent.targets.bangc import BangCTarget
    path = Path(directory)
    metadata = json.loads((path / "artifact.json").read_text())
    if target is None:
        target = BangCTarget(**{name: metadata[name] for name in
            ("architecture", "tile", "tile_m", "tile_n", "tile_k", "region_tile", "tasks", "local_bytes")})
    return materialize_bangc_artifact((path / "kernel.mlu").read_text(),
        (path / "program.mlir").read_text(), metadata, target.resolve())


__all__ = ["DeviceBuffer", "DeviceView", "NativeProgram", "compile_library", "export_artifact", "load_artifact"]
