from .buffer import Buffer
from .target import TargetProfile


def materialize_weft_artifact(program, target):
    from intent.runtime.artifact import CompiledArtifact
    from .compilation import export_artifact, compile_artifact
    from .program import NativeProgram

    exported = export_artifact(program, compiler=target.compiler, profile=target.profile)
    native = compile_artifact(exported, cc=target.cc, cflags=target.cflags)
    return CompiledArtifact(source=program.source, mlir=program.ir,
                            device=0, device_type="cpu", runtime=NativeProgram(native),
                            _backend_ir_collector=None)


__all__ = ["Buffer", "TargetProfile", "WeftArtifact", "NativeProgram", "export_artifact", "compile_artifact"]


def __getattr__(name):
    if name == "NativeProgram":
        from .program import NativeProgram
        return NativeProgram
    if name in {"WeftArtifact", "export_artifact", "compile_artifact"}:
        from . import compilation
        return getattr(compilation, name)
    raise AttributeError(name)
