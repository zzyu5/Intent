from .buffer import Buffer
from .target import TargetProfile


__all__ = ["Buffer", "TargetProfile", "NativeProgram", "export_artifact", "compile_artifact"]


def __getattr__(name):
    if name == "NativeProgram":
        from .program import NativeProgram
        return NativeProgram
    if name in {"export_artifact", "compile_artifact"}:
        from . import compilation
        return getattr(compilation, name)
    raise AttributeError(name)
