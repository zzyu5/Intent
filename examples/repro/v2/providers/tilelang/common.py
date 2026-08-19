from __future__ import annotations

from ...loading import load_module
from ...model import Context


def runtime_module(context: Context, relative_path: str, name: str):
    return load_module(context.project_root / relative_path, name)


def source_from_runtime(context: Context, runtime_path: str, name: str):
    runtime = runtime_module(context, runtime_path, f"{name}_runtime")
    return runtime, runtime.load_source()
