from __future__ import annotations

from pathlib import Path

from ...loading import load_module
from ...model import Context


def runtime_module(context: Context, relative_path: str, name: str):
    return load_module(context.project_root / relative_path, name)


def official_source(context: Context, runtime_path: str, name: str):
    runtime = runtime_module(context, runtime_path, f"{name}_runtime")
    return runtime.load_source()


def tilegym_source(
    context: Context,
    source_path: str,
    name: str,
    *,
    needs_utils: bool = False,
    needs_splitk: bool = False,
    needs_gelu: bool = False,
):
    runtime = runtime_module(
        context,
        "source/cutile/tilegym/support/runtime.py",
        "intent_v2_cutile_tilegym_runtime",
    )
    return runtime.load_source(
        Path(context.project_root / source_path),
        f"tilegym.ops.cutile._intent_v2_{name}",
        needs_utils=needs_utils,
        needs_splitk=needs_splitk,
        needs_gelu=needs_gelu,
    )
