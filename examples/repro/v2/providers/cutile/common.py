from __future__ import annotations

from pathlib import Path

from ...loading import load_module
from ...measurement import PipelineStageError
from ...measurement import report_stage
from ...model import Context


def runtime_module(context: Context, relative_path: str, name: str):
    report_stage("source_loading")
    try:
        module = load_module(context.project_root / relative_path, name)
    except Exception as error:
        raise PipelineStageError("source_loading", str(error)) from error
    report_stage("adapter_preparation")
    return module


def official_source(context: Context, runtime_path: str, name: str):
    runtime = runtime_module(context, runtime_path, f"{name}_runtime")
    report_stage("source_loading")
    try:
        module = runtime.load_source()
    except Exception as error:
        raise PipelineStageError("source_loading", str(error)) from error
    report_stage("adapter_preparation")
    return module


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
    module_name = (
        f"tilegym.ops.cutile.activation._intent_v2_{name}"
        if needs_gelu
        else f"tilegym.ops.cutile._intent_v2_{name}"
    )
    report_stage("source_loading")
    try:
        module = runtime.load_source(
            Path(context.project_root / source_path),
            module_name,
            needs_utils=needs_utils,
            needs_splitk=needs_splitk,
            needs_gelu=needs_gelu,
        )
    except Exception as error:
        raise PipelineStageError("source_loading", str(error)) from error
    report_stage("adapter_preparation")
    return module
