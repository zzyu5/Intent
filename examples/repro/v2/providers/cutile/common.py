from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from intent.runtime.artifact import ParameterRole

from ...loading import load_module
from ...measurement import PipelineStageError
from ...measurement import report_stage
from ...model import Context


def contraction_configs(
    artifact, arguments, *, m_axis, n_axis, k_axis, fixed_options, batch_axis=None,
    k_elements_per_unit=1,
):
    report_stage("generated_tuning_metadata")
    try:
        configurations = artifact.tuning_configurations(*arguments)
    except NotImplementedError as error:
        raise PipelineStageError("generated_tuning_metadata", str(error)) from error
    report_stage("source_candidate_binding")
    result = []
    for configuration in configurations:
        values = {}
        for parameter, value in zip(
            configuration.parameters, configuration.values, strict=True,
        ):
            role, axis = parameter.role, parameter.view_axis
            if role == ParameterRole.PROVIDER_LOAD_POLICY:
                # Source kernels retain their own load scheduling policy.
                continue
            elif role == ParameterRole.PROVIDER_ACCESS_FORM:
                field = "ACCESS_FORM"
            elif role == ParameterRole.PROVIDER_OCCUPANCY:
                field = "occupancy"
            elif role == ParameterRole.PROVIDER_CTAS:
                field = "num_ctas"
            elif role == ParameterRole.TRAVERSAL_GROUP:
                field = "GROUP_SIZE_M"
            elif role == ParameterRole.RESIDENT_WORKERS:
                field = "RESIDENT_WORKERS"
            elif role == ParameterRole.REDUCTION and axis == k_axis:
                field = "TILE_K"
                value *= k_elements_per_unit
            elif role in (ParameterRole.OWNERSHIP_M, ParameterRole.OWNERSHIP_N):
                if axis == m_axis:
                    field = "TILE_M"
                elif axis == n_axis:
                    field = "TILE_N"
                elif batch_axis is not None and axis == batch_axis and value == 1:
                    continue
                else:
                    raise PipelineStageError(
                        "source_candidate_binding",
                        f"source cannot bind ownership parameter {parameter}",
                    )
            else:
                raise PipelineStageError(
                    "source_candidate_binding",
                    f"source cannot bind physical parameter {parameter}",
                )
            if field in values:
                raise ValueError(f"multiple physical parameters bind source field {field}")
            values[field] = value
        if values.keys() & fixed_options.keys():
            raise PipelineStageError("source_candidate_binding", "fixed options overlap tuned parameters")
        values.update(fixed_options)
        required = {"TILE_M", "TILE_N", "TILE_K", "ACCESS_FORM", "occupancy",
                    "GROUP_SIZE_M", "num_ctas"}
        if "RESIDENT_WORKERS" in values:
            required.add("RESIDENT_WORKERS")
        if values.keys() != required:
            raise PipelineStageError(
                "source_candidate_binding", "source requires a complete contraction candidate",
            )
        candidate = SimpleNamespace(**values)
        if candidate not in result:
            result.append(candidate)
    report_stage("adapter_preparation")
    return tuple(result)


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
