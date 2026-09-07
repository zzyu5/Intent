from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from intent.runtime.artifact import ParameterRole
from kernels.streaming.gated_delta import chunk_gated_delta_prepare
from kernels.streaming.gated_delta import chunk_gated_delta_recurrence
from kernels.streaming.gated_delta import recurrent_gated_delta_fwd

from ...measurement import compile_single
from ...measurement import functional_launch
from ...measurement import PipelineStageError
from ...measurement import report_stage
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import tilegym_source


def recurrent_gated_delta(context: Context) -> PreparedComparison:
    batch, sequence, heads, key_dimension, value_dimension = 2, 2048, 8, 128, 128
    query = torch.randn(
        (batch, sequence, heads, key_dimension),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn(
        (batch, sequence, heads, value_dimension),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    gate = -torch.rand(
        (batch, sequence, heads), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    scale = 1.0 / math.sqrt(key_dimension)
    artifact, generated = compile_single(
        context,
        recurrent_gated_delta_fwd,
        (query, key, value, gate, beta, scale),
        constexprs={"HEAD_GROUP": 1},
    )
    report_stage("generated_tuning_metadata")
    try:
        configurations = artifact.tuning_configurations(
            query, key, value, gate, beta, *generated.outputs(), scale,
        )
    except NotImplementedError as error:
        raise PipelineStageError("generated_tuning_metadata", str(error)) from error
    report_stage("source_candidate_binding")
    configs = []
    for configuration in configurations:
        values = {}
        for parameter, parameter_value in zip(configuration.parameters, configuration.values, strict=True):
            role, axis = parameter.role, parameter.view_axis
            if role == ParameterRole.PROVIDER_ACCESS_FORM:
                field = "ACCESS_FORM"
            elif role == ParameterRole.PROVIDER_OCCUPANCY:
                field = "occupancy"
            elif role == ParameterRole.FULL_COVERAGE and axis == (0, 3):
                field = "BLOCK_K"
            elif role in (ParameterRole.OWNERSHIP_M, ParameterRole.OWNERSHIP_N) and axis == (2, 3):
                field = "BLOCK_V"
            else:
                raise PipelineStageError("source_candidate_binding",
                                         f"source cannot bind recurrent parameter {parameter}")
            if field in values:
                raise PipelineStageError("source_candidate_binding", f"duplicate recurrent field {field}")
            values[field] = parameter_value
        if values.keys() != {"ACCESS_FORM", "occupancy", "BLOCK_K", "BLOCK_V"}:
            raise PipelineStageError("source_candidate_binding", "incomplete recurrent candidate")
        configs.append(SimpleNamespace(**values))
    configs = tuple(configs)
    report_stage("adapter_preparation")
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/scan/gated_delta_recurrent/recurrent_gated_delta_rule.py",
        "recurrent_gated_delta",
        needs_utils=True,
    )
    source = functional_launch(
        lambda: source_module.recurrent_gated_delta_rule(
            query,
            key,
            value,
            gate,
            beta,
            initial_state=None,
            output_final_state=True,
            tuning_configs=configs,
            compiler_timeout=context.compiler_timeout_seconds,
        )
    )
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=1e-1, rtol=5e-2),
            Tolerance(atol=1e-1, rtol=5e-2),
        ),
        cuda_graph=False,
    )


def chunk_gated_delta(context: Context) -> PreparedComparison:
    batch, sequence, heads, key_dimension, value_dimension = 2, 2048, 8, 128, 128
    query = torch.randn(
        (batch, sequence, heads, key_dimension),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn(
        (batch, sequence, heads, value_dimension),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    gate = -torch.rand(
        (batch, sequence, heads), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    scale = 1.0 / math.sqrt(key_dimension)
    _, prepare = compile_single(
        context,
        chunk_gated_delta_prepare,
        (query, key, value, gate, beta, scale),
    )
    intermediate_outputs = prepare.outputs()
    _, recurrence = compile_single(
        context,
        chunk_gated_delta_recurrence,
        intermediate_outputs,
    )

    def generated_launch():
        prepare.launch()
        recurrence.launch()

    generated = PreparedLaunch(generated_launch, recurrence.outputs)
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/scan/gated_delta_chunk/chunk_gated_delta_rule.py",
        "chunk_gated_delta",
        needs_utils=True,
    )
    source = functional_launch(
        lambda: source_module.chunk_gated_delta_rule(
            query,
            key,
            value,
            gate,
            beta,
            chunk_size=64,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
    )
    tolerance = Tolerance(atol=1e-1, rtol=5e-2)
    return PreparedComparison(
        generated,
        source,
        (tolerance, tolerance),
        cuda_graph=False,
        note="同算法；中间精度和舍入不同",
    )


CASES = {
    "recurrent_gated_delta": recurrent_gated_delta,
    "chunk_gated_delta": chunk_gated_delta,
}
