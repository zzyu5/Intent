from __future__ import annotations

from types import SimpleNamespace

import torch

from intent.runtime.artifact import ParameterRole
from kernels.regularization.dropout import xor_shift_dropout

from ...measurement import PipelineStageError
from ...measurement import compile_single
from ...measurement import functional_launch
from ...measurement import report_stage
from ...model import Context
from ...model import PreparedComparison
from ...model import Tolerance
from .common import tilegym_source


def dropout(context: Context) -> PreparedComparison:
    x = torch.randn((8192, 4096), device="cuda", dtype=torch.float16)
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/regularization/dropout/dropout.py",
        "dropout",
    )
    seed = 12345
    probability = 0.1
    mixed_seed = source_module._mix_seed(seed)
    inverse_keep = 1.0 / (1.0 - probability)
    artifact, generated = compile_single(
        context,
        xor_shift_dropout,
        (
            x,
            mixed_seed,
            probability,
            inverse_keep,
        ),
    )
    report_stage("generated_tuning_metadata")
    try:
        configurations = artifact.tuning_configurations(
            x, generated.outputs(), mixed_seed, probability, inverse_keep,
        )
    except NotImplementedError as error:
        raise PipelineStageError("generated_tuning_metadata", str(error)) from error
    report_stage("source_candidate_binding")
    configs = []
    for configuration in configurations:
        values = {}
        for parameter, value in zip(configuration.parameters, configuration.values, strict=True):
            if parameter.role == ParameterRole.PROVIDER_ACCESS_FORM:
                field = "ACCESS_FORM"
            elif (parameter.role in (ParameterRole.OWNERSHIP_M, ParameterRole.OWNERSHIP_N)
                  and parameter.view_axis == (0, 1)):
                field = "TILE_SIZE"
            else:
                raise PipelineStageError("source_candidate_binding",
                                         f"source cannot bind dropout parameter {parameter}")
            if field in values:
                raise PipelineStageError("source_candidate_binding", f"duplicate dropout field {field}")
            values[field] = value
        if values.keys() != {"TILE_SIZE", "ACCESS_FORM"}:
            raise PipelineStageError("source_candidate_binding", "incomplete dropout candidate")
        if x.shape[1] % values["TILE_SIZE"] != 0:
            raise PipelineStageError("source_candidate_binding",
                                     "source flat tiles must preserve generated row ownership")
        configs.append(SimpleNamespace(**values))
    configs = tuple(configs)
    report_stage("adapter_preparation")
    source = functional_launch(
        lambda: source_module.dropout(
            x,
            seed,
            p=probability,
            training=True,
            inplace=False,
            tuning_configs=configs,
            compiler_timeout=context.compiler_timeout_seconds,
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=0.0),
        cuda_graph=True,
    )


CASES = {"dropout": dropout}
