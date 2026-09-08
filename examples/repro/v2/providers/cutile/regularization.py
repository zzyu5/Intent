from __future__ import annotations

from types import SimpleNamespace

import torch

from kernels.regularization.dropout import xor_shift_dropout

from ...measurement import compile_single
from ...measurement import functional_launch
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
    _, generated = compile_single(
        context,
        xor_shift_dropout,
        (
            x,
            mixed_seed,
            probability,
            inverse_keep,
        ),
    )
    # The source counter is the absolute flattened element offset, independent
    # of both source tile boundaries and generated row ownership.
    configs = tuple(
        SimpleNamespace(TILE_SIZE=1 << exponent, ACCESS_FORM=form)
        for exponent in range(5, 15)
        for form in (1, 2, 3)
    )
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
