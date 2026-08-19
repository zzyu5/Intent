from __future__ import annotations

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
    _, generated = compile_single(
        context,
        xor_shift_dropout,
        (
            x,
            source_module._mix_seed(seed),
            probability,
            1.0 / (1.0 - probability),
        ),
    )
    source = functional_launch(
        lambda: source_module.dropout(
            x,
            seed,
            p=probability,
            training=True,
            inplace=False,
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=0.0),
        cuda_graph=True,
    )


CASES = {"dropout": dropout}
