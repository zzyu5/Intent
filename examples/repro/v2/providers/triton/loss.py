from __future__ import annotations

import torch

from kernels.loss.cross_entropy import fused_cross_entropy_bf16

from ...loading import load_module
from ...measurement import compile_single
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


def cross_entropy(context: Context) -> PreparedComparison:
    tokens, vocabulary = 8192, 32768
    logits = torch.randn(
        (tokens, vocabulary), device="cuda", dtype=torch.bfloat16
    )
    labels = torch.randint(
        0,
        vocabulary,
        (tokens,),
        device="cuda",
        dtype=torch.int64,
    )
    labels[::17] = -100
    generated_logits = logits.clone()
    _, generated_base = compile_single(
        context,
        fused_cross_entropy_bf16,
        (generated_logits, labels),
        constexprs={"IGNORE_INDEX": -100},
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: (*generated_base.outputs(), generated_logits),
        prepare=lambda: generated_logits.copy_(logits),
    )
    runtime = load_module(
        context.project_root
        / "source/triton/liger-kernel/loss/cross_entropy/cross_entropy_runtime.py",
        "intent_v2_triton_cross_entropy",
    )
    source_logits = logits.clone()
    state: dict[str, object] = {}

    def source_launch():
        state["outputs"] = runtime.upstream((source_logits, labels))

    source_launch()
    source = PreparedLaunch(
        launch=source_launch,
        outputs=lambda: state["outputs"],
        prepare=lambda: source_logits.copy_(logits),
    )
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=5e-2, rtol=1e-2),
            Tolerance(atol=0.0),
            Tolerance(atol=5e-2, rtol=5e-2),
        ),
        cuda_graph=False,
    )


CASES = {
    "cross_entropy": cross_entropy,
}

