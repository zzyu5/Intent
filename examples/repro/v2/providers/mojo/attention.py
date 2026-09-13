import torch

from kernels.streaming.attention_f32 import causal_attention_f32, causal_linear_attention_f32
from kernels.streaming.attention import flash_attention_bf16_fwd
from ...model import Tolerance
from .common import prepare_comparison
from .common import prepare_host_comparison


def causal_attention(context):
    q = torch.randn((8, 128, 32), dtype=torch.float32)
    k, v = torch.randn_like(q), torch.randn_like(q)
    return prepare_comparison(context, causal_attention_f32, (q, k, v, 32 ** -0.5),
        "source/mojo/intentdsl/attention/causal/causal_runtime.py", Tolerance(2e-4, 1e-5),
        "独立 Mojo online attention source；沿用 CPU f32 contraction 容差；分段与归约括号可不同。")


def causal_linear_attention(context):
    q = torch.randn((8, 128, 32), dtype=torch.float32)
    k, v = torch.randn_like(q), torch.randn_like(q)
    return prepare_comparison(context, causal_linear_attention_f32, (q, k, v),
        "source/mojo/intentdsl/attention/linear/linear_runtime.py", Tolerance(2e-4, 1e-5),
        "独立 Mojo chunked linear attention source；输出包含最终状态；沿用 CPU f32 contraction 容差。")


def flash_attention_bf16(context):
    dimension = 128
    q = torch.randn((2, 32, 4096, dimension), dtype=torch.bfloat16)
    k = torch.randn((2, 8, 4096, dimension), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    return prepare_host_comparison(
        context,
        flash_attention_bf16_fwd,
        (q, k, v, dimension**-0.5),
        "flash_attention_bf16_fwd",
        Tolerance(atol=5e-2, rtol=2e-2),
        constexprs={"HEAD_GROUP": 4, "CAUSAL": True},
    )


CASES = {
    "causal_attention_f32": causal_attention,
    "causal_linear_attention_f32": causal_linear_attention,
    "flash_attention_bf16": flash_attention_bf16,
}
