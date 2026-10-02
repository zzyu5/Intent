import intent
import torch

from kernels.activation.pointwise import (
    addcmul_broadcast_bf16,
    geglu_tanh,
    gelu_tanh,
    relu_forward,
)
from kernels.activation.swiglu import FEATURES as SWIGLU_FEATURES
from kernels.activation.swiglu import silu_and_mul_packed
from kernels.activation.swiglu import swiglu_forward
from kernels.backward.swiglu import swiglu_backward

from experiments._common.loading import load_module
from experiments._common.measurement import report_stage
from experiments._common.model import PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget, prepare_host_comparison


def gelu(context):
    configure_cpu_budget()
    x = torch.randn((8192, 4096), dtype=torch.float16)
    return prepare_host_comparison(context, gelu_tanh, (x,), "gelu", Tolerance(atol=1e-2, rtol=1e-2))


def relu(context):
    configure_cpu_budget()
    x = torch.randn((8192, 4096), dtype=torch.float16)
    return prepare_host_comparison(context, relu_forward, (x,), "relu", Tolerance(atol=0.0))


def addcmul(context):
    configure_cpu_budget()
    shape = (64, 128, 4096)
    x = torch.randn(shape, dtype=torch.bfloat16)
    scale = torch.randn(shape[:2], dtype=torch.bfloat16)
    bias = torch.randn(shape[:2], dtype=torch.bfloat16)
    return prepare_host_comparison(context, addcmul_broadcast_bf16, (x, scale, bias),
                                   "addcmul", Tolerance(atol=2e-2, rtol=1e-2))


def swiglu(context):
    gate = torch.randn((8192, 14336), dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    return prepare_host_comparison(context, swiglu_forward, (gate, up), "swiglu",
                                   Tolerance(atol=2e-2, rtol=1e-2))


def swiglu_backward_case(context):
    configure_cpu_budget()
    shape = (4096, 4096)
    dc = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    a = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    b = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    return prepare_host_comparison(
        context,
        swiglu_backward,
        (dc, a, b),
        "swiglu_backward",
        (Tolerance(atol=5e-2), Tolerance(atol=5e-2)),
    )


def silu_and_mul(context):
    packed = torch.randn(
        (4096, 2 * SWIGLU_FEATURES), dtype=torch.bfloat16
    )
    return prepare_host_comparison(
        context,
        silu_and_mul_packed,
        (packed,),
        "silu_and_mul_packed",
        Tolerance(atol=2e-2, rtol=1e-2),
    )


def geglu(context):
    x = torch.randn((4096, 2 * 14336), dtype=torch.float16)
    report_stage("generated_compilation")
    artifact = intent.compile(geglu_tanh, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, options=context.compile_options)
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        output = torch.empty((4096, 14336), dtype=torch.float16)
        return PreparedLaunch(lambda: function(x, output), lambda: output)

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.geglu_tanh), Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有4096x28672 f16 GEGLU，独立InOut输出，完整host调用；单NUMA8核，PyTorch CPU reference。",
    )


CASES = {"gelu": gelu, "relu": relu, "flaggems_addcmul": addcmul, "swiglu": swiglu,
         "swiglu_backward": swiglu_backward_case,
         "silu_and_mul_packed": silu_and_mul, "geglu_tanh": geglu}
