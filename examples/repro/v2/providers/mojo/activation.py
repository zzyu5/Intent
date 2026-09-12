import torch

from kernels.activation.pointwise import addcmul_broadcast_bf16, gelu_tanh, relu_forward
from kernels.activation.swiglu import swiglu_forward

from ...model import Tolerance
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


CASES = {"gelu": gelu, "relu": relu, "flaggems_addcmul": addcmul, "swiglu": swiglu}
