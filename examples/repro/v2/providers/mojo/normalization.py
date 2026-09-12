import torch
from kernels.normalization.rms_norm import weighted_rms_norm, rms_norm_f32, rms_norm_bf16
from kernels.normalization.softmax import stable_softmax, stable_softmax_f16, chunked_softmax_bf16
from kernels.normalization.layer_norm import weighted_layer_norm, layer_norm_f16, layer_norm_bf16
from kernels.normalization.logsumexp import row_logsumexp
from kernels.backward.softmax import softmax_backward as backward_definition
from ...model import Tolerance
from .common import configure_cpu_budget, prepare_comparison, prepare_host_comparison


def rms_norm(context):
    x = torch.randn((8192, 4096), dtype=torch.float32)
    weight = torch.randn((4096,), dtype=torch.float32)
    return prepare_comparison(context, weighted_rms_norm, (x, weight, 1.0 / 4096, 1e-6),
        "source/mojo/modular/normalization/rms_norm/rms_norm_runtime.py", Tolerance(5e-5),
        "Source 调用安装的 Modular/MAX RMSNorm；双方归约括号与中间数学实现可能不同。")


def softmax(context):
    x = torch.randn((8192, 8192), dtype=torch.float32)
    return prepare_comparison(context, stable_softmax, (x,),
        "source/mojo/modular/normalization/softmax/softmax_runtime.py", Tolerance(1e-6, 1e-5),
        "Source 调用安装的 Modular/MAX CPU Softmax；双方 max/exp/sum 后显式以倒数乘法归一化，归约树仍可能不同。")


def layer_norm(context):
    x = torch.randn((8192, 4096), dtype=torch.float32)
    weight = torch.randn((4096,), dtype=torch.float32)
    bias = torch.randn((4096,), dtype=torch.float32)
    return prepare_comparison(context, weighted_layer_norm, (x, weight, bias, 1.0 / 4096, 1e-6),
        "source/mojo/modular/normalization/layer_norm/layer_norm_runtime.py", Tolerance(5e-5, 1e-5),
        "Source 使用 Modular/MAX rowwise 同矩统计算法 E[x²]-E[x]²；不是现成 Welford layer_norm。")


def logsumexp(context):
    configure_cpu_budget()
    x = torch.randn((8192, 8192), dtype=torch.float32)
    return prepare_host_comparison(context, row_logsumexp, (x,), "logsumexp", Tolerance(atol=1e-4, rtol=1e-5))


def half_softmax(context):
    configure_cpu_budget()
    x = torch.randn((8192, 8192), dtype=torch.float16)
    return prepare_host_comparison(context, stable_softmax_f16, (x,), "softmax", Tolerance(1e-2, 1e-2))


def bfloat_softmax(context):
    configure_cpu_budget()
    x = torch.randn((8192, 32768), dtype=torch.bfloat16)
    return prepare_host_comparison(context, chunked_softmax_bf16, (x,), "softmax", Tolerance(2e-2, 1e-2))


def mixed_layer_norm(context, definition, dtype, tolerance):
    configure_cpu_budget()
    x = torch.randn((8192, 4096), dtype=dtype)
    weight = torch.randn((4096,), dtype=dtype)
    bias = torch.randn((4096,), dtype=dtype)
    return prepare_host_comparison(context, definition, (x, weight, bias, 1.0 / 4096, 1e-5),
                                   "layer_norm", tolerance)


def plain_rms_norm(context):
    configure_cpu_budget()
    x = torch.randn((8192, 4096), dtype=torch.float32)
    return prepare_host_comparison(context, rms_norm_f32, (x, 1.0 / 4096, 1e-5),
                                   "rms_norm", Tolerance(1e-2, 1e-2))


def bfloat_rms_norm(context):
    configure_cpu_budget()
    x = torch.randn((8192, 4096), dtype=torch.bfloat16)
    weight = torch.randn((4096,), dtype=torch.bfloat16)
    return prepare_host_comparison(context, rms_norm_bf16, (x, weight, 1.0 / 4096, 1e-6),
                                   "weighted_rms_norm", Tolerance(2e-2, 1e-2))


def softmax_backward(context):
    configure_cpu_budget()
    probabilities = torch.randn((4096, 4097), dtype=torch.float32)
    upstream = torch.randn_like(probabilities)
    return prepare_host_comparison(context, backward_definition, (probabilities, upstream),
                                   "softmax_backward", Tolerance(1e-4, 1e-5))


CASES = {"weighted_rms_norm": rms_norm, "stable_softmax": softmax, "weighted_layer_norm": layer_norm,
         "flaggems_logsumexp": logsumexp, "fused_softmax": half_softmax, "chunked_softmax": bfloat_softmax,
         "layer_norm_f16": lambda context: mixed_layer_norm(context, layer_norm_f16, torch.float16, Tolerance(1e-2)),
         "layer_norm_bf16": lambda context: mixed_layer_norm(context, layer_norm_bf16, torch.bfloat16, Tolerance(2e-2, 1e-2)),
         "rms_norm_f32": plain_rms_norm, "rms_norm_bf16": bfloat_rms_norm,
         "flaggems_softmax_backward": softmax_backward}
