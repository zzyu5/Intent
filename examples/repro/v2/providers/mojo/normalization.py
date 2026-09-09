import torch
from kernels.normalization.rms_norm import weighted_rms_norm
from kernels.normalization.softmax import stable_softmax
from kernels.normalization.layer_norm import weighted_layer_norm
from ...model import Tolerance
from .common import prepare_comparison


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


CASES = {"weighted_rms_norm": rms_norm, "stable_softmax": softmax, "weighted_layer_norm": layer_norm}
