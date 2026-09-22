from experiments.gpu.registry import BY_PROVIDER as GPU
from experiments.cpu.registry import BY_PROVIDER as CPU
from experiments.mlu.registry import BY_PROVIDER as MLU


BY_PROVIDER = {**GPU, **CPU, **MLU}
PROVIDER_GROUPS = {
    **dict.fromkeys(GPU, "gpu"),
    **dict.fromkeys(CPU, "cpu"),
    **dict.fromkeys(MLU, "mlu"),
}
