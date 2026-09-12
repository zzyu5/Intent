import intent
import torch

from kernels.optimization.adamw import adamw_update
from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance


def adamw(context):
    gradient = torch.randn((8 * 1024 * 1024,), dtype=torch.float32)
    initial = (torch.randn_like(gradient), torch.randn_like(gradient), torch.rand_like(gradient) + 0.5)
    scalars = (1e-3, 0.9, 0.999, 1.0 - 0.9**10, 1.0 - 0.999**10, 1e-8, 0.01)
    report_stage("generated_compilation")
    artifact = intent.compile(adamw_update, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config)
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        state = tuple(value.clone() for value in initial)

        def prepare():
            for destination, source in zip(state, initial):
                destination.copy_(source)

        return PreparedLaunch(lambda: function(gradient, *state, *scalars), lambda: state, prepare=prepare)

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.adamw), Tolerance(atol=1e-5, rtol=1e-5),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有 AdamW N8388608 f32；PyTorch eager CPU reference，单 NUMA 8 核；每次调用前恢复相同的 parameter/moments，恢复不计时；双方计完整 host 调用。",
    )


CASES = {"flaggems_fused_adamw": adamw}
