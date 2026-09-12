import intent
import torch

from kernels.loss.cross_entropy import fused_cross_entropy
from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance


def cross_entropy(context):
    initial = torch.randn((4096, 16384), dtype=torch.float32) * 0.5
    labels = torch.randint(0, 16384, (4096,), dtype=torch.int32)
    labels[::17] = -100
    report_stage("generated_compilation")
    artifact = intent.compile(fused_cross_entropy, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, constexprs={"IGNORE_INDEX": -100})
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        logits = initial.clone()
        state = {}

        def launch():
            state["outputs"] = function(logits, labels)

        return PreparedLaunch(launch, lambda: state["outputs"], prepare=lambda: logits.copy_(initial))

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.fused_cross_entropy),
        (Tolerance(atol=2e-5), Tolerance(atol=2e-4), Tolerance(atol=0.0)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有4096x16384 f32 fused cross entropy，含原地logits gradient、loss和prediction；原ignore_index -100；单NUMA8核，PyTorch CPU reference，完整host调用；每次恢复logits且恢复不计时。",
    )


CASES = {"fused_cross_entropy": cross_entropy}
