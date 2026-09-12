import intent
import torch

from kernels.optimization.adamw import adamw_update
from kernels.optimization.adafactor import adafactor_update_rows, adafactor_update_columns, adafactor_apply
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


def adafactor(context):
    gradient = torch.randn((4096, 4096), dtype=torch.float32) * 0.01
    initial = (torch.randn_like(gradient), torch.rand((4096,)) * 0.01,
               torch.rand((4096,)) * 0.01, torch.zeros((1,)))
    decay, learning_rate, epsilon = 0.8, 1e-2, 1e-8
    report_stage("generated_compilation")
    rows, columns, apply = (
        intent.compile(definition, target=context.target, compiler=context.compiler,
                       tuning_config=context.tuning_config)
        for definition in (adafactor_update_rows, adafactor_update_columns, adafactor_apply)
    )
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def pipeline(gradient, parameter, row_state, column_state, row_mean, decay, learning_rate, epsilon):
        rows.run(gradient, row_state, row_mean, decay, 1.0 / 4096, 1.0 / 4096)
        columns.run(gradient, column_state, decay, 1.0 / 4096)
        apply.run(gradient, row_state, column_state, row_mean, parameter, learning_rate, epsilon)

    def side(function):
        state = tuple(value.clone() for value in initial)

        def prepare():
            for destination, source in zip(state, initial):
                destination.copy_(source)

        return PreparedLaunch(lambda: function(gradient, *state, decay, learning_rate, epsilon),
                              lambda: state, prepare=prepare)

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(pipeline), side(runtime.adafactor),
        (Tolerance(atol=2e-5), Tolerance(atol=2e-7), Tolerance(atol=2e-7), Tolerance(atol=2e-7)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有4096x4096 f32 Adafactor三kernel完整调用，含原子row mean和parameter/row/column状态；单NUMA8核，PyTorch CPU reference；每次恢复初始状态且恢复不计时。",
    )


CASES = {"flaggems_fused_adamw": adamw, "adafactor": adafactor}
