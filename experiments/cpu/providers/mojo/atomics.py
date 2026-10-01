import intent
import torch

from kernels.backward.embedding import embedding_backward_atomic
from kernels.synchronization.compare_exchange import claim_zero_slots

from experiments._common.loading import load_module
from experiments._common.measurement import report_stage
from experiments._common.model import PreparedComparison, PreparedLaunch, Tolerance


def selected_outputs(artifact, outputs):
    program = artifact.runtime
    for key, winner in program.winners.items():
        print(f"mojo: selected {program.candidates[winner]}; candidate_ms={program.timings[key]}", flush=True)
    return outputs


def embedding_backward(context):
    tokens, vocabulary, features = 32768, 8192, 1021
    token_ids = torch.arange(tokens, dtype=torch.int32)
    indices = (token_ids * 17) % vocabulary
    grad_output = torch.randn((tokens, features), dtype=torch.float32) * 1.0e-3
    initial_weight = torch.zeros((vocabulary, features), dtype=torch.float32)

    report_stage("generated_compilation")
    artifact = intent.compile(
        embedding_backward_atomic,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    generated_weight = initial_weight.clone()

    def restore_generated():
        generated_weight.copy_(initial_weight)

    generated = PreparedLaunch(
        lambda: artifact.run(indices, grad_output, generated_weight),
        lambda: selected_outputs(artifact, generated_weight),
        prepare=restore_generated,
    )

    runtime = load_module(
        context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py",
        "intent_cpu_reference_embedding_backward_atomic",
    )
    source_weight = initial_weight.clone()

    def restore_source():
        source_weight.copy_(initial_weight)

    source = PreparedLaunch(
        lambda: runtime.embedding_backward_atomic(indices, grad_output, source_weight),
        lambda: source_weight,
        prepare=restore_source,
    )
    report_stage("adapter_preparation")
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2.0e-6, rtol=0.0),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有 embedding backward atomic M32768/V8192/D1021 f32 InOut；PyTorch index_add_ reference，单 NUMA 8 核；每次调用前恢复 zero weight，恢复不计时；双方计完整 host 调用。",
    )


def compare_exchange(context):
    slots = 65536
    initial = torch.where(
        torch.arange(slots, dtype=torch.int32) % 3 == 0,
        torch.zeros((), dtype=torch.int32),
        torch.full((), 2, dtype=torch.int32),
    )

    report_stage("generated_compilation")
    artifact = intent.compile(
        claim_zero_slots,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    generated_state = initial.clone()
    generated_outputs = {"previous": None}

    def restore_generated():
        generated_state.copy_(initial)

    def generated_launch():
        _, generated_outputs["previous"] = artifact.run(generated_state)

    generated = PreparedLaunch(
        generated_launch,
        lambda: selected_outputs(artifact, (generated_state, generated_outputs["previous"])),
        prepare=restore_generated,
    )

    runtime = load_module(
        context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py",
        "intent_cpu_reference_claim_zero_slots",
    )
    source_state = initial.clone()
    source_outputs = {"previous": None}

    def restore_source():
        source_state.copy_(initial)

    def source_launch():
        _, source_outputs["previous"] = runtime.claim_zero_slots(source_state)

    source = PreparedLaunch(
        source_launch,
        lambda: (source_state, source_outputs["previous"]),
        prepare=restore_source,
    )
    report_stage("adapter_preparation")
    return PreparedComparison(
        generated,
        source,
        (Tolerance(atol=0.0), Tolerance(atol=0.0)),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有 claim_zero_slots i32 N65536 InOut/Out；每次调用前恢复 state，恢复不计时；双方计完整 host 调用。",
    )


CASES = {"embedding_backward_atomic": embedding_backward,
         "atomic_compare_exchange": compare_exchange}
