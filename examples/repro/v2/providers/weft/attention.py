from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import torch
import intent
from intent.runtime.weft import TargetProfile, export_artifact
from intent.runtime.weft.compilation import flattened_signature, lower_artifact
from kernels.streaming.attention_f32 import causal_attention_f32, causal_linear_attention_f32

from ...loading import load_module
from ...measurement import PipelineStageError, report_stage
from ...model import NativeComparisonResult, PreparedComparison, Tolerance


SOURCE = "source/weft/intentdsl/attention"


def compile_source(context, profile, compiler, linear):
    import weft
    family = "linear" if linear else "causal"
    module = load_module(context.project_root / SOURCE / family / f"{family}.py", f"weft_source_{family}_attention")
    definition = module.causal_linear_attention if linear else module.causal_attention
    canonical = weft.lower_to_mlir(definition)
    artifact = lower_artifact(canonical, compiler=compiler, profile=profile)
    return canonical, artifact


def source_artifact(directory, profile, metadata, linear, canonical, artifact):
    kernel, = artifact["kernels"]
    signature, _ = flattened_signature(metadata["parameters"])
    dimensions = {"B": "a0_d0", "S": "a0_d1", "TQ": "a0_d1", "TK": "a1_d1",
                  "D": "a0_d2", "DV": "a2_d2"}
    shape_arguments = [dimensions[name] for name in kernel["shape_parameters"]]
    pointer_types = [argument["c_type"] for argument in kernel["arguments"]]
    pointers = ["a0", "a1", "a2", "a3", "a4" if linear else "&a4"]
    host = (
        "#include <stdint.h>\n#include <stddef.h>\n"
        f"extern void {kernel['symbol']}({', '.join([*pointer_types, *(['size_t'] * len(shape_arguments))])});\n"
        f"void source_attention({', '.join(signature)}) {{\n"
        f"  {kernel['symbol']}({', '.join([*pointers, *shape_arguments])});\n}}\n"
    )
    source_metadata = {**metadata, "host_source": host,
        "tasks": [{"abi": {key: kernel[key] for key in ("symbol", "arguments", "shape_parameters")}}],
        "candidates": [{"entry": "source_attention", "values": [], "implementations": []}]}
    directory.mkdir()
    (directory / "canonical.mlir").write_text(canonical)
    (directory / "host.c").write_text(host)
    (directory / "kernels.c").write_text(artifact["intrinsic_c"])
    (directory / "artifact.json").write_text(json.dumps({
        "profile": asdict(profile), "program": source_metadata, "weft": artifact,
    }))


def prepare(context, linear):
    deployment_path = Path(os.environ.get("INTENT_WEFT_PROFILE", Path(__file__).with_name("rvv.json")))
    deployment = json.loads(deployment_path.read_text())
    deployment["kernel"] = "causal_linear_attention_f32" if linear else "causal_attention_f32"
    profile = TargetProfile(deployment["march"], deployment["abi"], deployment["vlen_bits"], tuple(deployment["cpus"]))
    compiler = os.environ["INTENT_WEFT_COMPILER"]
    directory = tempfile.TemporaryDirectory(prefix="intentdsl-weft-attention-")
    root = Path(directory.name)
    report_stage("source_compilation")
    with ThreadPoolExecutor(max_workers=2) as executor:
        generated = executor.submit(intent.generate,
            causal_linear_attention_f32 if linear else causal_attention_f32,
            target=context.target, compiler=context.compiler, tuning_config=context.tuning_config)
        source = executor.submit(compile_source, context, profile, compiler, linear)
        canonical, artifact = source.result()
        report_stage("generated_compilation")
        program = generated.result()
    source_artifact(root / "source", profile, program.metadata, linear, canonical, artifact)
    report_stage("generated_weft_compilation")
    export_artifact(program, root / "generated", compiler=compiler, profile=profile)
    shutil.copytree(context.project_root / "python/intent", root / "python/intent",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(context.project_root / SOURCE / "attention_runtime.py", root / "runtime.py")
    (root / "deployment.json").write_text(json.dumps(deployment))
    report_stage("native_deployment")
    host = deployment["host"]
    remote = subprocess.run(["ssh", host, "mktemp -d /tmp/intentdsl-weft-attention.XXXXXX"],
                            check=True, text=True, capture_output=True).stdout.strip()
    subprocess.run(["scp", "-qr", *(str(path) for path in sorted(root.iterdir())), f"{host}:{remote}/"], check=True)
    command = shlex.join(["env", f"PYTHONPATH={remote}/python", "python3", f"{remote}/runtime.py", remote])
    process = subprocess.Popen(["ssh", host, command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    ready = process.stdout.readline()
    if not ready:
        process.wait()
        raise PipelineStageError("native_preparation", f"remote native preparation exited {process.returncode}")
    if json.loads(ready) != {"ready": True}:
        raise RuntimeError("invalid native preparation response")

    def measure():
        try:
            process.stdin.write("benchmark\n")
            process.stdin.flush()
            output = process.stdout.readline()
            process.wait()
            if process.returncode:
                raise PipelineStageError("native_benchmark", f"remote invocation exited {process.returncode}")
            result = json.loads(output)
            print(f"weft: selected {result['winner']}", flush=True)
            return NativeComparisonResult(result["generated_ms"], result["source_ms"],
                torch.tensor(result["generated"], dtype=torch.float32), torch.tensor(result["source"], dtype=torch.float32))
        finally:
            process.stdin.close()
            process.stdout.close()
            directory.cleanup()

    return PreparedComparison(
        generated=None, source=None, tolerance=Tolerance(2e-4, 1e-5), cuda_graph=False,
        device_type="cpu", native_comparison=measure,
        note="单核 RVV；独立 Weft 同算法 source；完整 native invocation 含内部 workspace、状态与任务完成；不含外部输出分配。",
    )


CASES = {"causal_attention_f32": lambda context: prepare(context, False),
         "causal_linear_attention_f32": lambda context: prepare(context, True)}
