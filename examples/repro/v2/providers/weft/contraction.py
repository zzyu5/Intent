from __future__ import annotations

from dataclasses import asdict
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
from kernels.quantization.quantized_projection import quantized_projection

from ...loading import load_module
from ...measurement import NumericalComparisonError, PipelineStageError, report_stage
from ...model import NativeComparisonResult, PreparedComparison, Tolerance


SOURCE = "source/weft/tianchenrv/contraction/q4_k_projection"


def _source_artifact(context, directory: Path, profile: TargetProfile, compiler: str, metadata: dict) -> None:
    import weft
    module = load_module(context.project_root / SOURCE / "q4_k_projection.py", "weft_source_projection")
    canonical = weft.lower_to_mlir(module.production_mul_mat_q4_k)
    artifact = lower_artifact(canonical, compiler=compiler, profile=profile)
    kernel, = artifact["kernels"]
    symbol = kernel["symbol"]
    signature, _ = flattened_signature(metadata["parameters"])
    dimensions = {"N": "a0_d0", "K": "a1_d0 * 256", "M": "1"}
    shape_arguments = [dimensions[name] for name in kernel["shape_parameters"]]
    pointer_types = [argument["c_type"] for argument in kernel["arguments"]]
    host = (
        "#include <stdint.h>\n#include <stddef.h>\n#include <stdlib.h>\n"
        f"extern void {symbol}({', '.join([*pointer_types, *(['size_t'] * len(shape_arguments))])});\n"
        f"void source_projection({', '.join(signature)}) {{\n"
        "  size_t bytes = a1_d0 * 292;\n"
        "  void *quantized = aligned_alloc(16, (bytes + 15) / 16 * 16);\n"
        "  if (!quantized && bytes) abort();\n"
        f"  {symbol}({', '.join(['a0', 'a1', 'quantized', 'a2', *shape_arguments])});\n"
        "  free(quantized);\n}\n"
    )
    source_metadata = {**metadata, "host_source": host,
                       "tasks": [{"abi": {key: kernel[key] for key in ("symbol", "arguments", "shape_parameters")}}],
                       "candidates": [{"entry": "source_projection", "values": [], "implementations": []}]}
    directory.mkdir()
    (directory / "canonical.mlir").write_text(canonical)
    (directory / "host.c").write_text(host)
    (directory / "kernels.c").write_text(artifact["intrinsic_c"])
    (directory / "artifact.json").write_text(json.dumps({
        "profile": asdict(profile), "program": source_metadata, "weft": artifact,
    }))


def projection(context):
    deployment_path = Path(os.environ.get("INTENT_WEFT_PROFILE", Path(__file__).with_name("rvv.json")))
    deployment = json.loads(deployment_path.read_text())
    profile = TargetProfile(deployment["march"], deployment["abi"], deployment["vlen_bits"], tuple(deployment["cpus"]))
    compiler = os.environ["INTENT_WEFT_COMPILER"]
    directory = tempfile.TemporaryDirectory(prefix="intentdsl-weft-benchmark-")
    root = Path(directory.name)
    report_stage("generated_compilation")
    program = intent.generate(quantized_projection, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config)
    report_stage("source_compilation")
    _source_artifact(context, root / "source", profile, compiler, program.metadata)
    report_stage("generated_weft_compilation")
    export_artifact(program, root / "generated", compiler=compiler, profile=profile)
    shutil.copytree(context.project_root / "python/intent", root / "python/intent",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(context.project_root / SOURCE / "q4_k_projection_runtime.py", root / "runtime.py")
    (root / "deployment.json").write_text(json.dumps(deployment))
    report_stage("native_deployment")
    host = deployment["host"]
    remote = subprocess.run(["ssh", host, "mktemp -d /tmp/intentdsl-weft-benchmark.XXXXXX"],
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
            generated = torch.tensor(result["generated"], dtype=torch.float32)
            source = torch.tensor(result["source"], dtype=torch.float32)
            if not torch.isfinite(generated).all() or not torch.isfinite(source).all():
                raise NumericalComparisonError("Q4_K projection requires finite output values")
            print(f"weft: selected {result['winner']}", flush=True)
            return NativeComparisonResult(result["generated_ms"], result["source_ms"], generated, source)
        finally:
            process.stdin.close()
            process.stdout.close()
            directory.cleanup()

    return PreparedComparison(
        generated=None, source=None, tolerance=Tolerance(1e-4, 2e-3), cuda_graph=False,
        device_type="cpu", native_comparison=measure,
        note="单核 RVV；完整 native invocation 含 Q8_K 量化及内部 workspace 分配/释放；同算法、相同 cold-cache，10 次中位数。",
    )


CASES = {"q4_k_projection": projection}
