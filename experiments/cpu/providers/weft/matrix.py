from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import torch
import intent
from intent.runtime.weft import TargetProfile, export_artifact
from intent.runtime.weft.compilation import flattened_signature, lower_artifact
from intent.runtime.native import NativeABI
from kernels.contraction.gemm import gemm_i8

from experiments._common.loading import load_module
from experiments._common.measurement import PipelineStageError, report_stage
from experiments._common.model import NativeComparisonResult, PreparedComparison, Tolerance


SOURCE = "experiments/cpu/baselines/weft/intentdsl/contraction/i8"


def compile_source(context, profile, compiler, rows):
    import weft
    module = load_module(context.project_root / SOURCE / "i8.py", "weft_source_i8_matrix")
    canonical = weft.lower_to_mlir(module.gemm_bias)
    return canonical, lower_artifact(canonical, compiler=compiler, profile=profile,
                                     source_bindings=(("MR", 1 if rows == 1 else 4),))


def source_artifact(directory, profile, metadata, canonical, artifact):
    kernel, = artifact["kernels"]
    signature, _ = flattened_signature(NativeABI.read(metadata))
    dimensions = {"M": "a0_d0", "K": "a0_d1", "N": "a1_d1"}
    shape_arguments = [dimensions[name] for name in kernel["shape_parameters"]]
    pointer_types = [argument["c_type"] for argument in kernel["arguments"]]
    host = (
        "#include <stdint.h>\n#include <stddef.h>\n"
        f"extern void {kernel['symbol']}({', '.join([*pointer_types, *(['size_t'] * len(shape_arguments))])});\n"
        f"void source_matrix({', '.join(signature)}) {{\n"
        f"  {kernel['symbol']}({', '.join(['a0', 'a1', 'a2', 'a3', *shape_arguments])});\n}}\n"
    )
    source_metadata = {**metadata, "host_source": host,
        "tasks": [{"cpu_entry": "source_matrix", "abi": {key: kernel[key] for key in ("symbol", "arguments", "shape_parameters")}}],
        "candidates": [{"entry": "source_matrix", "values": [], "implementations": [],
                        "requires_matrix_i8_i32": False}]}
    directory.mkdir(exist_ok=True)
    (directory / "canonical.mlir").write_text(canonical)
    (directory / "host.c").write_text(host)
    (directory / "kernels.c").write_text(artifact["intrinsic_c"])
    (directory / "artifact.json").write_text(json.dumps({
        "profile": asdict(profile), "program": source_metadata, "weft": artifact,
    }))


def prepare(context, rows):
    deployment_path = Path(os.environ.get("INTENT_WEFT_PROFILE", Path(__file__).with_name("ime.json")))
    deployment = json.loads(deployment_path.read_text())
    profile = TargetProfile.from_deployment(deployment)
    compiler = os.environ["INTENT_WEFT_COMPILER"]
    root = Path.home() / ".cache/intentdsl/benchmarks" / context.project_root.name / f"i8_m{rows}"
    root.mkdir(parents=True, exist_ok=True)
    report_stage("generated_compilation")
    with ThreadPoolExecutor(max_workers=2) as executor:
        generated = executor.submit(intent.generate, gemm_i8, target=context.target,
                                    compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
        source = executor.submit(compile_source, context, profile, compiler, rows)
        program = generated.result()
        canonical, artifact = source.result()
    source_artifact(root / "source", profile, program.metadata, canonical, artifact)
    report_stage("generated_weft_compilation")
    export_artifact(program, root / "generated", compiler=compiler, profile=profile)
    torch.set_num_threads(8)
    generator = torch.Generator().manual_seed(0)
    a = torch.randint(-128, 128, (rows, 4096), dtype=torch.int8, generator=generator)
    b = torch.randint(-128, 128, (4096, 4096), dtype=torch.int8, generator=generator)
    bias = torch.randint(-1024, 1024, (4096,), dtype=torch.int32, generator=generator)
    # Every product and partial sum is an exactly representable f64 integer.
    reference = (a.to(torch.float64) @ b.to(torch.float64) + bias).to(torch.int32)
    for name, value in (("a", a), ("b", b), ("bias", bias)):
        (root / f"{name}.bin").write_bytes(value.numpy().tobytes())
    shutil.copytree(context.project_root / "python/intent", root / "python/intent", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(context.project_root / SOURCE / "i8_runtime.py", root / "runtime.py")
    deployment["shape"] = [rows, 4096, 4096]
    (root / "deployment.json").write_text(json.dumps(deployment))
    report_stage("native_deployment")
    host = deployment["host"]
    remote = subprocess.run(["ssh", host, "mktemp -d /tmp/intentdsl-weft-matrix.XXXXXX"],
                            check=True, text=True, capture_output=True).stdout.strip()
    subprocess.run(["scp", "-qr", *(str(path) for path in sorted(root.iterdir())), f"{host}:{remote}/"], check=True)
    command = shlex.join(["env", f"PYTHONPATH={remote}/python", "python3", f"{remote}/runtime.py", remote])
    process = subprocess.Popen(["ssh", host, command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    ready = process.stdout.readline()
    if not ready:
        process.wait()
        raise PipelineStageError("native_preparation", f"native preparation exited {process.returncode}; artifacts: {root}")
    if json.loads(ready) != {"ready": True}:
        raise RuntimeError("invalid native preparation response")

    def measure():
        try:
            process.stdin.write("benchmark\n")
            process.stdin.flush()
            output = process.stdout.readline()
            process.wait()
            if process.returncode:
                raise PipelineStageError("native_benchmark", f"native invocation exited {process.returncode}")
            result = json.loads(output)
            generated = torch.tensor(result["generated"], dtype=torch.int32).reshape(rows, 4096)
            source = torch.tensor(result["source"], dtype=torch.int32).reshape(rows, 4096)
            print(f"weft: selected {result['winner']}; extensions={result['used_extensions']}", flush=True)
            return NativeComparisonResult(result["generated_ms"], result["source_ms"],
                                          (generated, source), (reference, reference))
        finally:
            process.stdin.close()
            process.stdout.close()

    return PreparedComparison(generated=None, source=None, tolerance=Tolerance(0, 0), cuda_graph=False,
        device_type="cpu", native_comparison=measure,
        note="K1/X60 CPU3 VLEN256 IME1；i8×i8→i32+bias，同次两侧对精确整数 reference；完整 native 调用含内部分配与 packing，外部输出分配不计时；64 MiB eviction，10 次中位数。")


CASES = {"i8_gemv_bias": lambda context: prepare(context, 1),
         "i8_gemm_bias": lambda context: prepare(context, 128)}
