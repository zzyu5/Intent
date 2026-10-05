from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import torch

import intent
from intent.runtime.bangc import export_artifact
from experiments._common.loading import load_module
from experiments._common.measurement import PipelineStageError, report_stage
from experiments._common.model import NativeComparisonResult, PreparedComparison


DTYPES = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32",
          torch.int32: "i32", torch.int64: "i64", torch.bool: "bool"}
TORCH_DTYPES = {value: key for key, value in DTYPES.items()}


def _tree(function, value):
    return tuple(_tree(function, item) for item in value) if isinstance(value, tuple) else function(value)


def tilegym_source(context, path, name, *, needs_utils=False):
    runtime = load_module(context.project_root / "experiments/gpu/baselines/cutile/tilegym/support/runtime.py",
        "intent_bangc_tilegym_runtime")
    return runtime.load_source(context.project_root / path,
        "tilegym.ops.cutile._intent_bangc_" + name, needs_utils=needs_utils)


class RemoteSequence:
    """Transport an author-supplied kernel sequence for the existing production runner."""

    def __init__(self, context):
        self.context = context
        self.steps = []
        self.allocated_outputs = set()

    def add(self, definition, arguments, *, constexprs=None, target=None):
        views = {name: value for name, value in arguments.items() if isinstance(value, torch.Tensor)}
        target = replace(target or self.context.target,
            shapes={name: tuple(value.shape) for name, value in views.items()},
            strides={name: tuple(value.stride()) for name, value in views.items()})
        report_stage("generated_compilation")
        program = intent.generate(definition, target=target, compiler=self.context.compiler,
            constexprs=constexprs, tuning_config=self.context.tuning_config,
            options=self.context.compile_options)
        bound = dict(arguments)
        interface = program.interface
        dimensions = {}
        for parameter in interface.views:
            value = bound.get(parameter.name)
            if value is not None:
                for identity, extent in zip(parameter.dimensions, value.shape):
                    if identity > 0:
                        dimensions[identity] = extent
        outputs = {}
        for parameter in interface.parameters:
            name = parameter.name
            if isinstance(parameter, intent.ViewParameter) and parameter.output and name not in bound:
                shape = tuple(fixed if fixed >= 0 else dimensions[identity]
                    for fixed, identity in zip(parameter.shape, parameter.dimensions))
                bound[name] = torch.empty(shape, dtype=TORCH_DTYPES[parameter.dtype.name], device="cpu")
                storage = bound[name].untyped_storage()
                self.allocated_outputs.add((str(bound[name].device), storage.data_ptr(), storage.nbytes()))
            if name not in bound:
                raise ValueError(f"missing BANG C invocation argument {name}")
            if isinstance(parameter, intent.ViewParameter) and parameter.writable:
                outputs[name] = bound[name]
        self.steps.append((program, bound, interface))
        report_stage("adapter_preparation")
        return outputs

    def comparison(self, outputs, reference, tolerance, *, source=None):
        host = os.environ["INTENT_BANGC_HOST"]
        remote_root = os.environ["INTENT_BANGC_ROOT"]
        python = os.environ["INTENT_BANGC_PYTHON"]
        path = Path(tempfile.mkdtemp(prefix="intent-bangc-"))
        remote = remote_root.rstrip("/") + "/cases/" + path.name
        manifest = {"buffers": {}, "steps": [], "outputs": []}
        storage_ids = {}

        def describe(value):
            if not isinstance(value, torch.Tensor):
                return value
            dtype = DTYPES[value.dtype]
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key not in storage_ids:
                name = f"buffer{len(storage_ids)}"
                storage_ids[key] = (name, dtype)
                manifest["buffers"][name] = {"dtype": dtype,
                    "shape": [storage.nbytes() // value.element_size()]}
                if key not in self.allocated_outputs:
                    data = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (storage.nbytes(),), (1,))
                    data.cpu().numpy().tofile(path / (name + ".bin"))
                    manifest["buffers"][name]["file"] = name + ".bin"
            name, storage_dtype = storage_ids[key]
            if storage_dtype != dtype:
                raise NotImplementedError("BANG C benchmark storage aliases must preserve dtype")
            return {"buffer": name, "shape": list(value.shape), "strides": list(value.stride()),
                    "offset": value.storage_offset()}

        for index, (program, arguments, interface) in enumerate(self.steps):
            artifact = f"kernel{index}"
            export_artifact(program, path / artifact)
            manifest["steps"].append({"artifact": artifact,
                "arguments": [describe(arguments[parameter.name]) for parameter in interface.parameters]})

        def output_spec(value):
            specification = describe(value)
            index = len(manifest["outputs"])
            manifest["outputs"].append(specification)
            return index

        output_tree = _tree(output_spec, outputs)
        if source is not None:
            operation, inputs = source
            manifest["source"] = {"operation": operation, "inputs": [describe(value) for value in inputs]}
            shutil.copytree(self.context.project_root / "experiments/mlu/baselines/cnnl", path / "source")
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        report_stage("source_launch")
        expected = _tree(lambda value: value.detach().cpu(), reference())
        report_stage("remote_preparation")

        def execute(command, stage):
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise PipelineStageError(stage, result.stderr[-6000:] or result.stdout[-6000:])
            return result

        execute(["ssh", host, shlex.join(["mkdir", "-p", remote_root.rstrip("/") + "/cases"])], "remote_preparation")
        execute(["scp", "-q", "-r", str(path), host + ":" + remote_root.rstrip("/") + "/cases/"], "remote_preparation")

        def measure():
            environment = ["env", "PYTHONPATH=" + remote_root.rstrip("/") + "/python", python]
            arguments = [remote + "/manifest.json",
                "--outputs", remote + "/outputs", "--device", str(self.context.target.device),
                "--neuware", str(self.context.target.neuware)]
            command = [*environment, "-m", "intent.runtime.bangc.benchmark", *arguments]
            execute(["ssh", host, shlex.join(command)], "generated_launch")
            if source is not None:
                command = [*environment, remote + "/source/runtime.py", *arguments]
                execute(["ssh", host, shlex.join(command)], "source_native_execution")
            execute(["scp", "-q", "-r", host + ":" + remote + "/outputs", str(path)], "generated_result")
            measured = json.loads((path / "outputs/result.json").read_text())
            owners = {}
            for name, specification in measured["buffers"].items():
                owners[name] = torch.frombuffer(bytearray((path / "outputs" / specification["file"]).read_bytes()),
                    dtype=TORCH_DTYPES[specification["dtype"]])
            tensors = [owners[item["buffer"]].as_strided(tuple(item["shape"]), tuple(item["strides"]), item["offset"])
                       for item in measured["outputs"]]
            generated = _tree(lambda index: tensors[index], output_tree)
            if source is not None:
                source_result = json.loads((path / "outputs/source-result.json").read_text())
                specification = source_result["source_output"]
                native = torch.frombuffer(bytearray((path / "outputs" / specification["file"]).read_bytes()),
                    dtype=TORCH_DTYPES[specification["dtype"]]).reshape(specification["shape"])
                return NativeComparisonResult(measured["generated_ms"], source_result["source_ms"],
                    (generated, native), (expected, expected))
            return NativeComparisonResult(measured["generated_ms"], None, generated, expected)

        return PreparedComparison(None, None, (tolerance, tolerance) if source is not None else tolerance,
            cuda_graph=False, device_type="cpu",
            native_comparison=measure,
            note="MLU CNRT notifier timing; supplied tensor shapes and element strides specialize the compiled variant; "
                 + (f"CNNL same-device {source[0]} source; both results checked against the original numerical reference; "
                    + {"softmax": "accurate last-axis softmax; ", "relu": "NaN-propagating ReLU; ",
                       "matmul": "f16 inputs/output, f32 accumulation, alpha=1/beta=0, NN; "}[source[0]] +
                    "descriptor setup, workspace allocation and host transfers excluded" if source is not None else
                    "original native source is a numerical reference; no cross-device latency ratio"))
