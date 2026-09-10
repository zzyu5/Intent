from __future__ import annotations

import ast
from contextlib import redirect_stdout
from dataclasses import dataclass
import inspect
import io
from itertools import islice
import json
from pathlib import Path
import re

import torch

from repro.v2.model import Tolerance


SUITE_PATH = Path(__file__).with_name("suite.json")


def read_suite() -> dict:
    return json.loads(SUITE_PATH.read_text())


def catalog(root: Path, suite: dict) -> list[dict]:
    data_path = Path("data/TritonBench_T_comp_alpac_v1.json")
    prompts = json.loads((root / data_path).read_text())
    selected = {task["id"]: task for task in suite["tasks"]}
    source_dir = Path("data/TritonBench_T_v1")
    entries = {}
    for path in sorted((root / source_dir).glob("*.py")):
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and not node.name.startswith("test_")]
        for function in functions:
            entries[function.name] = (path.stem, function.lineno)
    rows = []
    for index, item in enumerate(prompts):
        match = re.search(r"Wrapper Entry Information:\s*(?:def\s+)?([\w.]+)\s*\(", item["instruction"])
        if match is None:
            raise ValueError(f"prompt {index} has no unambiguous wrapper entry")
        entry = match[1].rsplit(".", 1)[-1]
        if str(index) in suite["prompt_entry_overrides"]:
            entry = suite["prompt_entry_overrides"][str(index)]["entry"]
        task_id, line = entries[entry]
        disposition, reason = "not_selected", "Outside the fixed first 50; not a compiler-support judgment."
        if task_id in suite["outside_scope"]:
            disposition, reason = "out_of_scope", suite["outside_scope"][task_id]
        elif task_id in suite["deferred_contracts"]:
            disposition, reason = "contract_pending", suite["deferred_contracts"][task_id]
        elif "dropout" in task_id:
            disposition, reason = "out_of_scope", "Training-time RNG contract is not closed; do not silently disable dropout."
        elif task_id in selected:
            disposition, reason = "selected", selected[task_id].get("note", "Fixed deterministic forward performance profile.")
        rows.append({"task": task_id, "entry": entry, "prompt_index": index,
                     "prompt_file": str(data_path), "reference_file": str(source_dir / f"{task_id}.py"),
                     "reference_line": line, "profile_file": f"performance_metrics/perf_T/golden_metrics/{task_id}_perf.py",
                     "disposition": disposition, "reason": reason})
    if len(rows) != 166 or len({row["task"] for row in rows}) != 166:
        raise ValueError("TritonBench-T task correspondence is not the agreed 166 unique tasks")
    if len(selected) != 50 or set(selected) - {row["task"] for row in rows}:
        raise ValueError("the first study must contain exactly 50 distinct mapped tasks")
    return rows


def description(root: Path, row: dict) -> str:
    instruction = json.loads((root / row["prompt_file"]).read_text())[row["prompt_index"]]["instruction"]
    # The surrounding upstream role text is Triton-specific; the task body is shared.
    start = instruction.index("Functional Description:")
    end = instruction.index("After generation, verify")
    return instruction[start:end].strip()


def reference(root: Path, row: dict):
    path = root / row["reference_file"]
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                 or isinstance(node, ast.FunctionDef) and not node.name.startswith("test_")]
    namespace = {"__name__": f"tritonbench_reference_{row['task']}"}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[row["entry"]]


class _ProfileBase:
    def __init__(self, name, *, dtype, is_backward=False, **kwargs):
        self.dtype = dtype
        self.is_backward = is_backward


class _StreamInputs(ast.NodeTransformer):
    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            if ast.unparse(call.func) == "self.input_tensors.append":
                return ast.copy_location(ast.Expr(ast.Yield(call.args[0])), node)
        return self.generic_visit(node)


class _InvocationCaptured(Exception):
    def __init__(self, args, kwargs):
        self.args_value = args
        self.kwargs_value = kwargs


def _capture(*args, **kwargs):
    raise _InvocationCaptured(args, kwargs)


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


@dataclass
class Invocation:
    args: tuple
    kwargs: dict
    bound: dict

    def call(self, function):
        return function(*self.args, **self.kwargs)

    def to_device(self, device) -> Invocation:
        copies = {}

        def move(value):
            if isinstance(value, torch.Tensor):
                if id(value) not in copies:
                    copies[id(value)] = value.to(device)
                return copies[id(value)]
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, list):
                return [move(item) for item in value]
            return value
        return Invocation(move(self.args), {name: move(value) for name, value in self.kwargs.items()},
                          {name: move(value) for name, value in self.bound.items()})

    def metadata(self) -> dict:
        def describe(value):
            if isinstance(value, torch.Tensor):
                return {"shape": list(value.shape), "dtype": str(value.dtype),
                        "stride": list(value.stride()), "device": "cuda"}
            if isinstance(value, (tuple, list)):
                return [describe(item) for item in value]
            if isinstance(value, torch.dtype):
                return str(value)
            return value
        return {name: describe(value) for name, value in self.bound.items()}


def invocation(root: Path, row: dict, task: dict, suite: dict, *, device="cuda") -> Invocation:
    path = root / row["profile_file"]
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.bases = [ast.Name("_ProfileBase", ast.Load())]
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in {"__init__", "get_input_tensors", "call_op"}]
    cls = _StreamInputs().visit(cls)
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"torch": torch, "_ProfileBase": _ProfileBase, row["entry"]: _capture}
    exec(compile(module, str(path), "exec"), namespace)
    profile = namespace[cls.name](dtype=getattr(torch, suite["dtype"]))
    torch.manual_seed(suite["input_seed"])
    with redirect_stdout(io.StringIO()):
        inputs = next(islice(profile.get_input_tensors(), task["input_index"], None))
    inputs = _to_device(inputs, device)
    try:
        profile.call_op(inputs)
    except _InvocationCaptured as captured:
        function = reference(root, row)
        bound = inspect.signature(function).bind(*captured.args_value, **captured.kwargs_value)
        bound.apply_defaults()
        return Invocation(captured.args_value, captured.kwargs_value, dict(bound.arguments))
    raise ValueError(f"{row['task']} profile did not call its declared task entry")


def tolerance(task: dict, suite: dict) -> Tolerance:
    return Tolerance(**suite["tolerances"][task["tolerance"]])
