from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from intent.runtime.bangc.buffer import DeviceBuffer
from intent.runtime.bangc.program import benchmark_calls, launch_calls


class CNNLCall:
    """A source callable using the same CNRT queue/notifier timing as generated code."""

    def __init__(self, library, operation, source, output, rows, columns):
        self.program = self
        self.library = library
        self.source, self.output = source, output
        self.runtime = source.runtime
        self.target = SimpleNamespace(device=source.device)
        self.queue, self.context = ctypes.c_void_p(), ctypes.c_void_p()
        self.runtime.select(source.device)
        self.runtime.invoke("cnrtQueueCreate", ctypes.byref(self.queue))
        try:
            self.check(library.source_create(ctypes.byref(self.context), self.queue,
                {"relu": 0, "softmax": 1}[operation], rows, columns))
        except Exception:
            self.close()
            raise

    def check(self, status):
        if status:
            raise RuntimeError(f"CNNL source failed: {self.library.cnnlGetErrorString(status).decode()}")

    def check_open(self):
        if not self.context.value:
            raise ValueError("CNNL source is closed")

    def ensure_loaded(self):
        self.check_open()

    def ensure_queue(self):
        self.check_open()
        return self.queue

    def _validate(self):
        self.check_open()
        if not self.source.pointer or not self.output.pointer:
            raise ValueError("CNNL source refers to a closed device allocation")
        self.runtime.select(self.target.device)

    def _submit(self, queue):
        if queue.value != self.queue.value:
            raise ValueError("CNNL source must use its bound CNRT queue")
        self.check(self.library.source_run(self.context, self.source.pointer, self.output.pointer))

    def enqueue(self, queue):
        self._validate()
        self._submit(queue)

    def close(self):
        if self.queue.value:
            self.runtime.invoke("cnrtQueueSync", self.queue)
        if self.context.value:
            self.library.source_destroy(self.context)
            self.context = ctypes.c_void_p()
        if self.queue.value:
            self.runtime.invoke("cnrtQueueDestroy", self.queue)
            self.queue = ctypes.c_void_p()


def load_source(root: Path, neuware: str):
    sdk = Path(neuware)
    library = root / "cnnl-source.so"
    subprocess.run(["c++", "-std=c++17", "-O3", "-shared", "-fPIC",
        str(Path(__file__).with_name("operators.cpp")), "-I" + str(sdk / "include"),
        "-L" + str(sdk / "lib64"), "-Wl,-rpath," + str(sdk / "lib64"),
        "-lcnnl", "-lcnrt", "-o", str(library)], check=True)
    result = ctypes.CDLL(str(library))
    pointer = ctypes.c_void_p
    result.source_create.argtypes = [ctypes.POINTER(pointer), pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    result.source_create.restype = ctypes.c_int
    result.source_run.argtypes = [pointer, pointer, pointer]
    result.source_run.restype = ctypes.c_int
    result.source_destroy.argtypes = [pointer]
    result.source_destroy.restype = None
    result.cnnlGetErrorString.argtypes = [ctypes.c_int]
    result.cnnlGetErrorString.restype = ctypes.c_char_p
    return result


def run(manifest: Path, output: Path, *, device: int, neuware: str):
    request = json.loads(manifest.read_text())
    specification = request["source"]
    view = specification["input"]
    rows, columns = view["shape"]
    buffer = request["buffers"][view["buffer"]]
    if buffer["dtype"] != "f16" or view["strides"] != [columns, 1] or view["offset"]:
        raise NotImplementedError("CNNL production baselines require contiguous rank-2 f16 inputs")
    library = load_source(manifest.parent, neuware)
    output.mkdir(parents=True, exist_ok=True)
    source = DeviceBuffer((rows, columns), "f16", device=device, neuware=neuware)
    result = None
    call = None
    try:
        source.copy_from_host((manifest.parent / buffer["file"]).read_bytes())
        result = DeviceBuffer((rows, columns), "f16", device=device, neuware=neuware)
        call = CNNLCall(library, specification["operation"], source, result, rows, columns)
        launch_calls((call,))
        (output / "source.bin").write_bytes(result.to_host())
        milliseconds = benchmark_calls((call,))
        (output / "source-result.json").write_text(json.dumps({
            "source_ms": milliseconds,
            "source_output": {"file": "source.bin", "dtype": "f16", "shape": [rows, columns]},
        }, indent=2))
    finally:
        if call is not None:
            call.close()
        if result is not None:
            result.close()
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--neuware", required=True)
    arguments = parser.parse_args()
    run(arguments.manifest, arguments.outputs, device=arguments.device, neuware=arguments.neuware)
