from __future__ import annotations

import argparse
import json
from pathlib import Path

from intent.targets.bangc import BangCTarget
from .buffer import DeviceBuffer
from .program import NativeProgram, benchmark_calls, launch_calls


def run(manifest: Path, output: Path, *, device: int, neuware: str) -> None:
    request = json.loads(manifest.read_text())
    root = manifest.parent
    buffers, programs, initial = {}, [], {}
    calls = []
    reset = set()
    try:
        for name, specification in request["buffers"].items():
            buffers[name] = DeviceBuffer(tuple(specification["shape"]), specification["dtype"],
                device=device, neuware=neuware)
            if "file" in specification:
                data = (root / specification["file"]).read_bytes()
                buffers[name].copy_from_host(data)
                initial[name] = data

        def argument(specification):
            if not isinstance(specification, dict):
                return specification
            return buffers[specification["buffer"]].view(tuple(specification["shape"]),
                tuple(specification["strides"]), offset=specification["offset"])

        for step in request["steps"]:
            artifact = root / step["artifact"]
            metadata = json.loads((artifact / "artifact.json").read_text())
            target = BangCTarget(**{name: metadata[name] for name in
                ("architecture", "tile", "tile_m", "tile_n", "tile_k", "region_tile", "tasks", "local_bytes")},
                device=device, neuware=neuware).resolve()
            program = NativeProgram((artifact / "kernel.mlu").read_text(), metadata, target)
            programs.append(program)
            calls.append(program.prepare(tuple(argument(value) for value in step["arguments"]), explicit_outputs=True))
            for parameter, specification in zip(program.parameters, step["arguments"]):
                if parameter["kind"] == "view" and parameter["access"] == 2:
                    reset.add(specification["buffer"])
        sequence = tuple(calls)
        launch_calls(sequence)
        output.mkdir(parents=True, exist_ok=True)
        owners = {}
        for specification in request["outputs"]:
            name = specification["buffer"]
            if name not in owners:
                filename = name + ".bin"
                (output / filename).write_bytes(buffers[name].to_host())
                owners[name] = {"file": filename, "dtype": buffers[name].dtype}

        def prepare():
            for name in reset:
                if name in initial:
                    buffers[name].copy_from_host(initial[name])

        milliseconds = benchmark_calls(sequence, prepare=prepare)
        (output / "result.json").write_text(json.dumps({
            "generated_ms": milliseconds, "timing": "cnrt_notifier_duration",
            "buffers": owners, "outputs": request["outputs"],
        }, indent=2))
        print(json.dumps({"status": "executed", "generated_ms": milliseconds}))
    finally:
        for program in programs:
            program.close()
        for buffer in buffers.values():
            buffer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and measure an explicit sequence of exported BANG C kernels")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--neuware", default="/usr/local/neuware")
    args = parser.parse_args()
    run(args.manifest, args.outputs, device=args.device, neuware=args.neuware)


if __name__ == "__main__":
    main()
