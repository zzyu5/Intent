from __future__ import annotations

import argparse
import json
from pathlib import Path

from intent.targets.bangc import BangCTarget
from . import DeviceBuffer, NativeProgram


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and run an exported Intent BANG C artifact")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("inputs", type=Path, help="JSON array of scalar values or {file, shape, dtype} buffer descriptions")
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--neuware", default="/usr/local/neuware")
    args = parser.parse_args()
    metadata = json.loads((args.artifact / "artifact.json").read_text())
    target = BangCTarget(**{name: metadata[name] for name in
        ("architecture", "tile", "tile_m", "tile_n", "tile_k", "tasks", "local_bytes")},
        device=args.device, neuware=args.neuware).resolve()
    program = NativeProgram((args.artifact / "kernel.mlu").read_text(), metadata, target)
    buffers = []
    try:
        arguments = []
        for description in json.loads(args.inputs.read_text()):
            if isinstance(description, dict):
                buffer = DeviceBuffer.from_host((args.inputs.parent / description["file"]).read_bytes(),
                    shape=tuple(description.get("storage_shape", description["shape"])),
                    dtype=description["dtype"], device=args.device, neuware=args.neuware)
                buffers.append(buffer)
                if "strides" in description:
                    arguments.append(buffer.view(tuple(description["shape"]), tuple(description["strides"]),
                                                 offset=description.get("offset", 0)))
                else:
                    arguments.append(buffer)
            else:
                arguments.append(description)
        call = program.prepare(tuple(arguments))
        buffers.extend(call.outputs)
        call.launch()
        args.outputs.mkdir(parents=True, exist_ok=True)
        results = []
        # InOut buffers remain caller-owned, but the file interface exports
        # their post-call contents alongside explicitly produced Out values.
        for parameter, value in zip(program.parameters, call.arguments):
            if parameter["kind"] != "view" or parameter["access"] == 0:
                continue
            filename = f"output{len(results)}.bin"
            (args.outputs / filename).write_bytes(value.to_host())
            results.append({"name": parameter["name"], "file": filename,
                            "shape": value.shape, "dtype": value.dtype})
        (args.outputs / "outputs.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"status": "executed", "outputs": str(args.outputs / "outputs.json")}))
    finally:
        program.close()
        for buffer in buffers:
            buffer.close()


if __name__ == "__main__":
    main()
