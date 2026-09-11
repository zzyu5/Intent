from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import sys

from intent.runtime.weft import Buffer, NativeProgram, compile_artifact


def serve(directory: Path) -> None:
    configuration = json.loads((directory / "deployment.json").read_text())
    with ThreadPoolExecutor(max_workers=2) as executor:
        builds = [executor.submit(compile_artifact, directory / side,
                  cc=tuple(configuration["cc"]), cflags=tuple(configuration["cflags"]))
                  for side in ("generated", "source")]
        for build in builds:
            build.result()
    os.sched_setaffinity(0, configuration["cpus"])
    generated = NativeProgram(directory / "generated")
    source = NativeProgram(directory / "source")
    try:
        rng = random.Random(0)
        shape = (8, 128, 32)
        arguments = tuple(Buffer(array("f", (rng.gauss(0.0, 1.0) for _ in range(8 * 128 * 32))),
                                 shape=shape, dtype="f32") for _ in range(3))
        if configuration["kernel"] == "causal_attention_f32":
            arguments += (32 ** -0.5,)
        generated_call = generated.prepare(arguments)
        source_call = source.prepare(arguments)
        print(json.dumps({"ready": True}), flush=True)
        if sys.stdin.readline() != "benchmark\n":
            raise RuntimeError("benchmark coordinator disconnected")
        generated_call.choose()
        source_call.choose()
        generated_call.launch()
        source_call.launch()
        generated_ms = generated_call.benchmark()
        source_ms = source_call.benchmark()

        def flattened(result):
            outputs = result if isinstance(result, tuple) else (result,)
            return [number for output in outputs for number in output.storage.cast("f")]

        print(json.dumps({
            "generated_ms": generated_ms, "source_ms": source_ms,
            "generated": flattened(generated_call.result()),
            "source": flattened(source_call.result()),
            "winner": generated.candidates[generated_call.winner],
        }), flush=True)
    finally:
        generated.close()
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    serve(parser.parse_args().directory)
