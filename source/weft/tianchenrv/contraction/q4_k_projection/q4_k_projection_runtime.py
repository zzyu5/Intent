from __future__ import annotations

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import random
import struct
import sys

from intent.runtime.weft import Buffer, NativeProgram, compile_artifact


def inputs(n: int, k: int) -> tuple[Buffer, Buffer]:
    rng = random.Random(0)
    groups = k // 256
    records = bytearray(rng.randbytes(n * groups * 144))
    for offset in range(0, len(records), 144):
        struct.pack_into("<ee", records, offset, rng.uniform(0.001, 0.004), rng.uniform(0.002, 0.008))
    activation = array("f", (rng.uniform(-1.0, 1.0) for _ in range(k)))
    return Buffer(records, shape=(n, groups, 144), dtype="u8"), Buffer(activation, shape=(groups, 256), dtype="f32")


def serve(directory: Path) -> None:
    configuration = json.loads((directory / "deployment.json").read_text())
    os.sched_setaffinity(0, configuration["cpus"])
    with ThreadPoolExecutor(max_workers=2) as executor:
        builds = [executor.submit(compile_artifact, directory / side,
                  cc=tuple(configuration["cc"]), cflags=tuple(configuration["cflags"]))
                  for side in ("generated", "source")]
        for build in builds:
            build.result()
    generated = NativeProgram(directory / "generated")
    source = NativeProgram(directory / "source")
    try:
        arguments = inputs(4096, 4096)
        generated_call = generated.prepare(arguments)
        source_call = source.prepare(arguments)
        cache = bytearray(64 * 1024 * 1024)
        evict = generated.library.intent_weft_evict
        evict.argtypes, evict.restype = [ctypes.c_void_p, ctypes.c_int64], None
        cache_pointer = ctypes.addressof(ctypes.c_char.from_buffer(cache))

        def prepare():
            evict(cache_pointer, len(cache))

        print(json.dumps({"ready": True}), flush=True)
        if sys.stdin.readline() != "benchmark\n":
            raise RuntimeError("benchmark coordinator disconnected")
        generated_call.choose(prepare)
        source_call.choose(prepare)
        generated_call.launch()
        source_call.launch()
        generated_ms = generated_call.benchmark(prepare=prepare)
        source_ms = source_call.benchmark(prepare=prepare)
        print(json.dumps({
            "generated_ms": generated_ms, "source_ms": source_ms,
            "generated": list(generated_call.result().storage.cast("f")),
            "source": list(source_call.result().storage.cast("f")),
            "winner": generated.candidates[generated_call.winner],
        }), flush=True)
    finally:
        generated.close()
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    serve(parser.parse_args().directory)
