from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import sys
import threading

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
        m, n, k = configuration["shape"]
        arguments = tuple(Buffer(bytearray((directory / f"{name}.bin").read_bytes()), shape=shape, dtype=dtype)
                          for name, shape, dtype in (("a", (m, k), "i8"), ("b", (k, n), "i8"), ("bias", (n,), "i32")))
        generated_call, source_call = generated.prepare(arguments), source.prepare(arguments)
        eviction = Buffer.empty((64 * 1024 * 1024,), "u8")
        evict = generated.library.intent_weft_evict
        evict.argtypes, evict.restype = [ctypes.c_void_p, ctypes.c_int64], None
        prepare = lambda: evict(eviction.pointer, eviction.nbytes)
        print(json.dumps({"ready": True}), flush=True)
        if sys.stdin.readline() != "benchmark\n":
            raise RuntimeError("benchmark coordinator disconnected")
        def monitor_coordinator():
            os.read(sys.stdin.fileno(), 1)
            os._exit(1)
        threading.Thread(target=monitor_coordinator, daemon=True).start()
        generated_call.choose(prepare)
        source_call.choose(prepare)
        generated_call.launch()
        source_call.launch()
        generated_ms = generated_call.benchmark(prepare=prepare)
        source_ms = source_call.benchmark(prepare=prepare)
        print(json.dumps({
            "generated_ms": generated_ms, "source_ms": source_ms,
            "generated": list(generated_call.result().storage.cast("i")),
            "source": list(source_call.result().storage.cast("i")),
            "winner": generated.candidates[generated_call.winner].metadata(),
            "used_extensions": sorted(generated.candidate_extensions[generated_call.winner]),
        }), flush=True)
    finally:
        generated.close()
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    serve(parser.parse_args().directory)
