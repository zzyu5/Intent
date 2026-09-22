# Independently authored f32 online attention, using row tasks and SIMD dot products.
from max.algorithm import parallelize
from std.ffi import external_call
from std.math import exp, min
from std.memory import Layout, alloc, dealloc, unsafe_stack_allocation
from std.runtime import initialize_runtime
from std.sys import llvm_intrinsic


@always_inline
def dot(a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin], count: Int) -> Float32:
    var acc = SIMD[DType.float32, 8](0)
    var full = count // 8 * 8
    for i in range(0, full, 8):
        acc += a.unsafe_offset(i).unsafe_load[width=8]() * b.unsafe_offset(i).unsafe_load[width=8]()
    var result = acc.reduce_add()
    for i in range(full, count):
        result += a.unsafe_offset(i).unsafe_load() * b.unsafe_offset(i).unsafe_load()
    return result


@export("source_causal_attention")
@no_inline
def source_causal_attention(
    q: Pointer[Float32, MutUntrackedOrigin], qb: Int64, qq: Int64, qd: Int64, qs0: Int64, qs1: Int64, qs2: Int64,
    k: Pointer[Float32, MutUntrackedOrigin], kb: Int64, kk: Int64, kd: Int64, ks0: Int64, ks1: Int64, ks2: Int64,
    v: Pointer[Float32, MutUntrackedOrigin], vb: Int64, vk: Int64, vd: Int64, vs0: Int64, vs1: Int64, vs2: Int64,
    output: Pointer[Float32, MutUntrackedOrigin], ob: Int64, oq: Int64, od: Int64, os0: Int64, os1: Int64, os2: Int64,
    scale: Float32,
) abi("C"):
    initialize_runtime()
    var environment = external_call["intent_cpu_enter_ieee", UInt32]()
    var queries = Int(qq)
    var keys = Int(kk)
    var depth = Int(qd)
    var width = Int(vd)

    def row(task: Int) {imm q, imm k, imm v, imm output, imm scale, imm queries, imm keys, imm depth, imm width}:
        var saved = external_call["intent_cpu_enter_ieee", UInt32]()
        var batch = task // queries
        var query = task % queries
        var storage = alloc(Layout[Float32](count=width))
        var acc = storage.unsafe_ptr()
        var scores = unsafe_stack_allocation[64, Float32]()
        for d in range(width):
            acc.unsafe_offset(d).unsafe_store(Float32(0))
        var maximum = Float32(0)
        var denominator = Float32(0)
        var valid = False
        for begin in range(0, keys, 64):
            if begin > query:
                break
            var count = min(64, min(keys, query + 1) - begin)
            var local_max = Float32(-1) / Float32(0)
            for j in range(count):
                var score = dot(q.unsafe_offset((batch * queries + query) * depth), k.unsafe_offset((batch * keys + begin + j) * depth), depth) * scale
                scores.unsafe_offset(j).unsafe_store(score)
                local_max = llvm_intrinsic["llvm.maximum", Float32](local_max, score)
            var updated = llvm_intrinsic["llvm.maximum", Float32](maximum, local_max) if valid else local_max
            var rescale = exp(maximum - updated) if valid else Float32(0)
            denominator *= rescale
            for d in range(width):
                acc.unsafe_offset(d).unsafe_store(acc.unsafe_offset(d).unsafe_load() * rescale)
            for j in range(count):
                var probability = exp(scores.unsafe_offset(j).unsafe_load() - updated)
                denominator += probability
                var full = width // 8 * 8
                for d in range(0, full, 8):
                    var value = v.unsafe_offset((batch * keys + begin + j) * width + d).unsafe_load[width=8]()
                    var previous = acc.unsafe_offset(d).unsafe_load[width=8]()
                    acc.unsafe_offset(d).unsafe_store(previous + SIMD[DType.float32, 8](probability) * value)
                for d in range(full, width):
                    var value = v.unsafe_offset((batch * keys + begin + j) * width + d).unsafe_load()
                    acc.unsafe_offset(d).unsafe_store(acc.unsafe_offset(d).unsafe_load() + probability * value)
            maximum = updated
            valid = True
        for d in range(width):
            var result = acc.unsafe_offset(d).unsafe_load() / denominator if valid else Float32(0)
            output.unsafe_offset((batch * queries + query) * width + d).unsafe_store(result)
        dealloc(storage^)
        external_call["intent_cpu_leave_ieee", NoneType](saved)

    parallelize(row, Int(qb) * queries, 8)
    external_call["intent_cpu_leave_ieee", NoneType](environment)
