# Independently authored chunked linear attention with explicit incoming state.
from max.algorithm import parallelize
from std.ffi import external_call
from std.math import min
from std.memory import Layout, alloc, dealloc
from std.runtime import initialize_runtime


@export("source_linear_attention")
@no_inline
def source_linear_attention(
    q: Pointer[Float32, MutUntrackedOrigin], qb: Int64, qs: Int64, qd: Int64, qs0: Int64, qs1: Int64, qs2: Int64,
    k: Pointer[Float32, MutUntrackedOrigin], kb: Int64, ks: Int64, kd: Int64, ks0: Int64, ks1: Int64, ks2: Int64,
    v: Pointer[Float32, MutUntrackedOrigin], vb: Int64, vs: Int64, vd: Int64, vs0: Int64, vs1: Int64, vs2: Int64,
    output: Pointer[Float32, MutUntrackedOrigin], ob: Int64, os: Int64, od: Int64, os0: Int64, os1: Int64, os2: Int64,
    final_state: Pointer[Float32, MutUntrackedOrigin], fb: Int64, fk: Int64, fv: Int64, fs0: Int64, fs1: Int64, fs2: Int64,
) abi("C"):
    initialize_runtime()
    var environment = external_call["intent_cpu_enter_ieee", UInt32]()
    var sequence = Int(qs)
    var depth = Int(qd)
    var width = Int(vd)

    def batch_body(batch: Int) {imm q, imm k, imm v, imm output, imm final_state, imm sequence, imm depth, imm width}:
        var saved = external_call["intent_cpu_enter_ieee", UInt32]()
        var state_storage = alloc(Layout[Float32](count=depth * width))
        var state = state_storage.unsafe_ptr()
        var score_storage = alloc(Layout[Float32](count=64 * 64))
        var scores = score_storage.unsafe_ptr()
        for i in range(depth * width):
            state.unsafe_offset(i).unsafe_store(Float32(0))
        for begin in range(0, sequence, 64):
            var count = min(64, sequence - begin)
            for i in range(count):
                for j in range(i + 1):
                    var score = Float32(0)
                    for d in range(depth):
                        score += q.unsafe_offset((batch * sequence + begin + i) * depth + d).unsafe_load() * k.unsafe_offset((batch * sequence + begin + j) * depth + d).unsafe_load()
                    scores.unsafe_offset(i * 64 + j).unsafe_store(score)
                var full = width // 8 * 8
                for d in range(0, full, 8):
                    var inter = SIMD[DType.float32, 8](0)
                    var intra = SIMD[DType.float32, 8](0)
                    for x in range(depth):
                        var query = q.unsafe_offset((batch * sequence + begin + i) * depth + x).unsafe_load()
                        inter += SIMD[DType.float32, 8](query) * state.unsafe_offset(x * width + d).unsafe_load[width=8]()
                    for j in range(i + 1):
                        var score = scores.unsafe_offset(i * 64 + j).unsafe_load()
                        intra += SIMD[DType.float32, 8](score) * v.unsafe_offset((batch * sequence + begin + j) * width + d).unsafe_load[width=8]()
                    output.unsafe_offset((batch * sequence + begin + i) * width + d).unsafe_store(inter + intra)
                for d in range(full, width):
                    var inter = Float32(0)
                    var intra = Float32(0)
                    for x in range(depth):
                        inter += q.unsafe_offset((batch * sequence + begin + i) * depth + x).unsafe_load() * state.unsafe_offset(x * width + d).unsafe_load()
                    for j in range(i + 1):
                        intra += scores.unsafe_offset(i * 64 + j).unsafe_load() * v.unsafe_offset((batch * sequence + begin + j) * width + d).unsafe_load()
                    output.unsafe_offset((batch * sequence + begin + i) * width + d).unsafe_store(inter + intra)
            for x in range(depth):
                var full = width // 8 * 8
                for d in range(0, full, 8):
                    var summary = SIMD[DType.float32, 8](0)
                    for i in range(count):
                        var key = k.unsafe_offset((batch * sequence + begin + i) * depth + x).unsafe_load()
                        summary += SIMD[DType.float32, 8](key) * v.unsafe_offset((batch * sequence + begin + i) * width + d).unsafe_load[width=8]()
                    state.unsafe_offset(x * width + d).unsafe_store(state.unsafe_offset(x * width + d).unsafe_load[width=8]() + summary)
                for d in range(full, width):
                    var summary = Float32(0)
                    for i in range(count):
                        summary += k.unsafe_offset((batch * sequence + begin + i) * depth + x).unsafe_load() * v.unsafe_offset((batch * sequence + begin + i) * width + d).unsafe_load()
                    state.unsafe_offset(x * width + d).unsafe_store(state.unsafe_offset(x * width + d).unsafe_load() + summary)
        for i in range(depth * width):
            final_state.unsafe_offset(batch * depth * width + i).unsafe_store(state.unsafe_offset(i).unsafe_load())
        dealloc(score_storage^)
        dealloc(state_storage^)
        external_call["intent_cpu_leave_ieee", NoneType](saved)

    parallelize(batch_body, Int(qb), 8)
    external_call["intent_cpu_leave_ieee", NoneType](environment)
