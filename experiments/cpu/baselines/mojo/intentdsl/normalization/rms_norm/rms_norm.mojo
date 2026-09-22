# IntentDSL-authored CPU baseline. SIMD accumulators and outer-row parallelism
# follow Modular's CPU reduction composition; this is not an upstream copy.
from max.algorithm import parallelize
from std.ffi import external_call
from std.math import sqrt
from std.runtime import initialize_runtime


@export("source_rms_norm")
@no_inline
def source_rms_norm(
    a0: Pointer[Float32, MutUntrackedOrigin], a0_d0: Int64, a0_d1: Int64, a0_s0: Int64, a0_s1: Int64,
    a1: Pointer[Float32, MutUntrackedOrigin], a1_d0: Int64, a1_s0: Int64,
    a2: Pointer[Float32, MutUntrackedOrigin], a2_d0: Int64, a2_d1: Int64, a2_s0: Int64, a2_s1: Int64,
    a3: Float32, a4: Float32,
) abi("C"):
    initialize_runtime()
    var environment = external_call["intent_cpu_enter_ieee", UInt32]()
    var columns = Int(a0_d1)

    def row_body(row: Int) {imm a0, imm a1, imm a2, imm a3, imm a4, imm columns}:
        var saved = external_call["intent_cpu_enter_ieee", UInt32]()
        var full = columns // 16 * 16
        var sums = SIMD[DType.float32, 16](0)
        for column in range(0, full, 16):
            var x = a0.unsafe_offset(row * columns + column).unsafe_load[width=16]()
            sums = sums + x * x
        var total = sums.reduce_add()
        for column in range(full, columns):
            var x = a0.unsafe_offset(row * columns + column).unsafe_load()
            total = total + x * x
        var inverse = Float32(1.0) / sqrt(total * a3 + a4)
        var inverse_v = SIMD[DType.float32, 16](inverse)
        for column in range(0, full, 16):
            var x = a0.unsafe_offset(row * columns + column).unsafe_load[width=16]()
            var weight = a1.unsafe_offset(column).unsafe_load[width=16]()
            a2.unsafe_offset(row * columns + column).unsafe_store(x * inverse_v * weight)
        for column in range(full, columns):
            var x = a0.unsafe_offset(row * columns + column).unsafe_load()
            var weight = a1.unsafe_offset(column).unsafe_load()
            a2.unsafe_offset(row * columns + column).unsafe_store(x * inverse * weight)
        external_call["intent_cpu_leave_ieee", NoneType](saved)

    parallelize(row_body, Int(a0_d0), 8)
    external_call["intent_cpu_leave_ieee", NoneType](environment)
