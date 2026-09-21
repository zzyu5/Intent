static constexpr const char *tileImplementations = R"BANG(
#include <bang.h>
#include <bang_math_decls.h>
#include <cnrt.h>
#include <stdint.h>
#include <math.h>

__mlu_func__ float intent_bf16_to_float(uint16_t value) {
  __bang_bfloat16_raw raw;
  raw.x = value;
  return __bfloat162float__(bfloat16_t(raw));
}
__mlu_func__ uint16_t intent_float_to_bf16(float value) {
  __bang_bfloat16_raw raw = __float2bfloat16__(value);
  return raw.x;
}
__mlu_func__ uint64_t intent_round_shift(uint64_t value, int shift) {
  if (shift <= 0) return value << (-shift);
  if (shift > 64) return 0;
  if (shift == 64) return value > (uint64_t(1) << 63) ? 1 : 0;
  const uint64_t quotient = value >> shift;
  const uint64_t remainder = value & ((uint64_t(1) << shift) - 1);
  const uint64_t halfway = uint64_t(1) << (shift - 1);
  return quotient + (remainder > halfway || (remainder == halfway && (quotient & 1)));
}
__mlu_func__ uint16_t intent_number_to_bf16(double value) {
  union { double value; uint64_t bits; } converted;
  converted.value = value;
  const uint16_t sign = uint16_t(converted.bits >> 48) & 0x8000u;
  const unsigned exponent = unsigned((converted.bits >> 52) & 0x7ffu);
  const uint64_t fraction = converted.bits & ((uint64_t(1) << 52) - 1);
  if (exponent == 0x7ffu) return fraction ? 0x7fffu : (sign | 0x7f80u);
  if (exponent == 0) return sign;
  int e = int(exponent) - 1023;
  const uint64_t significand = (uint64_t(1) << 52) | fraction;
  if (e < -126) return sign | uint16_t(intent_round_shift(significand, -e - 81));
  if (e > 127) return sign | 0x7f80u;
  uint64_t rounded = intent_round_shift(significand, 45);
  if (rounded == 256) { rounded = 128; ++e; }
  if (e > 127) return sign | 0x7f80u;
  return sign | uint16_t((e + 127) << 7) | uint16_t(rounded - 128);
}
__mlu_func__ uint16_t intent_number_to_bf16(int64_t value) {
  if (!value) return 0;
  const uint16_t sign = value < 0 ? 0x8000u : 0;
  const uint64_t magnitude = value < 0 ? uint64_t(0) - uint64_t(value) : uint64_t(value);
  int exponent = 0;
  for (uint64_t remaining = magnitude; remaining > 1; remaining >>= 1) ++exponent;
  uint64_t rounded = intent_round_shift(magnitude, exponent - 7);
  if (rounded == 256) { rounded = 128; ++exponent; }
  return sign | uint16_t((exponent + 127) << 7) | uint16_t(rounded - 128);
}
__mlu_func__ uint16_t intent_number_to_bf16(float value) { return intent_float_to_bf16(value); }
__mlu_func__ uint16_t intent_number_to_bf16(half value) { return intent_float_to_bf16(float(value)); }
__mlu_func__ uint16_t intent_number_to_bf16(int32_t value) { return intent_number_to_bf16(int64_t(value)); }
__mlu_func__ uint16_t intent_number_to_bf16(bool value) { return value ? 0x3f80u : 0; }
template<class From, class To> struct IntentLocalCast {
  __mlu_func__ static To apply(From value) { return static_cast<To>(value); }
};
template<class To> struct IntentLocalCast<uint16_t, To> {
  __mlu_func__ static To apply(uint16_t value) { return static_cast<To>(intent_bf16_to_float(value)); }
};
template<class From> struct IntentLocalCast<From, uint16_t> {
  __mlu_func__ static uint16_t apply(From value) { return intent_number_to_bf16(value); }
};

template<class T> __mlu_func__ T intent_read_local(const T *p) { return __load_nram(p); }
template<class T> __mlu_func__ void intent_write_local(T *p, T v) { __store_nram(p, v); }
template<> __mlu_func__ bool intent_read_local(const bool *p) {
  return __load_nram(reinterpret_cast<const uint8_t *>(p)) != 0;
}
template<> __mlu_func__ void intent_write_local(bool *p, bool v) {
  __store_nram(reinterpret_cast<uint8_t *>(p), uint8_t(v));
}
template<class T, int N>
__mlu_func__ void intent_copy_local(T *output, const T *input) {
  for (int i = 0; i < N; ++i) intent_write_local(output + i, intent_read_local(input + i));
}
template<class From, class To, int N>
__mlu_func__ void intent_cast_local(To *output, const From *input) {
  for (int i = 0; i < N; ++i)
    intent_write_local(output + i, IntentLocalCast<From, To>::apply(intent_read_local(input + i)));
}
template<int N> __mlu_func__ void intent_bf16_to_f32_tile(float *output, const uint16_t *input) {
  __bang_bfloat162float(output, reinterpret_cast<const bfloat16_t *>(input), N);
}
template<int N> __mlu_func__ void intent_f32_to_bf16_tile(uint16_t *output, const float *input) {
  __bang_float2bfloat16_rn(reinterpret_cast<bfloat16_t *>(output), input, N);
}
__mlu_func__ float intent_flush_subnormal(float value) {
  union { float value; uint32_t bits; } converted;
  converted.value = value;
  if ((converted.bits & 0x7f800000u) == 0) converted.bits &= 0x80000000u;
  return converted.value;
}
__mlu_func__ float intent_exp2_subnormal(float value) {
  union { float value; uint32_t bits; } converted;
  converted.bits = 0;
  if (value < -150.0f) return converted.value;
  // The scaled result is normal. Round its mantissa directly into the f32
  // encoding, so a device's arithmetic FTZ mode cannot erase subnormal output.
  // compute_30 scalar exp2 has a documented 1-ULP bound; the final integer
  // rounding stays within the source approximate-exp2 contract's 2 ULP.
  const float scaled = exp2f(value + 149.0f);
  uint32_t mantissa = static_cast<uint32_t>(scaled);
  const float remainder = scaled - float(mantissa);
  if (remainder > 0.5f || (remainder == 0.5f && (mantissa & 1u))) ++mantissa;
  converted.bits = mantissa;
  return converted.value;
}
template<int N, bool Flush>
__mlu_func__ void intent_exp2_tile(float *output, const float *input) {
  if (Flush) {
    for (int i = 0; i < N; ++i) intent_write_local(output + i, intent_flush_subnormal(intent_read_local(input + i)));
    __sync();
    __cn_vector_exp2_f32(N, output, output);
  } else {
    __cn_vector_exp2_f32(N, output, input);
  }
  __sync_compute();
  for (int i = 0; i < N; ++i) {
    float value = intent_read_local(output + i);
    if (Flush) value = intent_flush_subnormal(value);
    else {
      const float x = intent_read_local(input + i);
      if (x < -126.0f) value = intent_exp2_subnormal(x);
    }
    intent_write_local(output + i, value);
  }
}
template<class T, int N>
__mlu_func__ void intent_fill_local(T *output, T value) {
  for (int i = 0; i < N; ++i) intent_write_local(output + i, value);
}
template<int N> __mlu_func__ void intent_zero_tile(float *output) { __bang_write_value(output, N, 0.0f); }
template<int N> __mlu_func__ void intent_zero_tile(half *output) { __bang_write_value(output, N, half(0)); }
template<int N, class T> __mlu_func__ void intent_zero_tile(T *output) { intent_fill_local<T, N>(output, T(0)); }
template<class T, int N>
__mlu_func__ void intent_divide_local(T *output, const T *lhs, const T *rhs) {
  // MLU370 __bang_div is an abs/reciprocal/sign/multiply sequence. Use scalar
  // division here to retain signed-zero, infinity and NaN behavior.
  for (int i = 0; i < N; ++i)
    intent_write_local(output + i, T(intent_read_local(lhs + i) / intent_read_local(rhs + i)));
}
template<int N, bool Approximate, bool Flush>
__mlu_func__ void intent_divide_f32(float *output, const float *lhs, const float *rhs) {
  for (int i = 0; i < N; ++i) {
    float a = intent_read_local(lhs + i), b = intent_read_local(rhs + i);
    if (Flush) { a = intent_flush_subnormal(a); b = intent_flush_subnormal(b); }
    union { float value; uint32_t bits; } left, right, result;
    left.value = a; right.value = b;
    const uint32_t sign = (left.bits ^ right.bits) & 0x80000000u;
    const uint32_t aa = left.bits & 0x7fffffffu, bb = right.bits & 0x7fffffffu;
    if (aa > 0x7f800000u || bb > 0x7f800000u) result.bits = 0x7fc00000u;
    else if (Approximate && bb > 0x7e800000u) result.bits = aa == 0x7f800000u ? 0x7fc00000u : sign;
    else if (bb == 0) result.bits = aa == 0 ? 0x7fc00000u : sign | 0x7f800000u;
    else if (aa == 0x7f800000u) result.bits = bb == 0x7f800000u ? 0x7fc00000u : sign | 0x7f800000u;
    else if (bb == 0x7f800000u || aa == 0) result.bits = sign;
    else result.value = __cn_scalar_div_f32_rn(a, b);
    // compute_30 div.rn is documented as RNE with no error. Approximate mode
    // additionally owns the large-denominator rule and the local FTZ boundary.
    intent_write_local(output + i, Flush ? intent_flush_subnormal(result.value) : result.value);
  }
}
template<class T, int N>
__mlu_func__ void intent_select_local(T *output, bool condition, const T *lhs, const T *rhs) {
  for (int i = 0; i < N; ++i) intent_write_local(output + i, intent_read_local((condition ? lhs : rhs) + i));
}
template<class T, int N>
__mlu_func__ void intent_select_local(T *output, const bool *condition, const T *lhs, const T *rhs) {
  for (int i = 0; i < N; ++i) intent_write_local(output + i, intent_read_local((intent_read_local(condition + i) ? lhs : rhs) + i));
}

template<class T, int R, int C>
__mlu_func__ void intent_load_tile(T *dst, const T *src, int64_t offset,
    int64_t row_stride, int64_t column_stride, int64_t rows, int64_t columns) {
  intent_zero_tile<R * C>(dst);
  __sync_compute();
  for (int64_t r = 0; r < rows; ++r) {
    if (column_stride == 1) {
      if (columns) __memcpy(dst + r * C, src + offset + r * row_stride,
                            columns * sizeof(T), GDRAM2NRAM);
    } else {
      for (int64_t c = 0; c < columns; ++c)
        dst[r * C + c] = src[offset + r * row_stride + c * column_stride];
    }
  }
}
template<class T, int R, int C>
__mlu_func__ void intent_store_tile(T *dst, const T *src, int64_t offset,
    int64_t row_stride, int64_t column_stride, int64_t rows, int64_t columns) {
  for (int64_t r = 0; r < rows; ++r) {
    if (column_stride == 1) {
      if (columns) __memcpy(dst + offset + r * row_stride, src + r * C,
                            columns * sizeof(T), NRAM2GDRAM);
    } else {
      for (int64_t c = 0; c < columns; ++c)
        dst[offset + r * row_stride + c * column_stride] = src[r * C + c];
    }
  }
}
template<class T, int R, int C>
__mlu_func__ void intent_load_local_tile(T *dst, const T *src, int64_t offset,
    int64_t row_stride, int64_t column_stride, int64_t rows, int64_t columns) {
  intent_zero_tile<R * C>(dst);
  __sync_compute();
  if (!rows || !columns) return;
  if (column_stride == 1 && row_stride >= columns) {
    __memcpy(dst, src + offset, columns * sizeof(T), NRAM2NRAM,
             C * sizeof(T), row_stride * sizeof(T), rows - 1);
    return;
  }
  for (int64_t r = 0; r < rows; ++r) {
    if (column_stride > 0)
      __memcpy(dst + r * C, src + offset + r * row_stride, sizeof(T), NRAM2NRAM,
               sizeof(T), column_stride * sizeof(T), columns - 1);
    else for (int64_t c = 0; c < columns; ++c)
      intent_write_local(dst + r * C + c, intent_read_local(src + offset + r * row_stride + c * column_stride));
  }
}
template<class T, int R, int C>
__mlu_func__ void intent_store_local_tile(T *dst, const T *src, int64_t offset,
    int64_t row_stride, int64_t column_stride, int64_t rows, int64_t columns) {
  if (!rows || !columns) return;
  if (column_stride == 1 && row_stride >= columns) {
    __memcpy(dst + offset, src, columns * sizeof(T), NRAM2NRAM,
             row_stride * sizeof(T), C * sizeof(T), rows - 1);
    return;
  }
  for (int64_t r = 0; r < rows; ++r) {
    if (column_stride > 0)
      __memcpy(dst + offset + r * row_stride, src + r * C, sizeof(T), NRAM2NRAM,
               column_stride * sizeof(T), sizeof(T), columns - 1);
    else for (int64_t c = 0; c < columns; ++c)
      intent_write_local(dst + offset + r * row_stride + c * column_stride, intent_read_local(src + r * C + c));
  }
}
template<class T, int K, int N>
__mlu_func__ void intent_prepare_matrix(T *packed, const T *row_major, T *transposed) {
  __bang_transpose(transposed, row_major, K, N);
  __sync_move();
  // For N=64 the MLU370 matmul reshape is the contiguous [N,K] transpose.
  // Larger N requires an additional four-row interleave.
  static_assert(N == 64, "matmul RHS implementation requires N=64");
  __memcpy(packed, transposed, K * N * sizeof(T), NRAM2WRAM);
  __sync_move();
}
template<class T> __mlu_func__ void intent_matrix_primitive(float *output, const T *a, const T *b, int m, int k, int n) {
  __bang_matmul(output, a, b, m, k, n);
}
template<> __mlu_func__ void intent_matrix_primitive(float *output, const uint16_t *a, const uint16_t *b, int m, int k, int n) {
  __bang_matmul(output, reinterpret_cast<const bfloat16_t *>(a), reinterpret_cast<const bfloat16_t *>(b), m, k, n);
}
template<class T, int M, int K, int N>
__mlu_func__ void intent_matmul(float *accumulator, const T *a, const T *packed_b, float *partial) {
  intent_matrix_primitive(partial, a, packed_b, M, K, N);
  __sync_compute();
  __bang_add(accumulator, accumulator, partial, M * N);
}
template<int N, int Kind>
__mlu_func__ void intent_reduce(float *out, const float *input, float *scratch,
                              int64_t count, float identity) {
  // The scratch and result storage are explicit operands in the DSA program.
  // MLU370 reduce_sum forms 128-byte partial sums; the remaining scalar fold
  // combines those partials and the declared identity.
  for (int64_t i = 0; i < N; ++i) scratch[i] = i < count ? input[i] : identity;
  if (Kind == 0) {
    static_assert(N >= 32 && N % 32 == 0, "reduce_sum requires 128-byte input groups");
    __bang_reduce_sum(scratch, scratch, N);
    __sync_compute();
    float value = identity;
    for (int i = 0; i < N / 32; ++i) value += scratch[i * 32];
    out[0] = value;
  } else {
    float value = identity;
    for (int64_t i = 0; i < count; ++i) {
      const float next = scratch[i];
      if (Kind == 7) value = isnan(value) || isnan(next) ? NAN : fmaxf(value, next);
      if (Kind == 8) value = isnan(value) || isnan(next) ? NAN : fminf(value, next);
      if (Kind == 9) value = fmaxf(value, next);
      if (Kind == 10) value = fminf(value, next);
    }
    out[0] = value;
  }
}
)BANG";
