#include "Syntax/Spelling.h"

#include <optional>

using namespace mlir;
using namespace llvm;

namespace intent::triton::emission::syntax {
namespace {

std::optional<std::string> blockParameter(StringRef role) {
  if (role == "lane_pack")
    return "BLOCK_SIZE_L";
  if (role.starts_with("lane_pack_"))
    return "BLOCK_SIZE_L" + role.drop_front(10).str();
  if (role == "program_m" || role == "ragged_member")
    return "BLOCK_SIZE_M";
  if (role.starts_with("ragged_member_"))
    return "BLOCK_SIZE_R" + role.drop_front(14).str();
  if (role == "program_n" || role == "feature")
    return "BLOCK_SIZE_N";
  if (role.starts_with("program_"))
    return "BLOCK_SIZE_P" + role.drop_front(8).str();
  if (role == "reduction")
    return "BLOCK_SIZE_K";
  if (role.starts_with("reduction_"))
    return "BLOCK_SIZE_K" + role.drop_front(10).str();
  if (role == "query")
    return "BLOCK_SIZE_Q";
  if (role.starts_with("query_"))
    return "BLOCK_SIZE_Q" + role.drop_front(6).str();
  if (role == "stream")
    return "BLOCK_SIZE_S";
  if (role == "scan")
    return "BLOCK_SIZE_SCAN";
  if (role.starts_with("scan_"))
    return "BLOCK_SIZE_SCAN" + role.drop_front(5).str();
  if (role == "stream_contract")
    return "BLOCK_SIZE_C";
  if (role.starts_with("stream_contract_"))
    return "BLOCK_SIZE_C" + role.drop_front(16).str();
  if (role.starts_with("stream_"))
    return "BLOCK_SIZE_S" + role.drop_front(7).str();
  return std::nullopt;
}

} // namespace

FailureOr<std::string> tile(Operation *operation, StringRef role) {
  if (role == "one")
    return std::string("1");
  if (role.starts_with("fixed_"))
    return role.drop_front(6).str();
  if (role == "row_vector")
    return std::string("BLOCK_SIZE");
  if (role.starts_with("row_vector_"))
    return "BLOCK_SIZE_V" + role.drop_front(11).str();
  if (std::optional<std::string> spelling = blockParameter(role))
    return *spelling;
  operation->emitOpError("has no Triton tile spelling for role ") << role;
  return failure();
}

FailureOr<std::string> parameter(Operation *operation, StringRef role) {
  if (role == "group_m")
    return std::string("GROUP_SIZE_M");
  if (role.starts_with("group_"))
    return "GROUP_SIZE_G" + role.drop_front(6).str();
  if (std::optional<std::string> spelling = blockParameter(role))
    return *spelling;
  operation->emitOpError("has no Triton tuner parameter for role ") << role;
  return failure();
}

FailureOr<StringRef> pointwise(Operation *operation, StringRef role) {
  if (role == "indices")
    return StringRef("logical_indices");
  if (role == "counter_random_f32")
    return StringRef("counter_xorshift32");
  if (role == "broadcast")
    return StringRef("alias");
  if (role == "cast")
    return StringRef("tl.cast");
  if (role == "reshape")
    return StringRef("tl.reshape");
  if (role == "transpose")
    return StringRef("tl.permute");
  if (role == "unary_exp")
    return StringRef("tl.exp");
  if (role == "unary_exp2")
    return StringRef("tl.exp2");
  if (role == "unary_log")
    return StringRef("tl.log");
  if (role == "unary_rsqrt")
    return StringRef("tl.rsqrt");
  if (role == "unary_sigmoid")
    return StringRef("tl.sigmoid");
  if (role == "unary_negate")
    return StringRef("python_negate");
  if (role == "unary_not")
    return StringRef("python_not");
  if (role == "binary_add")
    return StringRef("python_add");
  if (role == "binary_subtract")
    return StringRef("python_subtract");
  if (role == "binary_multiply")
    return StringRef("python_multiply");
  if (role == "binary_true_divide")
    return StringRef("python_true_divide");
  if (role == "binary_floor_divide")
    return StringRef("python_floor_divide");
  if (role == "binary_remainder")
    return StringRef("python_remainder");
  if (role == "binary_power")
    return StringRef("libdevice.pow");
  if (role == "binary_bitwise_and")
    return StringRef("python_bitwise_and");
  if (role == "binary_bitwise_or")
    return StringRef("python_bitwise_or");
  if (role == "binary_bitwise_xor")
    return StringRef("python_bitwise_xor");
  if (role == "binary_left_shift")
    return StringRef("python_left_shift");
  if (role == "binary_right_shift")
    return StringRef("python_right_shift");
  if (role == "binary_logical_and")
    return StringRef("python_logical_and");
  if (role == "binary_logical_or")
    return StringRef("python_logical_or");
  if (role == "binary_maximum")
    return StringRef("tl.maximum");
  if (role == "binary_minimum")
    return StringRef("tl.minimum");
  if (role == "compare_equal")
    return StringRef("python_equal");
  if (role == "compare_not_equal")
    return StringRef("python_not_equal");
  if (role == "compare_less")
    return StringRef("python_less");
  if (role == "compare_less_equal")
    return StringRef("python_less_equal");
  if (role == "compare_greater")
    return StringRef("python_greater");
  if (role == "compare_greater_equal")
    return StringRef("python_greater_equal");
  if (role == "mask" || role == "select")
    return StringRef("tl.where");
  if (role == "full")
    return StringRef("tl.full");
  if (role == "zeros")
    return StringRef("tl.zeros");
  if (role == "members")
    return StringRef("tl.members");
  if (role == "expand_dims")
    return StringRef("expand_dims");
  if (role == "indirect_gather")
    return StringRef("tl.indirect_gather");
  if (role == "extract_unit_scalar")
    return StringRef("tl.extract_unit_scalar");
  operation->emitOpError("has no Triton pointwise spelling for role ") << role;
  return failure();
}

StringRef reduction(StringRef role) {
  return role == "reduce_argmax"    ? "tl.max_with_index"
         : role == "reduce_generic" ? "tl.reduce"
         : role == "reduce_maximum" ? "tl.max"
         : role == "reduce_any"     ? "tl.reduce_or"
         : role == "reduce_all"     ? "tl.reduce_all"
                                      : "tl.sum";
}

StringRef scan(StringRef role) {
  return role == "scan_generic_inclusive" ? "tl.associative_scan"
                                            : "tl.cumsum";
}

StringRef contraction() { return "tl.dot"; }

std::string cast(StringRef value, StringRef targetType, bool decodeE8M0,
                 bool resultIsF32) {
  if (!decodeE8M0)
    return "tl.cast(" + value.str() + ", " + targetType.str() + ")";
  std::string bits = "tl.cast(" + value.str() + ", tl.uint32)";
  std::string normalBits = "(" + bits + " << 23)";
  std::string encoded =
      "tl.where(" + bits + " == 255, tl.cast(2143289344, tl.uint32), "
      "tl.where(" + bits +
      " == 0, tl.cast(4194304, tl.uint32), " + normalBits + "))";
  std::string decoded =
      "tl.cast(" + encoded + ", tl.float32, bitcast=True)";
  return resultIsF32
             ? decoded
             : "tl.cast(" + decoded + ", " + targetType.str() + ")";
}

} // namespace intent::triton::emission::syntax
