#include "Syntax/Spelling.h"

#include <optional>

using namespace mlir;
using namespace llvm;

namespace intent::tilelang::emission::syntax {
namespace {

std::optional<std::string> blockParameter(StringRef role) {
  if (role == "lane_pack")
    return "TILE_SIZE_L";
  if (role.starts_with("lane_pack_"))
    return "TILE_SIZE_L" + role.drop_front(10).str();
  if (role == "program_m" || role == "ragged_member")
    return "TILE_SIZE_M";
  if (role.starts_with("ragged_member_"))
    return "TILE_SIZE_R" + role.drop_front(14).str();
  if (role.starts_with("query_"))
    return "TILE_SIZE_Q" + role.drop_front(6).str();
  if (role == "query")
    return "TILE_SIZE_Q";
  if (role == "program_n" || role == "feature")
    return "TILE_SIZE_N";
  if (role == "stream")
    return "TILE_SIZE_S";
  if (role == "scan")
    return "TILE_SIZE_SCAN";
  if (role.starts_with("scan_"))
    return "TILE_SIZE_SCAN" + role.drop_front(5).str();
  if (role == "stream_contract")
    return "TILE_SIZE_C";
  if (role.starts_with("stream_contract_"))
    return "TILE_SIZE_C" + role.drop_front(16).str();
  if (role.starts_with("stream_"))
    return "TILE_SIZE_S" + role.drop_front(7).str();
  if (role.starts_with("program_"))
    return "TILE_SIZE_P" + role.drop_front(8).str();
  if (role == "reduction")
    return "TILE_SIZE_K";
  if (role.starts_with("reduction_"))
    return "TILE_SIZE_K" + role.drop_front(10).str();
  return std::nullopt;
}

} // namespace

std::string dimension(StringRef symbol) {
  return symbol == "T" ? "DIM_T" : symbol.str();
}

FailureOr<std::string> tile(Operation *operation, StringRef role) {
  if (role == "one")
    return std::string("1");
  if (role.starts_with("fixed_"))
    return role.drop_front(6).str();
  if (role == "row_vector")
    return std::string("TILE_SIZE");
  if (role.starts_with("row_vector_"))
    return "TILE_SIZE_V" + role.drop_front(11).str();
  if (std::optional<std::string> spelling = blockParameter(role))
    return *spelling;
  operation->emitOpError("has no TileLang tile spelling for role ") << role;
  return failure();
}

FailureOr<std::string> parameter(Operation *operation, StringRef role) {
  if (role == "group_m")
    return std::string("GROUP_SIZE_M");
  if (role.starts_with("group_"))
    return "GROUP_SIZE_G" + role.drop_front(6).str();
  if (std::optional<std::string> spelling = blockParameter(role))
    return *spelling;
  operation->emitOpError("has no TileLang tuner parameter for role ") << role;
  return failure();
}

FailureOr<StringRef> pointwise(Operation *operation, StringRef role,
                               StringRef materialization) {
  if (role == "indices")
    return StringRef("logical_indices");
  if (role == "counter_random_f32")
    return StringRef("counter_xorshift32");
  if (role == "broadcast")
    return StringRef("alias");
  if (role == "cast")
    return materialization == "contract_operand" ? StringRef("T.copy_cast")
                                                   : StringRef("T.cast");
  if (role == "reshape")
    return StringRef("T.reshape");
  if (role == "transpose")
    return StringRef("T.transpose");
  if (role == "unary_exp")
    return StringRef("T.exp");
  if (role == "unary_exp2")
    return StringRef("T.exp2");
  if (role == "unary_log")
    return StringRef("T.log");
  if (role == "unary_rsqrt")
    return StringRef("T.rsqrt");
  if (role == "unary_sigmoid")
    return StringRef("T.sigmoid");
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
    return StringRef("T.pow");
  if (role == "binary_bitwise_and")
    return StringRef("T.bitwise_and");
  if (role == "binary_bitwise_or")
    return StringRef("T.bitwise_or");
  if (role == "binary_bitwise_xor")
    return StringRef("T.bitwise_xor");
  if (role == "binary_left_shift")
    return StringRef("T.shift_left");
  if (role == "binary_right_shift")
    return StringRef("T.shift_right");
  if (role == "binary_logical_and")
    return StringRef("python_logical_and");
  if (role == "binary_logical_or")
    return StringRef("python_logical_or");
  if (role == "binary_maximum")
    return StringRef("T.max");
  if (role == "binary_minimum")
    return StringRef("T.min");
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
    return StringRef("T.if_then_else");
  if (role == "full")
    return StringRef("T.fill");
  if (role == "zeros")
    return StringRef("T.clear");
  if (role == "members")
    return StringRef("T.members");
  if (role == "expand_dims")
    return StringRef("expand_dims");
  if (role == "indirect_gather")
    return StringRef("T.indirect_gather");
  if (role == "extract_unit_scalar")
    return StringRef("T.extract_unit_scalar");
  operation->emitOpError("has no TileLang pointwise spelling for role ") << role;
  return failure();
}

StringRef bufferSpace(StringRef space) {
  if (space == "external" || space == "workspace")
    return "global";
  if (space == "shared")
    return "shared";
  if (space == "private_fragment")
    return "fragment";
  if (space == "private_scalar")
    return "local";
  if (space == "none")
    return "none";
  return {};
}

StringRef reduction(StringRef role) {
  return role == "reduce_argmax"    ? "T.reduce_max_with_index"
         : role == "reduce_maximum" ? "T.reduce_max"
         : role == "reduce_any"     ? "T.reduce_any_i32"
         : role == "reduce_all"     ? "T.reduce_all_i32"
                                      : "T.reduce_sum";
}

StringRef scan() { return "T.cumsum"; }

StringRef contraction() { return "T.gemm"; }

std::string cast(StringRef value, StringRef targetType, bool decodeE8M0,
                 bool resultIsF32) {
  if (!decodeE8M0)
    return "T.cast(" + value.str() + ", " + targetType.str() + ")";
  std::string bits = "T.reinterpret(" + value.str() + ", T.uint8)";
  std::string bits32 = "T.cast(" + bits + ", T.uint32)";
  std::string normalBits = "T.shift_left(" + bits32 + ", 23)";
  std::string encoded =
      "T.if_then_else(" + bits +
      " == T.cast(255, T.uint8), T.cast(2143289344, T.uint32), "
      "T.if_then_else(" + bits +
      " == T.cast(0, T.uint8), T.cast(4194304, T.uint32), " + normalBits +
      "))";
  std::string decoded = "T.reinterpret(" + encoded + ", T.float32)";
  return resultIsF32
             ? decoded
             : "T.cast(" + decoded + ", " + targetType.str() + ")";
}

} // namespace intent::tilelang::emission::syntax
