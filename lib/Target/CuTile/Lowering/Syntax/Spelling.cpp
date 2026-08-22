#include "Syntax/Spelling.h"

#include "Intent/Target/CuTile/Lowering/Passes.h"

#include <optional>

using namespace mlir;
using namespace llvm;

namespace intent::cutile::lowering::syntax {
namespace {

std::optional<std::string> blockParameter(StringRef role) {
  if (role == "lane_pack")
    return "TILE_SIZE_L";
  if (role.starts_with("lane_pack_"))
    return "TILE_SIZE_L" + role.drop_front(10).str();
  if (role == "pointwise_lane")
    return "TILE_SIZE_M";
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
  if (role == "stream_scaled")
    return "TILE_SIZE_SCALE_GROUPS";
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

FailureOr<std::string> tile(Operation *operation, StringRef role) {
  if (role == "one")
    return std::string("1");
  if (role.starts_with("fixed_"))
    return role.drop_front(6).str();
  if (role.starts_with("partition_extent_"))
    return "PARTITION_EXTENT_" + role.drop_front(17).str();
  if (role == "row_vector")
    return std::string("TILE_SIZE");
  if (role.starts_with("row_vector_"))
    return "TILE_SIZE_V" + role.drop_front(11).str();
  if (std::optional<std::string> spelling = blockParameter(role))
    return *spelling;
  operation->emitOpError("has no cuTile tile spelling for role ") << role;
  return failure();
}

FailureOr<std::string> parameter(Operation *operation, StringRef role) {
  if (role == gatherSpellingRole)
    return std::string("GATHER_SPELLING");
  if (role == "group_m")
    return std::string("GROUP_SIZE_M");
  if (role.starts_with("group_"))
    return "GROUP_SIZE_G" + role.drop_front(6).str();
  if (std::optional<std::string> spelling = blockParameter(role))
    return *spelling;
  operation->emitOpError("has no cuTile tuner parameter for role ") << role;
  return failure();
}

FailureOr<StringRef> pointwise(Operation *operation, StringRef role,
                               StringRef resultSpace) {
  if (role == "indices")
    return StringRef("logical_indices");
  if (role == "counter_random_f32")
    return StringRef("counter_xorshift32");
  if (role == "broadcast")
    return StringRef("alias");
  if (role == "cast")
    return resultSpace == "private_scalar" ? StringRef("ct.full_cast")
                                            : StringRef("ct.astype");
  if (role == "bitcast")
    return StringRef("ct.bitcast");
  if (role == "reshape")
    return StringRef("ct.reshape");
  if (role == "transpose")
    return StringRef("ct.permute");
  if (role == "unary_exp")
    return StringRef("ct.exp");
  if (role == "unary_exp2")
    return StringRef("ct.exp2");
  if (role == "unary_log")
    return StringRef("ct.log");
  if (role == "unary_sin")
    return StringRef("ct.sin");
  if (role == "unary_cos")
    return StringRef("ct.cos");
  if (role == "unary_floor")
    return StringRef("ct.floor");
  if (role == "unary_rsqrt")
    return StringRef("ct.rsqrt");
  if (role == "unary_sigmoid")
    return StringRef("python_sigmoid");
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
    return StringRef("ct.pow");
  if (role == "binary_bitwise_and")
    return StringRef("ct.bitwise_and");
  if (role == "binary_bitwise_or")
    return StringRef("ct.bitwise_or");
  if (role == "binary_bitwise_xor")
    return StringRef("ct.bitwise_xor");
  if (role == "binary_left_shift")
    return StringRef("ct.bitwise_lshift");
  if (role == "binary_right_shift")
    return StringRef("ct.bitwise_rshift");
  if (role == "binary_logical_and")
    return StringRef("python_logical_and");
  if (role == "binary_logical_or")
    return StringRef("python_logical_or");
  if (role == "binary_maximum")
    return StringRef("ct.maximum");
  if (role == "binary_minimum")
    return StringRef("ct.minimum");
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
    return StringRef("ct.where");
  if (role == "full")
    return StringRef("ct.full");
  if (role == "zeros")
    return StringRef("ct.zeros");
  if (role == "members")
    return StringRef("ct.members");
  if (role == "expand_dims")
    return StringRef("expand_dims");
  if (role == "indirect_gather")
    return StringRef("ct.indirect_gather");
  if (role == "extract_first_scalar")
    return StringRef("ct.extract_first_scalar");
  operation->emitOpError("has no cuTile pointwise spelling for role ") << role;
  return failure();
}

StringRef reduction(StringRef role) {
  return role == "reduce_argmax"    ? "ct.max_with_index"
         : role == "reduce_generic" ? "ct.reduce"
         : role == "reduce_maximum" ? "ct.max"
         : role == "reduce_any"     ? "ct.any_via_max"
         : role == "reduce_all"     ? "ct.all_via_min"
                                      : "ct.sum";
}

StringRef scan(StringRef role) {
  return role == "scan_generic_inclusive" ? "ct.scan" : "ct.cumsum";
}

StringRef contraction() { return "ct.mma"; }

StringRef scaledContraction() { return "ct.mma_scaled"; }

std::string gather(StringRef array, StringRef indices, StringRef padding,
                   StringRef mask) {
  std::string result = "ct.gather(" + array.str() + ", " + indices.str();
  if (!mask.empty())
    result += ", mask=" + mask.str();
  return result + ", check_bounds=True, padding_value=" + padding.str() + ")";
}

std::string cast(StringRef lowering, StringRef value, StringRef targetType) {
  if (lowering == "ct.full_cast")
    return "ct.full((), " + value.str() + ", dtype=" + targetType.str() + ")";
  return "ct.astype(" + value.str() + ", " + targetType.str() + ")";
}

std::string bitcast(StringRef value, StringRef targetType) {
  return "ct.bitcast(" + value.str() + ", " + targetType.str() + ")";
}

} // namespace intent::cutile::lowering::syntax
