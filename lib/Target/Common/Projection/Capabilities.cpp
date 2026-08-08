#include "Intent/Target/Common/Projection/Capabilities.h"

#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace intent::target {
namespace {

constexpr StringLiteral concepts[] = {
    "tile_shape",          "ownership",
    "traversal",           "boundary",
    "value_residency",     "layout",
    "matrix_primitive",    "pipeline",
    "launch_resources",    "register_allocation",
    "instruction_selection", "autotune_candidates",
    "explicit_onchip_buffer_allocation",
};

LogicalResult verifyCategory(Operation *operation, ArrayAttr actual,
                             ArrayRef<StringRef> expected, StringRef category) {
  llvm::StringSet<> actualValues;
  for (Attribute attribute : actual)
    actualValues.insert(cast<StringAttr>(attribute).getValue());
  if (actualValues.size() != expected.size())
    return operation->emitOpError()
           << "does not match the " << category << " capability profile";
  for (StringRef value : expected)
    if (!actualValues.contains(value))
      return operation->emitOpError()
             << "does not match the " << category << " capability profile";
  return success();
}

} // namespace

LogicalResult verifyCapabilityPartition(Operation *operation,
                                        ArrayAttr intentDecided,
                                        ArrayAttr delegated,
                                        ArrayAttr absent) {
  llvm::StringSet<> expected;
  for (StringRef concept : concepts)
    expected.insert(concept);
  llvm::StringSet<> seen;
  auto consume = [&](ArrayAttr values) -> LogicalResult {
    for (Attribute attribute : values) {
      auto value = dyn_cast<StringAttr>(attribute);
      if (!value || !expected.contains(value.getValue()))
        return operation->emitOpError(
            "contains an unknown physical capability concept");
      if (!seen.insert(value.getValue()).second)
        return operation->emitOpError(
            "assigns one physical concept more than once");
    }
    return success();
  };
  if (failed(consume(intentDecided)) || failed(consume(delegated)) ||
      failed(consume(absent)))
    return failure();
  if (seen.size() != expected.size())
    return operation->emitOpError(
        "must classify every physical concept as decided, delegated, or absent");
  return success();
}

LogicalResult verifyCapabilityProfile(Operation *operation,
                                      ArrayAttr intentDecided,
                                      ArrayAttr delegated, ArrayAttr absent,
                                      const CapabilityProfile &profile) {
  if (failed(verifyCapabilityPartition(operation, intentDecided, delegated,
                                       absent)))
    return failure();
  if (failed(verifyCategory(operation, intentDecided, profile.intentDecided,
                            "intent-decided")) ||
      failed(verifyCategory(operation, delegated, profile.delegated,
                            "delegated")) ||
      failed(verifyCategory(operation, absent, profile.absent, "absent")))
    return failure();
  return success();
}

LogicalResult verifyParameterMap(Operation *operation,
                                 DictionaryAttr parameterMap) {
  if (parameterMap.empty())
    return operation->emitOpError("requires at least one target parameter mapping");
  llvm::StringSet<> roles;
  for (NamedAttribute mapping : parameterMap) {
    auto role = dyn_cast<StringAttr>(mapping.getValue());
    if (!role || role.getValue().empty() ||
        !roles.insert(role.getValue()).second)
      return operation->emitOpError(
          "target parameter mappings require unique canonical roles");
  }
  return success();
}

} // namespace intent::target
