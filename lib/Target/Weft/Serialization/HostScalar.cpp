#include "Intent/Target/Weft/Serialization/HostScalar.h"
#include "Intent/Serialization/ScalarEmitters.h"

using namespace mlir;
namespace intent::weft_provider {
std::optional<CScalarType> hostScalarType(Type type) {
  if (type.isF32()) return CScalarType{"float", {}, {}};
  if (type.isF64()) return CScalarType{"double", {}, {}};
  if (type.isIndex()) return CScalarType{"int64_t", {}, {}};
  if (type.isInteger(1)) return CScalarType{"int", {}, {}};
  auto integer = dyn_cast<IntegerType>(type);
  if (!integer || !integer.isSignless() ||
      (integer.getWidth() != 8 && integer.getWidth() != 16 &&
       integer.getWidth() != 32 && integer.getWidth() != 64))
    return std::nullopt;
  return CScalarType{"int" + std::to_string(integer.getWidth()) + "_t", {}, {}};
}
namespace {
FailureOr<std::string> render(Operation *operation, ArrayRef<std::string> operands) {
  auto scalar = ScalarOperation::read(operation);
  if (failed(scalar)) return failure();
  return emitCScalar(*scalar, operands, hostScalarType);
}
const auto &emitters() {
  static const auto table = scalarExpressionEmitters(render);
  return table;
}
} // namespace

FailureOr<std::string> emitHostScalar(Operation *operation, ArrayRef<std::string> operands) {
  return emitScalarExpression(emitters(), operation, operands);
}
LogicalResult verifyHostScalar(Operation *operation) { return emitters().verify(operation); }

} // namespace intent::weft_provider
