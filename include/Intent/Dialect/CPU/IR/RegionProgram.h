#ifndef INTENT_DIALECT_CPU_IR_REGIONPROGRAM_H
#define INTENT_DIALECT_CPU_IR_REGIONPROGRAM_H

#include "Intent/Dialect/CPU/IR/CPUOps.h"

namespace intent::cpu {

// Both operations keep source, identity, initial-state, capture and destination
// partitions in this order. Product fields are flattened into typed slots.
class RegionProgram {
public:
  explicit RegionProgram(mlir::Operation *operation) : operation(operation) {}
  bool isScan() const { return mlir::isa<RegionScanOp>(operation); }
  int64_t count(llvm::StringRef name) const {
    return operation->getAttrOfType<mlir::IntegerAttr>(name).getInt();
  }
  mlir::OperandRange sources() const { return operation->getOperands().take_front(count("source_count")); }
  mlir::OperandRange identities() const {
    return operation->getOperands().slice(count("source_count"), count("identity_count"));
  }
  mlir::OperandRange initialState() const {
    return operation->getOperands().slice(count("source_count") + count("identity_count"), count("state_count"));
  }
  mlir::OperandRange captures() const {
    return operation->getOperands().slice(count("source_count") + count("identity_count") + count("state_count"), count("capture_count"));
  }
  mlir::OperandRange outputs() const {
    return operation->getOperands().take_back(count("output_count") + (isScan() ? count("state_count") : count("identity_count")));
  }
  mlir::Region &summarize() const { return operation->getRegion(0); }
  mlir::Region &combine() const { return operation->getRegion(1); }
  mlir::Region &apply() const { return operation->getRegion(2); }
  mlir::Region &emit() const { return operation->getRegion(3); }
  mlir::Operation *getOperation() const { return operation; }

private:
  mlir::Operation *operation;
};

}
#endif
