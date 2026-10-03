#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_ACCESSFORMSELECTION_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_ACCESS_ACCESSFORMSELECTION_H

#include "Intent/Dialect/GPU/Transforms/Configuration/TuningProfiles.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace intent::cutile {

// Lazy bindings are shared by all access forms in a single native phase.
class AccessFormSelection {
public:
  AccessFormSelection(mlir::func::FuncOp kernel,
                      const gpu::TuningProfiles &profiles)
      : kernel(kernel), profiles(profiles) {}
  mlir::FailureOr<mlir::Value> loadFormCondition();
  mlir::FailureOr<mlir::Value> tmaCondition();
  mlir::FailureOr<mlir::Value> loadPolicyValue();

private:
  mlir::FailureOr<mlir::Value> accessFormValue();
  mlir::func::FuncOp kernel;
  const gpu::TuningProfiles &profiles;
  mlir::Value accessForm;
  mlir::Value loadPolicy;
  mlir::Value preferTileLoads;
  mlir::Value allowNativeTMA;
};

} // namespace intent::cutile

#endif
