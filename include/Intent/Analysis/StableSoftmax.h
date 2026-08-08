#ifndef INTENT_ANALYSIS_STABLESOFTMAX_H
#define INTENT_ANALYSIS_STABLESOFTMAX_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <string>

namespace intent {

struct StableSoftmaxMatch {
  mlir::func::FuncOp entry;
  mlir::Operation *columnDomain;
  mlir::Operation *rowLoop;
  mlir::Operation *load;
  mlir::Operation *reduceMax;
  mlir::Operation *maxBroadcast;
  mlir::Operation *subtract;
  mlir::Operation *exponential;
  mlir::Operation *reduceSum;
  mlir::Operation *sumBroadcast;
  mlir::Operation *divide;
  mlir::Operation *store;
  int64_t inputValueID;
  int64_t outputValueID;
  std::string rowExtent;
  std::string columnExtent;
};

mlir::FailureOr<StableSoftmaxMatch>
matchStableSoftmax(mlir::ModuleOp module);

int64_t getIntentNodeID(mlir::Operation *operation);

} // namespace intent

#endif
