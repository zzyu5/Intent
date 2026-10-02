#ifndef INTENT_TARGET_TILELANG_IR_TILELANGOPS_H
#define INTENT_TARGET_TILELANG_IR_TILELANGOPS_H

#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Target/TileLang/IR/TileLangDialect.h"
#include "Intent/Target/TileLang/IR/TileLangTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Intent/Target/TileLang/IR/TileLangOps.h.inc"

namespace mlir::func { class FuncOp; }
namespace intent::tilelang {
// The native launch reads the unique kernel-owned thread declaration. Its
// selected value is bound only by the complete configuration table.
mlir::FailureOr<gpu::ParameterAttr> queryThreadParameter(mlir::func::FuncOp kernel);
}

#endif
