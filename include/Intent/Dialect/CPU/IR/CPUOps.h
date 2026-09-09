#ifndef INTENT_DIALECT_CPU_IR_CPUOPS_H
#define INTENT_DIALECT_CPU_IR_CPUOPS_H
#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#define GET_OP_CLASSES
#include "Intent/Dialect/CPU/IR/CPUOps.h.inc"
#endif
