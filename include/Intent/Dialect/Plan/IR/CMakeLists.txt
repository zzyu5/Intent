set(LLVM_TARGET_DEFINITIONS PlanDialect.td)
mlir_tablegen(PlanDialect.h.inc -gen-dialect-decls -dialect=intent_plan)
mlir_tablegen(PlanDialect.cpp.inc -gen-dialect-defs -dialect=intent_plan)

set(LLVM_TARGET_DEFINITIONS PlanOps.td)
mlir_tablegen(PlanOps.h.inc -gen-op-decls)
mlir_tablegen(PlanOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(IntentPlanIncGen)
