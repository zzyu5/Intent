#include "Intent/Dialect/Intent/IR/IntentDialect.h"
#include "Intent/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  intent::registerIntentPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<intent::IntentDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Intent kernel optimizer\n", registry));
}
