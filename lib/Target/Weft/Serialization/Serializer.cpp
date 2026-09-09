#include "Intent/Target/Weft/Serialization/Serializer.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

mlir::LogicalResult intent::weft_provider::serializeProgram(
    mlir::ModuleOp program, std::string &source) {
  if (mlir::failed(mlir::verify(program))) return mlir::failure();
  llvm::raw_string_ostream output(source);
  program.print(output);
  output << '\n';
  return mlir::success();
}
