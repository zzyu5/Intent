#ifndef INTENT_DIALECT_GPU_TRANSFORMS_TUNINGPROFILES_H
#define INTENT_DIALECT_GPU_TRANSFORMS_TUNINGPROFILES_H

#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace intent::gpu {

struct TuningProfileSource {
  llvm::StringRef name;
  std::string filename;
  llvm::ArrayRef<llvm::StringRef> columns;
};

class TuningProfiles {
public:
  using Row = llvm::SmallVector<int64_t, 7>;
  using Table = llvm::SmallVector<Row, 5>;

  static mlir::FailureOr<TuningProfiles>
  read(mlir::Location location, llvm::ArrayRef<TuningProfileSource> defaults,
       llvm::StringRef overrideFilename);

  mlir::FailureOr<llvm::ArrayRef<Row>>
  get(llvm::StringRef space, llvm::StringRef family,
      mlir::Location location) const;

private:
  struct Namespace {
    size_t width;
    llvm::StringMap<Table> families;
  };
  llvm::StringMap<Namespace> spaces;
};

} // namespace intent::gpu

#endif
