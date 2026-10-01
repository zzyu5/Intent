#ifndef INTENT_DIALECT_GPU_TRANSFORMS_TUNINGPROFILES_H
#define INTENT_DIALECT_GPU_TRANSFORMS_TUNINGPROFILES_H

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace intent::gpu {

struct TuningProfileSchema {
  llvm::StringRef space;
  llvm::ArrayRef<llvm::StringRef> columns;
};

struct TuningProfileSource {
  TuningProfileSchema schema;
  std::string filename;
};

const TuningProfileSchema &sharedTuningProfileSchema();

class TuningProfiles {
public:
  using Row = llvm::SmallVector<int64_t, 7>;
  using Table = llvm::SmallVector<Row, 5>;

  static mlir::FailureOr<TuningProfiles>
  read(mlir::Location location, llvm::ArrayRef<TuningProfileSource> defaults,
       llvm::StringRef overrideFilename);

  static mlir::FailureOr<TuningProfiles> from(mlir::ModuleOp module);
  void attach(mlir::ModuleOp module) const;

  mlir::FailureOr<Table>
  get(const TuningProfileSchema &schema, llvm::StringRef family,
      mlir::Location location) const;

private:
  explicit TuningProfiles(TuningProfilesAttr profiles) : profiles(profiles) {}
  TuningProfilesAttr profiles;
};

} // namespace intent::gpu

#endif
