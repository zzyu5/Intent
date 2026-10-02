#ifndef INTENT_TARGET_CUTILE_TRANSFORMS_NATIVEACCESS_H
#define INTENT_TARGET_CUTILE_TRANSFORMS_NATIVEACCESS_H
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/Transforms/TuningProfiles.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

namespace intent::cutile {

// The input operations stay live with their original def-use throughout native
// construction. Replacements become visible together, after every current-GPU
// fact has been consumed. Nothing from this transaction survives the phase.
class NativeFormRewriter {
public:
  void replace(mlir::Operation *operation, mlir::ValueRange values);
  void erase(mlir::Operation *operation);
  void commit();
private:
  struct Replacement {
    mlir::Operation *operation;
    llvm::SmallVector<mlir::Value> values;
  };
  llvm::SmallVector<Replacement> replacements;
};

// The same operation classification applies before and after native formation.
// A caller observes its current IR; this is never retained across a phase.
struct NativeProgramFeatures {
  bool matrixCompute = false;
  bool occupancySensitive = false;
  void observe(mlir::Operation *operation);
};
NativeProgramFeatures queryNativeProgramFeatures(mlir::func::FuncOp kernel);

struct NativeProgramInputs {
  llvm::SmallVector<gpu::LoadOp> loads;
  llvm::SmallVector<gpu::GatherOp> gathers;
  llvm::SmallVector<gpu::StoreOp> stores;
  llvm::SmallVector<gpu::AtomicRMWOp> atomics;
  llvm::SmallVector<gpu::ContractOp> contracts;
  llvm::SmallVector<gpu::ScaledContractOp> scaledContracts;
  llvm::SmallVector<gpu::ReduceOp> reductions;
  llvm::SmallVector<gpu::HistogramOp> histograms;
  llvm::SmallVector<gpu::ScanOp> scans;
  llvm::SmallVector<gpu::AssumeInBoundsOp> assumptions;
};

struct NativeTileAxisPlan {
  llvm::SmallVector<unsigned> computationAxes;
  mlir::Value scalarIndex;
  int64_t divisor = 0;
  mlir::OpFoldResult modulus;
  llvm::SmallVector<std::pair<gpu::MakeRangeOp, int64_t>> ranges;
  llvm::SmallVector<std::pair<mlir::Value, int64_t>> offsets;
  bool originInBounds = false;
};

struct NativeTileAccessPlan {
  gpu::FragmentType resourceType;
  gpu::FragmentType packedType;
  mlir::ArrayAttr resourceToPacked;
  mlir::ArrayAttr packedToResource;
  llvm::SmallVector<NativeTileAxisPlan> axes;
  llvm::SmallVector<int64_t> toComputation;
  llvm::SmallVector<int64_t> toResource;
};


// Only coordinate decomposition and legality live here. Layout packing, tile
// indices and runtime guards remain cuTile constructs and are not shared facts.
mlir::FailureOr<NativeTileAccessPlan> analyzeNativeTileAccess(
    gpu::AccessOpInterface access, mlir::func::FuncOp kernel,
    const gpu::PhysicalAccessBoundsFact &accessBounds);
mlir::FailureOr<unsigned> nativeAccessRangeAxis(gpu::AccessOpInterface access,
    unsigned coordinateIndex, gpu::MakeRangeOp range);
mlir::Value uniformScalarFill(mlir::Value fill);
bool isUnitExtent(mlir::Attribute attribute);
bool isProvably(mlir::Value value, int64_t expected);
mlir::Value stripIndexIdentities(mlir::Value value);
mlir::FailureOr<llvm::SmallVector<mlir::Value>> uniformAlignmentFactors(
    mlir::Value value, mlir::Value divisor, unsigned depth = 0);
bool isAlignedPeriodicTile(mlir::Value start, mlir::Value extent, mlir::Value period);
mlir::scf::ForOp completeAlignedTileLoop(mlir::Value start, mlir::Value extent);
bool scalarCoordinatesInView(mlir::ValueRange coordinates, gpu::ViewType view,
                             mlir::func::FuncOp kernel);
mlir::Type withElementType(mlir::Type type, mlir::Type elementType);
mlir::Value materializeFullTileCondition(mlir::OpBuilder &builder,
    mlir::Location location, mlir::Value resource, gpu::FragmentType tile);
bool supportsE8M0ScaledMMA(gpu::CapabilitiesAttr capabilities);
mlir::LogicalResult formNativeAccesses(mlir::func::FuncOp kernel,
    const gpu::TuningProfiles &profiles, const NativeProgramInputs &inputs,
    bool matrixCompute, NativeFormRewriter &rewriter);
mlir::LogicalResult formComputePrimitives(mlir::func::FuncOp kernel,
    const NativeProgramInputs &inputs, NativeFormRewriter &rewriter);
} // namespace intent::cutile
#endif
