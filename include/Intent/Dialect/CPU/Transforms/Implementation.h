#ifndef INTENT_DIALECT_CPU_TRANSFORMS_IMPLEMENTATION_H
#define INTENT_DIALECT_CPU_TRANSFORMS_IMPLEMENTATION_H

#include "Intent/Dialect/CPU/IR/CPUAttrs.h"
#include "Intent/Dialect/CPU/Transforms/Configuration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <array>
#include <functional>
#include <optional>
#include <string>
#include <variant>

namespace intent::cpu {

enum class InputReuse { Group, Consumers };

// Storage order is [panel, unsplit source axes..., lane within panel]. A group
// supply covers one compute group; consumer reuse preserves a source snapshot.
struct InputRequirement {
  unsigned operand;
  mlir::Type elementType;
  unsigned panelAxis;
  int64_t panelSize;
  int64_t alignment;
  InputReuse reuse;
  int64_t windowAlignment; // Required panel-axis origin multiple for non-singleton windows.
};

std::optional<std::string> checkInputRequirement(
    mlir::Value source, mlir::Value element, const InputRequirement &requirement);
std::optional<std::string> checkInputRequirements(
    mlir::linalg::GenericOp operation, llvm::ArrayRef<InputRequirement> requirements);

struct InputSupply {
  unsigned operand;
  unsigned panelAxis;
  int64_t panelSize;
  mlir::Value storage;
  llvm::SmallVector<mlir::Value> begins; // Logical source coordinates of the supplied window.
};

struct ContractionTile {
  mlir::Value lhs, rhs, output, initial;
  mlir::Value mBegin, mCount, nBegin, nCount, kBegin, depth;
  bool first;
  llvm::ArrayRef<InputSupply> inputs;
};

struct ContractionRequirements {
  bool completePrivateInitialization = false;
  bool staticReductionExtent = false;
  bool staticParallelExtent = false;
  std::array<bool, 2> unitInnerStride{false, false};

  bool acceptsInputLayout(unsigned operand, mlir::MemRefType type) const;
};

struct ImplementationParameterDomain {
  bool powerOfTwo = false;
  std::optional<int64_t> maximum;
  int64_t laneBits = 0;
  llvm::SmallVector<int64_t, 2> values;
};

struct ImplementationParameter {
  struct Local { llvm::StringRef name; };
  struct Constant { int64_t value; };
  enum class Axis { TileM, TileN };
  struct Minimum { int64_t limit; llvm::SmallVector<Axis, 2> axes; };

  llvm::StringRef name;
  std::variant<Local, Constant, Minimum> source;
  ImplementationParameterDomain domain;

  static ImplementationParameter local(llvm::StringRef name, ImplementationParameterDomain domain = {});
  static ImplementationParameter alias(llvm::StringRef name, llvm::StringRef source);
  static ImplementationParameter constant(llvm::StringRef name, int64_t value);
  static ImplementationParameter minimum(llvm::StringRef name, int64_t limit, llvm::ArrayRef<Axis> axes);
};

struct Implementation {
  llvm::StringRef name;
  std::function<bool(mlir::Operation *)> applicable;
  std::function<std::optional<std::string>(mlir::Operation *, CapabilitiesAttr,
                                         const Configuration &)> check;
  llvm::SmallVector<ImplementationParameter, 5> parameters;
  std::function<mlir::LogicalResult(mlir::OpBuilder &, mlir::linalg::GenericOp,
      const ContractionTile &, ConfigurationAttr, ImplementationAttr)> formTile;
  std::function<mlir::FailureOr<llvm::SmallVector<mlir::Value>>(
      mlir::OpBuilder &, mlir::Operation *, mlir::ValueRange, int64_t &)> expand;
  ContractionRequirements contraction;
  std::function<int64_t(ImplementationAttr)> parallelWindow;
  bool requiresMatrixI8I32 = false;
  std::function<llvm::SmallVector<InputRequirement>(mlir::linalg::GenericOp,
      ConfigurationAttr, ImplementationAttr)> inputs;
  // Leading parallel rows retained together inside one independent work item.
  std::function<int64_t(mlir::linalg::GenericOp, ImplementationAttr)> worksetRows;
  std::function<std::optional<std::string>(mlir::DictionaryAttr, CapabilitiesAttr)> parameterRelations;

  // A read-only query of the selected implementation's surrounding supply.
  // An empty result requires no preparation; it is not a performance estimate.
  llvm::SmallVector<InputRequirement> inputRequirements(
      mlir::linalg::GenericOp operation, ConfigurationAttr configuration,
      ImplementationAttr binding) const;
};

class ImplementationRegistry {
public:
  std::function<llvm::StringRef(mlir::func::FuncOp)> profile;
  void addProfile(llvm::StringRef name,
                  llvm::ArrayRef<llvm::StringRef> localParameters);
  std::optional<llvm::ArrayRef<llvm::StringRef>>
  profileParameters(llvm::StringRef name) const;
  void add(Implementation implementation) { implementations.push_back(std::move(implementation)); }
  mlir::FailureOr<const Implementation *> lookup(mlir::Operation *operation) const;
  // Validate retained bindings without rerunning pre-expansion applicability.
  mlir::FailureOr<const Implementation *> verifyBinding(
      ImplementationAttr binding, CapabilitiesAttr capabilities,
      ConfigurationAttr configuration, mlir::Operation *diagnostic) const;
  mlir::LogicalResult verifyBindings(mlir::ModuleOp module) const;
  llvm::SmallVector<llvm::SmallVector<ImplementationAttr>> candidates(
      mlir::func::FuncOp function, CapabilitiesAttr capabilities,
      const Configuration &configuration,
      llvm::function_ref<void(mlir::Operation *, llvm::StringRef, llvm::StringRef)> rejected) const;
  mlir::LogicalResult bind(mlir::func::FuncOp function, CapabilitiesAttr capabilities,
                           const Configuration &configuration,
                           llvm::ArrayRef<ImplementationAttr> bindings) const;

private:
  struct ProfileSchema {
    llvm::StringRef name;
    llvm::SmallVector<llvm::StringRef> localParameters;
  };
  llvm::SmallVector<ProfileSchema> profiles;
  llvm::SmallVector<Implementation, 0> implementations;
};

/// Resolve a provider from this compiler context. Missing registration and an
/// absent/unknown provider are errors, never a request for a default registry.
mlir::FailureOr<const ImplementationRegistry *>
lookupImplementationProvider(mlir::Operation *operation, llvm::StringRef provider);
mlir::LogicalResult verifyImplementationBindings(mlir::ModuleOp module, llvm::StringRef provider);

bool needsImplementation(mlir::Operation *operation);
int64_t implementationParameter(ImplementationAttr binding, llvm::StringRef name);

}
#endif
