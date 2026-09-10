#include "Intent/Dialect/CPU/IR/RegionProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Visitors.h"

using namespace mlir;
using namespace intent::cpu;

namespace {

LogicalResult verifyRegionProgram(Operation *operation) {
  RegionProgram program(operation);
  int64_t sources = program.count("source_count"), identities = program.count("identity_count");
  int64_t states = program.count("state_count"), outputs = program.count("output_count");
  int64_t captures = program.count("capture_count"), axis = program.count("axis");
  if (sources <= 0 || identities <= 0 || states < 0 || outputs < 0 || captures < 0 || axis < 0 ||
      (!program.isScan() && (states || outputs)) ||
      (program.isScan() && (!states || !outputs)) ||
      operation->getNumOperands() != sources + identities + states + captures + outputs + (program.isScan() ? states : identities))
    return operation->emitOpError("region source, identity, state, capture and destination partitions are incomplete");
  if (auto size = operation->getAttrOfType<IntegerAttr>("segment_size"))
    if (size.getInt() <= 0) return operation->emitOpError("region segment size must be positive");
  for (Value source : program.sources()) {
    auto memory = dyn_cast<MemRefType>(source.getType());
    if (!memory || axis >= memory.getRank())
      return operation->emitOpError("region source must contain its explicit member axis");
  }
  for (Value output : program.outputs())
    if (!isa<MemRefType>(output.getType()))
      return operation->emitOpError("region results require explicit destination storage");
  auto fields = operation->getAttrOfType<ArrayAttr>("summary_fields");
  auto stateFields = operation->getAttrOfType<ArrayAttr>("state_fields");
  if (!fields || fields.size() != static_cast<size_t>(identities) || !stateFields || stateFields.size() != static_cast<size_t>(states))
    return operation->emitOpError("region product fields must name each typed summary/state component");
  auto outputAxes = operation->getAttrOfType<DenseI64ArrayAttr>("output_axes");
  if (!outputAxes || outputAxes.size() != outputs)
    return operation->emitOpError("each emitted output requires its source member axis");
  for (auto [output, memberAxis] : llvm::zip(program.outputs().take_front(outputs), outputAxes.asArrayRef()))
    if (memberAxis < 0 || memberAxis >= cast<MemRefType>(output.getType()).getRank())
      return operation->emitOpError("emitted output member axis is outside its rank");
  auto slotType = [](Type type) {
    if (auto memory = dyn_cast<MemRefType>(type)) return MemRefType::get(memory.getShape(), memory.getElementType());
    return MemRefType::get({}, type);
  };
  SmallVector<Type> sourceTypes, summaryTypes, stateTypes, captureTypes, outputTypes;
  auto sliceType = [&](Type type, int64_t axis) {
    auto memory = cast<MemRefType>(type);
    SmallVector<int64_t> shape(memory.getShape()); shape[axis] = ShapedType::kDynamic;
    return MemRefType::get(shape, memory.getElementType(),
        StridedLayoutAttr::get(operation->getContext(), ShapedType::kDynamic, SmallVector<int64_t>(shape.size(), ShapedType::kDynamic)));
  };
  for (Value source : program.sources()) sourceTypes.push_back(sliceType(source.getType(), axis));
  for (Value identity : program.identities()) summaryTypes.push_back(slotType(identity.getType()));
  for (Value state : program.initialState()) stateTypes.push_back(slotType(state.getType()));
  for (Value capture : program.captures()) captureTypes.push_back(capture.getType());
  for (auto [output, memberAxis] : llvm::zip(program.outputs().take_front(outputs), outputAxes.asArrayRef()))
    outputTypes.push_back(sliceType(output.getType(), memberAxis));
  auto initial = program.isScan() ? program.initialState() : program.identities();
  for (auto [value, output] : llvm::zip(initial, program.outputs().take_back(initial.size())))
    if (slotType(value.getType()) != output.getType())
      return operation->emitOpError("identity/state and destination types must agree");
  SmallVector<int64_t> arities{sources + captures + identities, 3 * identities};
  if (program.isScan()) {
    arities.push_back(identities + 2 * states);
    arities.push_back(sources + states + captures + outputs);
  }
  SmallVector<SmallVector<Type>> signatures(2);
  signatures[0] = sourceTypes;
  llvm::append_range(signatures[0], captureTypes); llvm::append_range(signatures[0], summaryTypes);
  for (unsigned i = 0; i < 3; ++i) llvm::append_range(signatures[1], summaryTypes);
  if (program.isScan()) {
    signatures.push_back(summaryTypes);
    llvm::append_range(signatures[2], stateTypes); llvm::append_range(signatures[2], stateTypes);
    signatures.push_back(sourceTypes);
    llvm::append_range(signatures[3], stateTypes); llvm::append_range(signatures[3], captureTypes);
    llvm::append_range(signatures[3], outputTypes);
  }
  SmallVector<int64_t> destinations{identities, identities};
  if (program.isScan()) { destinations.push_back(states); destinations.push_back(outputs); }
  for (auto [region, arity, signature, destinationCount] : llvm::zip(operation->getRegions(), arities, signatures, destinations)) {
    if (!llvm::hasSingleElement(region) || region.front().getNumArguments() != arity ||
        region.front().empty() || !isa<RegionYieldOp>(region.front().getTerminator()))
      return operation->emitOpError("region helpers require explicit typed arguments, destinations and region_yield");
    for (auto [argument, expected] : llvm::zip(region.front().getArguments(), signature))
      if (argument.getType() != expected) return operation->emitOpError("region helper argument type disagrees with its source/state/output schema");
    bool invalid = false;
    region.walk([&](Operation *nested) {
      if (nested->getName().getDialectNamespace() == "intent") {
        nested->emitOpError("CPU region helper cannot retain canonical operations"); invalid = true;
      }
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(nested)) {
        SmallVector<MemoryEffects::EffectInstance> instances; effects.getEffects(instances);
        for (auto &effect : instances) {
          if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect())) continue;
          Value root = effect.getValue();
          while (root) {
            if (auto view = root.getDefiningOp<memref::SubViewOp>()) root = view.getSource();
            else if (auto cast = root.getDefiningOp<memref::CastOp>()) root = cast.getSource();
            else break;
          }
          auto argument = dyn_cast_or_null<BlockArgument>(root);
          if (argument && argument.getOwner() == &region.front() &&
              (argument.getArgNumber() < arity - destinationCount || isa<MemoryEffects::Free>(effect.getEffect()))) {
            nested->emitOpError("region helper inputs are read-only and destinations are caller-owned"); invalid = true;
          }
        }
      }
    });
    if (invalid) return failure();
  }
  return success();
}

}

LogicalResult RegionFoldOp::verify() { return verifyRegionProgram(*this); }
LogicalResult RegionScanOp::verify() { return verifyRegionProgram(*this); }
