#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Visitors.h"

#include "Intent/Dialect/CPU/IR/CPUOpInterfaces.cpp.inc"

using namespace mlir;
using namespace intent::cpu;

FailureOr<SmallVector<Value>> RegionOpInterface::bindRegionArguments(
    Region &region, llvm::function_ref<ValueRange(RegionArgumentKind)> values) {
  auto schema = getRegionSchema(region);
  if (failed(schema)) return failure();
  SmallVector<Value> arguments;
  for (unsigned begin = 0; begin < schema->size();) {
    unsigned end = begin + 1;
    while (end < schema->size() && (*schema)[end].kind == (*schema)[begin].kind) ++end;
    ValueRange actual = values((*schema)[begin].kind);
    if (actual.size() != end - begin)
      return emitOpError("region argument binding does not match its declared group"), failure();
    llvm::append_range(arguments, actual);
    begin = end;
  }
  return arguments;
}

FailureOr<SmallVector<RegionArgumentSchema>>
intent::cpu::detail::regionArgumentSchema(Operation *operation, Region &region) {
  auto program = cast<RegionOpInterface>(operation);
  if (region.getParentOp() != operation || !llvm::hasSingleElement(region))
    return operation->emitOpError("region helper must have one owned block"), failure();
  struct Entry {
    Value prototype;
    RegionArgumentKind kind;
    std::optional<int64_t> sliceAxis;
  };
  SmallVector<Entry> entries;
  auto append = [&](ValueRange values, RegionArgumentKind kind,
                    std::optional<int64_t> axis = std::nullopt) {
    for (Value value : values) entries.push_back({value, kind, axis});
  };
  using K = RegionArgumentKind;
  if (&region == &program.getSummarize()) {
    append(program.getSources(), K::Sources, program.getAxis());
    append(program.getCaptures(), K::Captures);
    append(program.getIdentities(), K::Destinations);
  } else if (&region == &program.getCombine()) {
    append(program.getIdentities(), K::LeftSummary);
    append(program.getIdentities(), K::RightSummary);
    append(program.getIdentities(), K::Destinations);
  } else if (&region == program.getApplyRegion()) {
    append(program.getIdentities(), K::Summary);
    append(program.getInitialStates(), K::State);
    append(program.getInitialStates(), K::Destinations);
  } else if (&region == program.getEmitRegion()) {
    append(program.getSources(), K::Sources, program.getAxis());
    append(program.getInitialStates(), K::State);
    append(program.getCaptures(), K::Captures);
    auto axes = operation->getAttrOfType<DenseI64ArrayAttr>("output_axes");
    if (!axes || static_cast<size_t>(axes.size()) != program.getEmittedOutputs().size())
      return operation->emitOpError("each emitted output requires its source member axis"), failure();
    for (auto [output, axis] : llvm::zip(program.getEmittedOutputs(), axes.asArrayRef()))
      entries.push_back({output, K::Destinations, axis});
  } else {
    return operation->emitOpError("unrecognized region helper"), failure();
  }
  Block &body = region.front();
  if (body.getNumArguments() != entries.size())
    return operation->emitOpError("region helper argument groups are incomplete"), failure();
  SmallVector<RegionArgumentSchema> result;
  for (auto [argument, entry] : llvm::zip(body.getArguments(), entries))
    result.push_back({argument, entry.prototype, entry.kind, entry.sliceAxis});
  return result;
}

Block::BlockArgListType intent::cpu::detail::regionArguments(
    Operation *operation, Region &region, RegionArgumentKind kind) {
  auto schema = regionArgumentSchema(operation, region);
  assert(succeeded(schema) && "helper arguments require verified region schema");
  unsigned begin = 0;
  while (begin < schema->size() && (*schema)[begin].kind != kind) ++begin;
  unsigned end = begin;
  while (end < schema->size() && (*schema)[end].kind == kind) ++end;
  return region.front().getArguments().slice(begin, end - begin);
}

namespace {

Type slotType(Type type) {
  if (auto memory = dyn_cast<MemRefType>(type))
    return MemRefType::get(memory.getShape(), memory.getElementType());
  return MemRefType::get({}, type);
}

LogicalResult verifyRegionProgram(Operation *operation) {
  auto program = cast<RegionOpInterface>(operation);
  if (program.getSources().empty() || program.getIdentities().empty() ||
      (program.isScan() && (program.getInitialStates().empty() || program.getEmittedOutputs().empty())) ||
      program.getInitialValues().size() != program.getFinalDestinations().size())
    return operation->emitOpError("region source, identity, state and destination groups are incomplete");
  if (auto size = operation->getAttrOfType<IntegerAttr>("segment_size"))
    if (size.getInt() <= 0) return operation->emitOpError("region segment size must be positive");
  for (Value source : program.getSources()) {
    auto type = cast<MemRefType>(source.getType());
    if (program.getAxis() >= static_cast<uint64_t>(type.getRank()))
      return operation->emitOpError("region source must contain its explicit member axis");
  }
  auto fields = operation->getAttrOfType<ArrayAttr>("summary_fields");
  if (!fields || fields.size() != program.getIdentities().size())
    return operation->emitOpError("region product fields must name every typed summary component");
  if (program.isScan()) {
    auto fields = operation->getAttrOfType<ArrayAttr>("state_fields");
    if (!fields || fields.size() != program.getInitialStates().size())
      return operation->emitOpError("region product fields must name every typed state component");
  }
  for (auto [value, output] : llvm::zip(program.getInitialValues(), program.getFinalDestinations()))
    if (slotType(value.getType()) != output.getType())
      return operation->emitOpError("identity/state and destination types must agree");
  for (Region &region : operation->getRegions()) {
    auto schema = program.getRegionSchema(region);
    if (failed(schema)) return failure();
    if (region.front().empty() || !isa<RegionYieldOp>(region.front().getTerminator()))
      return operation->emitOpError("region helpers require region_yield");
    for (const auto &relation : *schema) {
      Type expected = relation.prototype.getType();
      if (relation.sliceAxis) {
        auto memory = dyn_cast<MemRefType>(expected);
        int64_t axis = *relation.sliceAxis;
        if (!memory || axis < 0 || axis >= memory.getRank())
          return operation->emitOpError("helper slice axis is outside its source rank");
        SmallVector<int64_t> shape(memory.getShape());
        shape[axis] = ShapedType::kDynamic;
        expected = MemRefType::get(shape, memory.getElementType(),
            StridedLayoutAttr::get(operation->getContext(), ShapedType::kDynamic,
                SmallVector<int64_t>(shape.size(), ShapedType::kDynamic)));
      } else if (relation.kind != RegionArgumentKind::Captures) {
        expected = slotType(expected);
      }
      if (relation.argument.getType() != expected)
        return operation->emitOpError("region helper argument type disagrees with its source/state/output schema");
    }
    bool invalid = false;
    region.walk([&](Operation *nested) {
      if (nested->getName().getDialectNamespace() == "intent") {
        nested->emitOpError("CPU region helper cannot retain canonical operations");
        invalid = true;
      }
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(nested)) {
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
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
              ((*schema)[argument.getArgNumber()].kind != RegionArgumentKind::Destinations ||
               isa<MemoryEffects::Free>(effect.getEffect()))) {
            nested->emitOpError("region helper inputs are read-only and destinations are caller-owned");
            invalid = true;
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
