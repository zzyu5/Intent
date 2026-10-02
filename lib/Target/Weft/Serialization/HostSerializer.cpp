#include "Intent/Target/Weft/Serialization/Serializer.h"
#include "Intent/Target/Weft/IR/HostScalar.h"
#include "Intent/Serialization/Source.h"
#include "Intent/Serialization/MemoryDescriptor.h"
#include "Intent/Dialect/CPU/IR/CPUOps.h"
#include "TaskABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
namespace intent::weft_provider {

FailureOr<NativeABI> queryHostABI(func::FuncOp function) {
  return queryNativeABI(function, getPublicInterface(function), [](Type type) -> FailureOr<Type> {
    if (type.isIndex()) return IntegerType::get(type.getContext(), 64);
    if (type.isInteger(1)) return IntegerType::get(type.getContext(), 32);
    if (type.isSignlessInteger() || type.isF32() || type.isF64()) return type;
    return failure();
  });
}

namespace {

using Memory = MemoryDescriptor;

class HostSerializer : public SourceEmitter {
public:
  HostSerializer(ModuleOp module, llvm::raw_ostream &out)
      : SourceEmitter(out, 2), module(module) {}

  LogicalResult run() {
    output << "#include <stdint.h>\n#include <stddef.h>\n#include <stdlib.h>\n#include <math.h>\n";
    for (auto function : module.getOps<func::FuncOp>()) {
      if (!function.isExternal()) continue;
      output << "extern void " << function.getName() << "(";
      for (auto [i, type] : llvm::enumerate(function.getArgumentTypes())) {
        if (i) output << ", ";
        output << (isa<MemRefType>(type) ? "void *" : scalarType(type));
      }
      output << ");\n";
    }
    for (auto function : module.getOps<func::FuncOp>()) {
      if (function.isExternal()) continue;
      auto abi = queryHostABI(function);
      if (mlir::failed(abi)) return failure();
      ScopedValues local(*this);
      memories.clear();
      output << "void " << function.getName() << "(";
      bool first = true;
      auto parameter = [&](const std::string &type, const std::string &name) {
        if (!first) output << ", ";
        output << type << " " << name; first = false;
      };
      for (auto [i, argument] : llvm::enumerate(function.getArguments())) {
        std::string name = "a" + std::to_string(i);
        if (auto type = dyn_cast<MemRefType>(argument.getType())) {
          Memory memory{name, "0", {}, {}};
          SmallVector<int64_t> strides;
          int64_t offset;
          if (mlir::failed(type.getStridesAndOffset(strides, offset)))
            return function.emitError("host entry requires a strided memory descriptor");
          for (int64_t size : type.getShape())
            memory.sizes.push_back(ShapedType::isDynamic(size) ? "" : std::to_string(size));
          for (int64_t stride : strides)
            memory.strides.push_back(ShapedType::isDynamic(stride) ? "" : std::to_string(stride));
          memories[argument] = std::move(memory);
        }
        else values[argument] = name;
      }
      for (const NativeSlot &slot : abi->slots) {
        std::string name = slot.name();
        parameter(slot.role == NativeSlotRole::Pointer
            ? scalarType(slot.element) + " *" : scalarType(slot.carrier), name);
        if (slot.axis) {
          auto &memory = memories.find(function.getArgument(slot.parameter))->second;
          auto &entries = slot.role == NativeSlotRole::Extent ? memory.sizes : memory.strides;
          entries[*slot.axis] = name;
        }
      }
      output << ") {\n";
      indent = 1;
      if (mlir::failed(block(function.front()))) return failure();
      output << "}\n";
    }
    return failure(hasFailed());
  }

private:
  std::string scalarType(Type type) { return hostScalarType(type)->name; }
  std::string fresh() { return newName(); }
  std::string value(Value input) { return valueString(input); }
  std::string bound(OpFoldResult input) {
    return isa<Attribute>(input) ? std::to_string(cast<IntegerAttr>(cast<Attribute>(input)).getInt())
                                 : value(cast<Value>(input));
  }
  std::string address(Value memory, ValueRange indices) {
    auto &entry = memories.at(memory);
    std::string result = pointer(memory);
    for (auto [index, stride] : llvm::zip(indices, entry.strides))
      result += " + (" + value(index) + ") * (" + stride + ")";
    return "(" + result + ")";
  }
  std::string pointer(Value memory) {
    const auto &entry = memories.at(memory);
    return entry.offset == "0" ? entry.base : "(" + entry.base + " + (" + entry.offset + "))";
  }

  SmallVector<std::string> components(Value input) {
    if (isa<MemRefType>(input.getType())) return memories.at(input).components();
    return {value(input)};
  }
  SmallVector<std::string> componentTypes(Type type) {
    if (auto memory = dyn_cast<MemRefType>(type)) {
      SmallVector<std::string> types{scalarType(memory.getElementType()) + " *"};
      types.append(1 + 2 * memory.getRank(), "int64_t");
      return types;
    }
    return {scalarType(type)};
  }
  void declare(Value input, Value initial = {}) {
    SmallVector<std::string> fields;
    auto sources = initial ? components(initial) : SmallVector<std::string>{};
    for (auto [index, type] : llvm::enumerate(componentTypes(input.getType()))) {
      fields.push_back(fresh());
      line(type + " " + fields.back() + (initial ? " = " + sources[index] : "") + ";");
    }
    if (auto memory = dyn_cast<MemRefType>(input.getType()))
      memories[input] = Memory::fromComponents(memory, fields);
    else values[input] = fields.front();
  }
  void alias(Value result, Value source) {
    if (isa<MemRefType>(result.getType())) memories[result] = memories.at(source);
    else values[result] = value(source);
  }
  void transfer(ValueRange from, ValueRange to) {
    SmallVector<std::pair<std::string, std::string>> assignments;
    for (auto [source, target] : llvm::zip(from, to)) {
      auto sources = components(source), targets = components(target);
      for (auto [index, type] : llvm::enumerate(componentTypes(source.getType()))) {
        auto temporary = fresh();
        line(type + " " + temporary + " = " + sources[index] + ";");
        assignments.emplace_back(targets[index], temporary);
      }
    }
    for (auto &[destination, temporary] : assignments) line(destination + " = " + temporary + ";");
  }

  LogicalResult view(Operation *operation) {
    if (auto cast = dyn_cast<memref::CastOp>(operation)) {
      if (!isa<MemRefType>(cast.getSource().getType()) ||
          !isa<MemRefType>(cast.getResult().getType()))
        return cast.emitError("native host cast requires ranked memory descriptors");
      Memory source = memories.at(cast.getSource());
      memories[cast.getResult()] = std::move(source);
      return success();
    }
    if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(operation)) {
      const Memory source = memories.at(metadata.getSource());
      memories[metadata.getBaseBuffer()] = Memory{source.base, "0", {}, {}};
      values[metadata.getOffset()] = source.offset;
      for (auto [result, size] : llvm::zip(metadata.getSizes(), source.sizes)) values[result] = size;
      for (auto [result, stride] : llvm::zip(metadata.getStrides(), source.strides)) values[result] = stride;
      return success();
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(operation)) {
      if (!isa<MemRefType>(reinterpret.getSource().getType()))
        return reinterpret.emitError("native host reinterpretation requires a ranked source descriptor");
      // Reinterpretation replaces metadata relative to the storage base. It
      // must not add the source view's offset to the replacement offset.
      Memory target{memories.at(reinterpret.getSource()).base,
                    bound(reinterpret.getMixedOffsets().front()), {}, {}};
      for (OpFoldResult size : reinterpret.getMixedSizes()) target.sizes.push_back(bound(size));
      for (OpFoldResult stride : reinterpret.getMixedStrides()) target.strides.push_back(bound(stride));
      memories[reinterpret.getResult()] = std::move(target);
      return success();
    }
    auto subview = cast<memref::SubViewOp>(operation);
    const Memory source = memories.at(subview.getSource());
    Memory target{source.base, source.offset, {}, {}};
    auto offsets = subview.getMixedOffsets(), sizes = subview.getMixedSizes(), steps = subview.getMixedStrides();
    auto dropped = subview.getDroppedDims();
    for (unsigned axis = 0; axis < offsets.size(); ++axis) {
      target.offset += " + (" + bound(offsets[axis]) + ") * (" + source.strides[axis] + ")";
      if (!dropped.test(axis)) {
        target.sizes.push_back(bound(sizes[axis]));
        target.strides.push_back("(" + source.strides[axis] + ") * (" + bound(steps[axis]) + ")");
      }
    }
    target.offset = "(" + target.offset + ")";
    memories[subview.getResult()] = std::move(target);
    return success();
  }
  void bind(Value result, const std::string &expression) {
    std::string name = fresh();
    line(scalarType(result.getType()) + " " + name + " = " + expression + ";");
    values[result] = name;
  }
  LogicalResult block(Block &body) {
    for (auto &operation : body)
      if (mlir::failed(emit(&operation))) return failure();
    return success();
  }
  LogicalResult emit(Operation *operation) {
    if (isa<scf::YieldOp, scf::ConditionOp, scf::ReduceOp>(operation)) return success();
    if (isa<func::ReturnOp>(operation)) { line("return;"); return success(); }
    if (isStandardScalarOperation(operation)) {
      SmallVector<std::string> operands;
      for (Value input : operation->getOperands()) operands.push_back(value(input));
      auto expression = emitHostScalar(operation, operands);
      if (mlir::failed(expression)) return failure();
      if (isa<arith::ConstantOp>(operation)) values[operation->getResult(0)] = *expression;
      else bind(operation->getResult(0), *expression);
      return success();
    }
    if (auto dim = dyn_cast<memref::DimOp>(operation)) {
      auto axis = dim.getConstantIndex();
      if (!axis) return dim.emitError("host dimension must name a constant axis");
      values[dim.getResult()] = memories.at(dim.getSource()).sizes[*axis];
      return success();
    }
    if (auto pointer = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(operation)) {
      bind(pointer.getResult(), "(int64_t)(uintptr_t)" + memories.at(pointer.getSource()).base);
      return success();
    }
    if (isa<memref::CastOp, memref::ExtractStridedMetadataOp,
            memref::ReinterpretCastOp, memref::SubViewOp>(operation)) return view(operation);
    if (isa<memref::AllocOp, memref::AllocaOp>(operation)) {
      auto type = cast<MemRefType>(operation->getResult(0).getType());
      if (!type.getLayout().isIdentity() || type.getMemorySpaceAsInt() != 0)
        return operation->emitError("native host allocation requires contiguous host storage");
      std::string name = fresh(), elements = "1";
      Memory memory{name, "0", {}, {}};
      unsigned dynamic = 0;
      for (int64_t size : type.getShape())
        memory.sizes.push_back(ShapedType::isDynamic(size) ? value(operation->getOperand(dynamic++)) : std::to_string(size));
      memory.strides.resize(type.getRank());
      for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
        memory.strides[axis] = elements;
        elements = "(" + elements + ") * (" + memory.sizes[axis] + ")";
      }
      auto element = scalarType(type.getElementType());
      int64_t alignment = 16;
      if (auto attr = operation->getAttrOfType<IntegerAttr>("alignment")) alignment = std::max(alignment, attr.getInt());
      if (isa<memref::AllocaOp>(operation))
        line("_Alignas(" + std::to_string(alignment) + ") " + element + " " + name + "[" + elements + "];");
      else {
        std::string bytes = "sizeof(" + element + ") * (" + elements + ")";
        line(element + " *" + name + " = aligned_alloc(" + std::to_string(alignment) + ", ((" + bytes +
            " + " + std::to_string(alignment - 1) + ") / " + std::to_string(alignment) + ") * " + std::to_string(alignment) + ");");
        line("if (!" + name + " && (" + elements + ")) abort();");
      }
      memories[operation->getResult(0)] = memory; return success();
    }
    if (auto dealloc = dyn_cast<memref::DeallocOp>(operation)) {
      line("free(" + memories.at(dealloc.getMemref()).base + ");"); return success();
    }
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      bind(load.getResult(), "*" + address(load.getMemref(), load.getIndices())); return success();
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      line("*" + address(store.getMemref(), store.getIndices()) + " = " + value(store.getValue()) + ";"); return success();
    }
    if (auto call = dyn_cast<cpu::InvokeOp>(operation)) {
      std::string text = call.getCallee().str() + "(";
      for (auto [i, argument] : llvm::enumerate(call.getOperands())) {
        if (i) text += ", ";
        text += isa<MemRefType>(argument.getType()) ? pointer(argument) : value(argument);
      }
      line(text + ");"); return success();
    }
    if (auto loop = dyn_cast<scf::ParallelOp>(operation)) {
      if (loop.getNumLoops() != 1 || loop.getNumResults()) return loop.emitError("native task loop requires one linear workset");
      auto caps = module->getAttrOfType<cpu::CapabilitiesAttr>("intent_cpu.capabilities");
      if (caps.getWorkers() != 1) line("#pragma omp parallel for num_threads(" + std::to_string(caps.getWorkers()) + ")");
      std::string iv = fresh(); values[loop.getInductionVars()[0]] = iv;
      line("for (int64_t " + iv + " = " + value(loop.getLowerBound()[0]) + "; " + iv + " < " +
          value(loop.getUpperBound()[0]) + "; " + iv + " += " + value(loop.getStep()[0]) + ") {");
      ++indent; if (mlir::failed(block(*loop.getBody()))) return failure(); --indent; line("}"); return success();
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      for (auto [argument, initial] : llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs())) declare(argument, initial);
      std::string iv = fresh(); values[loop.getInductionVar()] = iv;
      line("for (int64_t " + iv + " = " + value(loop.getLowerBound()) + "; " + iv + " < " + value(loop.getUpperBound()) +
          "; " + iv + " += " + value(loop.getStep()) + ") {");
      ++indent; if (mlir::failed(block(*loop.getBody()))) return failure();
      transfer(loop.getBody()->getTerminator()->getOperands(), loop.getRegionIterArgs());
      --indent; line("}");
      for (auto [result, argument] : llvm::zip(loop.getResults(), loop.getRegionIterArgs())) alias(result, argument);
      return success();
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      Block &before = loop.getBefore().front(), &after = loop.getAfter().front();
      for (auto [argument, initial] : llvm::zip(before.getArguments(), loop.getInits())) declare(argument, initial);
      for (Value argument : after.getArguments()) declare(argument);
      line("while (1) {"); ++indent;
      if (mlir::failed(block(before))) return failure();
      auto condition = cast<scf::ConditionOp>(before.getTerminator());
      transfer(condition.getArgs(), after.getArguments());
      line("if (!(" + value(condition.getCondition()) + ")) break;");
      if (mlir::failed(block(after))) return failure();
      transfer(after.getTerminator()->getOperands(), before.getArguments());
      --indent; line("}");
      for (auto [result, argument] : llvm::zip(loop.getResults(), after.getArguments())) alias(result, argument);
      return success();
    }
    if (auto condition = dyn_cast<scf::IfOp>(operation)) {
      for (Value result : condition.getResults()) declare(result);
      auto branch = [&](Block &body) {
        if (mlir::failed(block(body))) return failure();
        transfer(body.getTerminator()->getOperands(), condition.getResults());
        return success();
      };
      line("if (" + value(condition.getCondition()) + ") {");
      ++indent; if (mlir::failed(branch(*condition.thenBlock()))) return failure(); --indent;
      if (!condition.getElseRegion().empty()) {
        line("} else {"); ++indent;
        if (mlir::failed(branch(*condition.elseBlock()))) return failure(); --indent;
      }
      line("}"); return success();
    }
    return operation->emitError("operation has no legalized native host serialization: ") << operation->getName();
  }

  ModuleOp module;
  llvm::DenseMap<Value, Memory> memories;
};

}

LogicalResult serializeHostProgram(ModuleOp program, std::string &source) {
  if (mlir::failed(verify(program)) || mlir::failed(verifyHostScalarOperations(program))) return failure();
  llvm::raw_string_ostream stream(source);
  return HostSerializer(program, stream).run();
}

}
