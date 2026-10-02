#include "Intent/Target/BangC/Passes.h"
#include "Intent/Dialect/DSA/IR/DSAOps.h"
#include "Intent/Serialization/NativeABI.h"
#include "Intent/Dialect/Intent/IR/CompileOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
using namespace mlir;
namespace intent::bangc {
namespace {
#include "../Runtime/TileImplementations.inc"
std::string ctype(Type type) {
  if (type.isF16()) return "half";
  if (type.isBF16()) return "uint16_t";
  if (type.isF32()) return "float";
  if (type.isF64()) return "double";
  if (type.isInteger(1)) return "bool";
  if (type.isIndex()) return "int64_t";
  return "int" + std::to_string(cast<IntegerType>(type).getWidth()) + "_t";
}
std::string unsignedType(Type type) {
  unsigned width = type.isIndex() ? 64 : cast<IntegerType>(type).getWidth();
  return "uint" + std::to_string(std::max(8u, width)) + "_t";
}
class Serializer {
public:
  Serializer(func::FuncOp function, llvm::raw_ostream &output) : function(function), out(output) {}
  LogicalResult emit(llvm::json::Object &metadata) {
    auto options = readCompileOptions(function);
    if (failed(options)) return failure();
    auto interface = getPublicInterface(function);
    auto publicMetadata = serializePublicInterface(function, interface);
    if (failed(publicMetadata)) return failure();
    auto nativeABI = queryNativeABI(function, interface, [](Type type) -> FailureOr<Type> {
      if (type.isIndex()) return IntegerType::get(type.getContext(), 64);
      if (type.isInteger(1) || type.isInteger(32) || type.isInteger(64))
        return IntegerType::get(type.getContext(), cast<IntegerType>(type).getWidth());
      if (type.isF32()) return type;
      return failure();
    });
    if (failed(nativeABI)) return failure();
    auto entry = function->getAttrOfType<dsa::EntryRequirementsAttr>(dsa::entryRequirementsAttr);
    auto requirements = queryNativeEntryRequirements(function, interface,
        entry.getDisjointOutputs(), [](unsigned, intent::ViewType) -> FailureOr<NativeViewRequirements> {
          return NativeViewRequirements{NativeViewLayout::Strided, 1};
        });
    if (failed(requirements)) return failure();
    auto config = function->getAttrOfType<dsa::ConfigurationAttr>("intent_dsa.configuration");
    for (const NativeSlot &slot : nativeABI->slots) {
      std::string name = slot.name();
      call.push_back(name);
      if (slot.role == NativeSlotRole::Pointer) {
        auto view = getPublicView(interface, slot.parameter);
        signature.push_back((view.getAccess() == 0 ? "const " : "") + ctype(slot.element) + " *" + name);
      } else signature.push_back(ctype(slot.carrier) + " " + name);
      if (!slot.axis) names[function.getArgument(slot.parameter)] = name;
    }
    llvm::json::Array fullExtents;
    for (int64_t value : function->getAttrOfType<DenseI64ArrayAttr>("intent_dsa.full_extent_dimensions").asArrayRef())
      fullExtents.push_back(value);
    metadata = llvm::json::Object{{"provider", "bangc"}, {"entry_name", function.getName()},
        {"compile_options", serializeCompileOptions(*options)},
        {"interface", std::move(*publicMetadata)}, {"entry", "intent_launch"},
        {"target", llvm::json::Object{{"family", "dsa"}, {"architecture", "mtp_372"},
            {"tile", config.getTile()}, {"tasks", config.getTasks()},
            {"tile_m", config.getTileM()}, {"tile_n", config.getTileN()}, {"tile_k", config.getTileK()},
            {"region_tile", config.getRegionTile()}, {"local_bytes", config.getLocalBytes()}}},
        {"full_extent_dimensions", std::move(fullExtents)},
        {"native", llvm::json::Object{{"requirements", requirements->serialize()},
            {"slots", nativeABI->serialize()}}}};
    out << "#pragma bang walign(" << function->getAttrOfType<IntegerAttr>("bangc.wram_align").getInt() << ")\n"
        << tileImplementations << "\n__mlu_global__ void intent_device(" << llvm::join(signature, ", ") << ") {\n";
    auto bytes = [&](StringRef name) { return function->getAttrOfType<IntegerAttr>(name).getInt(); };
    if (bytes("bangc.nram_bytes")) line("__nram__ __attribute__((aligned(128))) unsigned char local_nram[" + std::to_string(bytes("bangc.nram_bytes")) + "];", 1);
    if (bytes("bangc.wram_bytes")) line("__wram__ __attribute__((aligned(128))) unsigned char local_wram[" + std::to_string(bytes("bangc.wram_bytes")) + "];", 1);
    if (bytes("bangc.sram_bytes")) line("__mlu_shared__ __attribute__((aligned(128))) unsigned char group_sram[" + std::to_string(bytes("bangc.sram_bytes")) + "];", 1);
    if (failed(block(function.front(), 1))) return failure();
    out << "}\n\nextern \"C\" int intent_launch(void *stream";
    if (!signature.empty()) out << ", " << llvm::join(signature, ", ");
    auto group = function->getAttrOfType<IntegerAttr>("intent_dsa.group_width");
    out << ") {\n  cnrtDim3_t dim = {" << (group ? group.getInt() : config.getTasks()) << ", "
        << (group ? config.getTasks() / group.getInt() : 1) << ", 1};\n"
        << "  intent_device<<<dim, " << (group ? "cnrtFuncTypeUnion1" : "cnrtFuncTypeBlock") << ", static_cast<cnrtQueue_t>(stream)>>>("
        << llvm::join(call, ", ") << ");\n  return static_cast<int>(cnrtGetLastError());\n}\n";
    return success();
  }
private:
  void line(const std::string &text, unsigned depth) { out.indent(depth * 2) << text << "\n"; }
  std::string name(Value value) { return names.lookup(value); }
  std::string unsignedValue(Value value) {
    return "static_cast<" + unsignedType(value.getType()) + ">(" + name(value) + ")";
  }
  std::string bind(Value value) {
    std::string result = "v" + std::to_string(next++); names[value] = result; return result;
  }
  std::string count(Value value) { return std::to_string(cast<MemRefType>(value.getType()).getNumElements()); }
  std::string shape(Value value) {
    auto type = cast<MemRefType>(value.getType());
    return ctype(type.getElementType()) + ", " + std::to_string(type.getDimSize(0)) + ", " + std::to_string(type.getDimSize(1));
  }
  std::string floatLiteral(FloatAttr attribute) {
    if (attribute.getType().isBF16()) return std::to_string(attribute.getValue().bitcastToAPInt().getZExtValue());
    double value = attribute.getValueAsDouble();
    if (std::isnan(value)) return "NAN";
    if (std::isinf(value)) return value < 0 ? "(-INFINITY)" : "INFINITY";
    SmallString<32> text;
    attribute.getValue().toString(text);
    std::string literal = text.str().str();
    if (literal.find_first_of(".eE") == std::string::npos) literal += ".0";
    return attribute.getType().isF64() ? literal : literal + "f";
  }
  LogicalResult block(Block &body, unsigned depth) {
    for (Operation &op : body) if (failed(operation(&op, depth))) return failure();
    return success();
  }
  LogicalResult operation(Operation *op, unsigned depth) {
    if (isa<func::ReturnOp, scf::YieldOp>(op)) return success();
    if (auto sync = dyn_cast<dsa::SynchronizeOp>(op)) {
      line(sync.getLocalOnly() ? "intent_sync_local();" : "__sync();", depth); return success();
    }
    if (isa<dsa::GroupSynchronizeOp>(op)) { line("__sync(); __sync_cluster();", depth); return success(); }
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (branch.getNumResults()) return op->emitError("BANG C conditional results must be materialized");
      line("if (" + name(branch.getCondition()) + ") {", depth);
      if (failed(block(branch.getThenRegion().front(), depth + 1))) return failure();
      if (!branch.getElseRegion().empty()) {
        line("} else {", depth);
        if (failed(block(branch.getElseRegion().front(), depth + 1))) return failure();
      }
      line("}", depth); return success();
    }
    if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      if (loop.getNumResults() || loop.getNumOperands()) return op->emitError("BANG C while state must be materialized");
      line("while (true) {", depth);
      if (failed(block(loop.getBefore().front(), depth + 1)) || failed(block(loop.getAfter().front(), depth + 1))) return failure();
      line("}", depth); return success();
    }
    if (auto condition = dyn_cast<scf::ConditionOp>(op)) {
      if (!condition.getArgs().empty()) return op->emitError("BANG C condition state must be materialized");
      line("if (!(" + name(condition.getCondition()) + ")) break;", depth); return success();
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (loop.getNumResults()) return op->emitError("BANG C serialization requires materialized loop state");
      std::string iv = bind(loop.getInductionVar());
      line("for (int64_t " + iv + " = " + name(loop.getLowerBound()) + "; " + iv + " < " + name(loop.getUpperBound()) +
           "; " + iv + " += " + name(loop.getStep()) + ") {", depth);
      if (failed(block(*loop.getBody(), depth + 1))) return failure();
      line("}", depth); return success();
    }
    if (auto allocation = dyn_cast<memref::AllocaOp>(op)) {
      auto type = allocation.getType();
      std::string buffer = type.getMemorySpaceAsInt() == dsa::matrixSpace ? "local_wram" :
          type.getMemorySpaceAsInt() == dsa::sharedSpace ? "group_sram" : "local_nram";
      line(ctype(type.getElementType()) + " *" + bind(allocation.getResult()) + " = reinterpret_cast<" +
          ctype(type.getElementType()) + " *>(" + buffer + " + " +
          std::to_string(op->getAttrOfType<IntegerAttr>("bangc.offset").getInt()) + ");", depth);
      return success();
    }
    if (auto view = dyn_cast<memref::ReinterpretCastOp>(op)) {
      line(ctype(view.getType().getElementType()) + " *" + bind(view.getResult()) + " = " + name(view.getSource()) + ";", depth);
      return success();
    }
    if (auto load = dyn_cast<dsa::LoadTileOp>(op)) {
      bool local = cast<MemRefType>(load.getSource().getType()).getMemorySpaceAsInt() == dsa::nramSpace;
      bool shared = cast<MemRefType>(load.getSource().getType()).getMemorySpaceAsInt() == dsa::sharedSpace;
      line(std::string(local ? "intent_load_local_tile<" : "intent_load_tile<") + shape(load.getOutput()) +
          (shared ? (load.getAsynchronous() ? ", true, true" : ", false, true") : (load.getAsynchronous() ? ", true" : "")) + ">(" + name(load.getOutput()) + ", " + name(load.getSource()) +
          ", " + name(load.getOffset()) + ", " + name(load.getRowStride()) + ", " + name(load.getColumnStride()) +
          ", " + name(load.getRows()) + ", " + name(load.getColumns()) + ");", depth); return success();
    }
    if (auto stage = dyn_cast<dsa::StageTileOp>(op)) {
      line("intent_stage_tile<" + shape(stage.getOutput()) + ">(" + name(stage.getOutput()) + ", " +
          name(stage.getSource()) + ", " + name(stage.getOffset()) + ", " + name(stage.getRowStride()) + ", " +
          name(stage.getColumnStride()) + ", " + name(stage.getRows()) + ", " + name(stage.getColumns()) + ");", depth);
      return success();
    }
    if (auto plan = dyn_cast<dsa::GatherPlanOp>(op)) {
      line("intent_prepare_gather_runs<" + count(plan.getRowOffsets()) + ">(" + name(plan.getRowOffsets()) + ", " +
          name(plan.getRows()) + ", " + name(plan.getOutput()) + ", " + name(plan.getLaneIndices()) + ");", depth);
      return success();
    }
    if (auto gather = dyn_cast<dsa::GatherRowsOp>(op)) {
      line(std::string(gather.getPlan() ? "intent_gather_runs<" : "intent_gather_rows<") + shape(gather.getOutput()) + ">(" + name(gather.getOutput()) + ", " +
          name(gather.getSource()) + ", " + name(gather.getRowOffsets()) + ", " + name(gather.getColumnStride()) +
          ", " + name(gather.getRows()) + ", " + name(gather.getColumns()) +
          (gather.getPlan() ? ", " + name(gather.getPlan()) : "") + ");", depth); return success();
    }
    if (auto gather = dyn_cast<dsa::GroupGatherRowsOp>(op)) {
      line("intent_group_gather_rows<" + shape(gather.getOutput()) + ">(" + name(gather.getOutput()) + ", " +
          name(gather.getSource()) + ", " + name(gather.getRowOffsets()) + ", " + name(gather.getPlan()) + ", " +
          name(gather.getSharedData()) + ", " + name(gather.getSharedMetadata()) + ", " +
          name(gather.getRows()) + ", " + name(gather.getLane()) + ");", depth);
      return success();
    }
    if (auto store = dyn_cast<dsa::StoreTileOp>(op)) {
      bool local = cast<MemRefType>(store.getDestination().getType()).getMemorySpaceAsInt() == dsa::nramSpace;
      line(std::string(local ? "intent_store_local_tile<" : "intent_store_tile<") + shape(store.getInput()) + ">(" + name(store.getDestination()) + ", " + name(store.getInput()) +
          ", " + name(store.getOffset()) + ", " + name(store.getRowStride()) + ", " + name(store.getColumnStride()) +
          ", " + name(store.getRows()) + ", " + name(store.getColumns()) + ");", depth); return success();
    }
    if (auto iota = dyn_cast<dsa::IotaOp>(op)) {
      line("intent_iota_local<" + count(iota.getOutput()) + ">(" + name(iota.getOutput()) + ");", depth);
      return success();
    }
    if (auto broadcast = dyn_cast<dsa::BroadcastRowsOp>(op)) {
      line("intent_broadcast_rows<" + shape(broadcast.getOutput()) + ">(" + name(broadcast.getOutput()) + ", " +
          name(broadcast.getInput()) + ", " + name(broadcast.getScratch()) + ", " + name(broadcast.getOffset()) + ", " +
          name(broadcast.getRows()) + ", " + name(broadcast.getColumns()) + ");", depth);
      return success();
    }
    if (auto layout = dyn_cast<dsa::IndexLayoutOp>(op)) {
      auto output = cast<MemRefType>(layout.getOutput().getType());
      std::string width = std::to_string(output.getDimSize(1));
      std::string dst = "reinterpret_cast<uint32_t *>(" + name(layout.getOutput()) + ")";
      if (layout.getInput().getType().isInteger(64)) {
        line("__bang_write_value(" + dst + ", " + width + ", uint32_t(" + name(layout.getInput()) + "));", depth);
        line("__bang_write_value(" + dst + " + " + width + ", " + width + ", uint32_t(uint64_t(" + name(layout.getInput()) + ") >> 32));", depth);
      } else {
        bool split = cast<MemRefType>(layout.getInput().getType()).getElementType().isInteger(64);
        line("__bang_transpose(" + dst + ", reinterpret_cast<const uint32_t *>(" + name(layout.getInput()) + "), " +
             (split ? width + ", 2" : "2, " + width) + ");", depth);
      }
      return success();
    }
    if (auto binary = dyn_cast<dsa::IndexBinaryOp>(op)) {
      std::string width = std::to_string(cast<MemRefType>(binary.getOutput().getType()).getDimSize(1));
      std::string dst = "reinterpret_cast<uint32_t *>(" + name(binary.getOutput()) + ")";
      std::string lhs = "reinterpret_cast<const uint32_t *>(" + name(binary.getLhs()) + ")";
      bool shift = binary.getKind() == BinaryOperator::LeftShift || binary.getKind() == BinaryOperator::RightShift;
      if (shift) {
        std::string amount = std::to_string(binary.getRhs().getDefiningOp<arith::ConstantIntOp>().value());
        line(std::string(binary.getKind() == BinaryOperator::LeftShift ? "intent_index_shl<" : "intent_index_sar<") +
             width + ", " + amount + ">(" + dst + ", " + lhs + ");", depth);
      } else {
        StringRef kind = binary.getKind() == BinaryOperator::Add ? "add" : binary.getKind() == BinaryOperator::Subtract ? "sub" : "mul";
        std::string rhs = isa<MemRefType>(binary.getRhs().getType()) ?
            "reinterpret_cast<const uint32_t *>(" + name(binary.getRhs()) + ")" : name(binary.getRhs());
        line("intent_index_" + kind.str() + "<" + width + ">(" + dst + ", " + lhs + ", " + rhs + ");", depth);
      }
      return success();
    }
    if (auto fill = dyn_cast<dsa::FillOp>(op)) {
      if (fill.getValue().getType().isF16() || fill.getValue().getType().isF32())
        line("__bang_write_value(" + name(fill.getOutput()) + ", " + count(fill.getOutput()) + ", " + name(fill.getValue()) + ");", depth);
      else line("intent_fill_local<" + ctype(fill.getValue().getType()) + ", " + count(fill.getOutput()) + ">(" + name(fill.getOutput()) + ", " + name(fill.getValue()) + ");", depth);
      return success();
    }
    if (auto select = dyn_cast<dsa::SelectOp>(op)) {
      if (Value scratch = select.getScratch()) {
        line("intent_select_bits<" + count(select.getOutput()) + ", " + std::to_string(cast<MemRefType>(scratch.getType()).getDimSize(1)) + ">(" +
            name(select.getOutput()) + ", " + name(select.getCondition()) + ", " + name(select.getTrueValue()) + ", " +
            name(select.getFalseValue()) + ", " + name(scratch) + ");", depth);
        return success();
      }
      line("intent_select_local<" + ctype(cast<MemRefType>(select.getOutput().getType()).getElementType()) + ", " +
          count(select.getOutput()) + ">(" + name(select.getOutput()) + ", " + name(select.getCondition()) + ", " +
          name(select.getTrueValue()) + ", " + name(select.getFalseValue()) + ");", depth); return success();
    }
    if (auto prepare = dyn_cast<dsa::PrepareMatrixViewOp>(op)) {
      auto output = cast<MemRefType>(prepare.getOutput().getType());
      auto input = cast<MemRefType>(prepare.getInputSlice().getType());
      line("intent_prepare_matrix_view<" + ctype(output.getElementType()) + ", " +
          std::to_string(output.getDimSize(0)) + ", " + std::to_string(output.getDimSize(1)) + ", " +
          std::to_string(input.getDimSize(0)) + ", " + std::to_string(input.getDimSize(1)) + ">(" + name(prepare.getOutput()) + ", " + name(prepare.getSource()) +
          ", " + name(prepare.getInputSlice()) + ", " + name(prepare.getTransposedSlice()) + ", " +
          name(prepare.getOffset()) + ", " + name(prepare.getRowStride()) + ", " + name(prepare.getColumnStride()) +
          ", " + name(prepare.getColumns()) + ");", depth);
      return success();
    }
    if (auto transpose = dyn_cast<dsa::TransposeOp>(op)) {
      line("intent_transpose_tile<" + shape(transpose.getInput()) + ">(" + name(transpose.getOutput()) + ", " +
          name(transpose.getInput()) + ", " + name(transpose.getRows()) + ", " + name(transpose.getColumns()) + ");", depth);
      return success();
    }
    if (auto compare = dyn_cast<dsa::CompareRangeOp>(op)) {
      auto type = cast<MemRefType>(compare.getOutput().getType());
      line("intent_compare_range<" + std::to_string(type.getDimSize(0)) + ", " +
          std::to_string(type.getDimSize(1)) + ">(" + name(compare.getOutput()) + ", " +
          name(compare.getRowCoordinates()) + ", " + name(compare.getRows()) + ", " +
          std::to_string(compare.getBaseAttr().getInt()) + "LL);", depth);
      return success();
    }
    if (auto compare = dyn_cast<dsa::CompareRampOp>(op)) {
      auto type = cast<MemRefType>(compare.getOutput().getType());
      line("intent_compare_ramp<" + std::to_string(type.getDimSize(0)) + ", " +
          std::to_string(type.getDimSize(1)) + ">(" + name(compare.getOutput()) + ", " +
          (compare.getScratch() ? name(compare.getScratch()) : "nullptr") + ", " + name(compare.getRowBegin()) + ", " +
          std::to_string(compare.getBaseAttr().getInt()) + "LL);", depth);
      return success();
    }
    if (auto masked = dyn_cast<dsa::MaskedFillOp>(op)) {
      line("intent_masked_fill<" + count(masked.getOutput()) + ">(" + name(masked.getOutput()) + ", " +
          name(masked.getInput()) + ", " + name(masked.getMask()) + ", " + name(masked.getValue()) + ");", depth);
      return success();
    }
    if (auto compare = dyn_cast<dsa::CompareOp>(op)) {
      if (compare.getScratch()) {
        auto workspace = cast<MemRefType>(compare.getScratch().getType());
        line("intent_compare_i64<" + count(compare.getOutput()) + ", " + std::to_string(workspace.getDimSize(1)) +
            ", " + std::to_string(static_cast<unsigned>(compare.getPredicate())) + ">(" + name(compare.getOutput()) +
            ", " + name(compare.getLhs()) + ", " + name(compare.getRhs()) + ", " + name(compare.getScratch()) + ");", depth);
        return success();
      }
      auto callee = op->getAttrOfType<StringAttr>("bangc.callee");
      if (!callee) return op->emitError("comparison requires a selected BANG C primitive");
      line(callee.getValue().str() + "(" + count(compare.getOutput()) + ", " + name(compare.getOutput()) + ", " +
          name(compare.getLhs()) + ", " + name(compare.getRhs()) + ");", depth);
      return success();
    }
    if (auto store = dyn_cast<dsa::StoreScalarOp>(op)) {
      line(name(store.getDestination()) + "[" + name(store.getOffset()) + "] = " + name(store.getValue()) + ";", depth); return success();
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      auto type = cast<MemRefType>(store.getMemref().getType());
      line("intent_write_local(" + name(store.getMemref()) + " + " + name(store.getIndices()[0]) + " * " + std::to_string(type.getDimSize(1)) +
          " + " + name(store.getIndices()[1]) + ", " + name(store.getValue()) + ");", depth); return success();
    }
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      line("intent_copy_local<" + ctype(cast<MemRefType>(copy.getSource().getType()).getElementType()) + ", " +
          count(copy.getSource()) + ">(" + name(copy.getTarget()) + ", " + name(copy.getSource()) + ");", depth); return success();
    }
    if (auto binary = dyn_cast<dsa::BinaryOp>(op)) {
      if (auto implementation = op->getAttrOfType<StringAttr>("bangc.implementation");
          implementation && implementation.getValue() == "row_scalar") {
        auto type = cast<MemRefType>(binary.getOutput().getType());
        line("intent_binary_rows<" + ctype(type.getElementType()) + ", " +
            std::to_string(type.getDimSize(0)) + ", " + std::to_string(type.getDimSize(1)) + ", " +
            std::to_string(static_cast<int>(binary.getKind())) + ">(" + name(binary.getOutput()) + ", " +
            name(binary.getLhs()) + ", " + name(binary.getRhs()) + ");", depth);
        return success();
      }
      if (auto implementation = op->getAttrOfType<StringAttr>("bangc.implementation");
          implementation && implementation.getValue() == "reciprocal_f32_ftz") {
        line("intent_reciprocal_ftz<" + count(binary.getOutput()) + ", " +
            std::to_string(cast<MemRefType>(binary.getScratch().getType()).getDimSize(1)) + ">(" +
            name(binary.getOutput()) + ", " + name(binary.getRhs()) + ", " + name(binary.getScratch()) + ");", depth);
        return success();
      }
      if (binary.getScratch()) {
        auto workspace = cast<MemRefType>(binary.getScratch().getType());
        line("intent_numeric_extrema<" + ctype(workspace.getElementType()) + ", " + count(binary.getOutput()) +
            ", " + std::to_string(workspace.getNumElements()) + ", " +
            (binary.getKind() == BinaryOperator::MaximumNum ? "true" : "false") + ">(" + name(binary.getOutput()) +
            ", " + name(binary.getLhs()) + ", " + name(binary.getRhs()) + ", " + name(binary.getScratch()) + ");", depth);
        return success();
      }
      if (cast<MemRefType>(binary.getOutput().getType()).getElementType().isInteger(64)) {
        auto callee = op->getAttrOfType<StringAttr>("bangc.callee");
        if (!callee) return op->emitError("unbound BANG integer tile operation");
        line(callee.getValue().str() + "(" + count(binary.getOutput()) + ", " + name(binary.getOutput()) + ", " +
            name(binary.getLhs()) + ", " + name(binary.getRhs()) + ");", depth);
        return success();
      }
      if (binary.getKind() == BinaryOperator::TrueDivide) {
        if (op->getAttrOfType<StringAttr>("bangc.implementation").getValue() == "divide_f32") {
          line("intent_divide_f32<" + count(binary.getOutput()) + ", " + (binary.getApproximate() ? "true" : "false") + ", " +
              (binary.getFlushToZero() ? "true" : "false") + ">(" + name(binary.getOutput()) + ", " + name(binary.getLhs()) + ", " + name(binary.getRhs()) + ");", depth);
          return success();
        }
        line("intent_divide_local<" + ctype(cast<MemRefType>(binary.getOutput().getType()).getElementType()) + ", " +
            count(binary.getOutput()) + ">(" + name(binary.getOutput()) + ", " + name(binary.getLhs()) + ", " + name(binary.getRhs()) + ");", depth);
        return success();
      }
      std::string intrinsic;
      switch (binary.getKind()) {
      case BinaryOperator::Add: intrinsic = "__bang_add"; break;
      case BinaryOperator::Subtract: intrinsic = "__bang_sub"; break;
      case BinaryOperator::Multiply: intrinsic = "__bang_mul"; break;
      case BinaryOperator::Maximum: intrinsic = "__bang_nan_maximum"; break;
      case BinaryOperator::Minimum: intrinsic = "__bang_nan_minimum"; break;
      case BinaryOperator::MaximumNum: intrinsic = "__bang_maximum"; break;
      case BinaryOperator::MinimumNum: intrinsic = "__bang_minimum"; break;
      default: return op->emitError("unbound BANG binary operation");
      }
      if (!isa<MemRefType>(binary.getRhs().getType())) intrinsic += "_scalar";
      if (auto callee = op->getAttrOfType<StringAttr>("bangc.callee")) intrinsic = callee.getValue().str();
      line(intrinsic + "(" + name(binary.getOutput()) + ", " + name(binary.getLhs()) + ", " +
          name(binary.getRhs()) + ", " + count(binary.getOutput()) + ");", depth); return success();
    }
    if (auto unary = dyn_cast<dsa::UnaryOp>(op)) {
      if (Value scratch = unary.getScratch()) {
        if (cast<MemRefType>(scratch.getType()).getElementType().isF32()) {
          line("intent_exp_f32_tile<" + count(unary.getOutput()) + ", " +
              std::to_string(cast<MemRefType>(scratch.getType()).getDimSize(1)) +
              (unary.getKind() == UnaryOperator::Exp2 ? ", true>(" : ">(") +
              name(unary.getOutput()) + ", " + name(unary.getInput()) + ", " + name(scratch) + ");", depth);
          return success();
        }
        line("intent_exp2_ftz_tile<" + count(unary.getOutput()) + ", " + count(scratch) +
            (unary->hasAttr("bangc.input_non_subnormal") ? ", true>(" : ">(") +
            name(unary.getOutput()) + ", " + name(unary.getInput()) + ", " + name(scratch) + ");", depth);
        return success();
      }
      if (auto callee = op->getAttrOfType<StringAttr>("bangc.callee")) {
        line(callee.getValue().str() + "(" + count(unary.getOutput()) + ", " + name(unary.getOutput()) + ", " + name(unary.getInput()) + ");", depth);
        return success();
      }
      if (auto implementation = op->getAttrOfType<StringAttr>("bangc.implementation");
          implementation && implementation.getValue() == "exp2_f32") {
        line("intent_exp2_tile<" + count(unary.getOutput()) + ", " + (unary.getFlushToZero() ? "true" : "false") + ">(" +
            name(unary.getOutput()) + ", " + name(unary.getInput()) + ");", depth);
        return success();
      }
      std::string intrinsic;
      switch (unary.getKind()) {
      case UnaryOperator::Abs: intrinsic = "__bang_abs"; break;
      case UnaryOperator::Negate:
        line("__bang_mul_scalar(" + name(unary.getOutput()) + ", " + name(unary.getInput()) + ", " +
            ctype(cast<MemRefType>(unary.getInput().getType()).getElementType()) + "(-1), " + count(unary.getOutput()) + ");", depth); return success();
      default: return op->emitError("unbound BANG unary operation");
      }
      line(intrinsic + "(" + name(unary.getOutput()) + ", " + name(unary.getInput()) + ", " + count(unary.getOutput()) + ");", depth); return success();
    }
    if (auto castOp = dyn_cast<dsa::CastOp>(op)) {
      if (auto callee = op->getAttrOfType<StringAttr>("bangc.callee")) {
        line(callee.getValue().str() + "(" + count(castOp.getOutput()) + ", " + name(castOp.getOutput()) + ", " + name(castOp.getInput()) + ");", depth);
        return success();
      }
      auto from = cast<MemRefType>(castOp.getInput().getType()).getElementType();
      auto to = cast<MemRefType>(castOp.getOutput().getType()).getElementType();
      if (from == to) line("__memcpy(" + name(castOp.getOutput()) + ", " + name(castOp.getInput()) + ", " +
          count(castOp.getOutput()) + " * sizeof(" + ctype(to) + "), NRAM2NRAM);", depth);
      else if (from.isInteger(1) && to.isF32())
        line("intent_bool_to_f32_tile<" + count(castOp.getOutput()) + ">(" + name(castOp.getOutput()) + ", " +
            name(castOp.getInput()) + ");", depth);
      else if ((from.isF16() && to.isF32()) || (from.isF32() && to.isF16()))
        line(std::string(to.isF32() ? "__bang_half2float(" : "__bang_float2half_rn(") + name(castOp.getOutput()) + ", " +
            name(castOp.getInput()) + ", " + count(castOp.getOutput()) + ");", depth);
      else if ((from.isBF16() && to.isF32()) || (from.isF32() && to.isBF16()))
        line(std::string(to.isF32() ? "intent_bf16_to_f32_tile<" : "intent_f32_to_bf16_tile<") + count(castOp.getOutput()) + ">(" +
            name(castOp.getOutput()) + ", " + name(castOp.getInput()) + ");", depth);
      else line("intent_cast_local<" + ctype(from) + ", " + ctype(to) + ", " + count(castOp.getOutput()) + ">(" +
          name(castOp.getOutput()) + ", " + name(castOp.getInput()) + ");", depth);
      return success();
    }
    if (auto reduce = dyn_cast<dsa::ReduceOp>(op)) {
      auto input = cast<MemRefType>(reduce.getInput().getType());
      if (input.getElementType().isF16()) {
        line("intent_reduce_half_extrema<" + std::to_string(input.getDimSize(0)) + ", " +
            std::to_string(input.getDimSize(1)) + ", " + std::to_string(static_cast<int>(reduce.getKind())) + ">(" +
            name(reduce.getOutput()) + ", " + name(reduce.getInput()) + ", " + name(reduce.getScratch()) + ", " +
            name(reduce.getCount()) + ", " + name(reduce.getIdentity()) + ");", depth);
        return success();
      }
      if (reduce.getAxis() == 0) {
        line("intent_reduce_rows<" + std::to_string(cast<MemRefType>(reduce.getInput().getType()).getDimSize(1)) + ">(" +
            name(reduce.getOutput()) + ", " + name(reduce.getInput()) + ", " + name(reduce.getScratch()) + ", " +
            name(reduce.getCount()) + ", " + name(reduce.getIdentity()) + ");", depth);
        return success();
      }
      if (input.getDimSize(0) > 1) {
        auto scratch = cast<MemRefType>(reduce.getScratch().getType());
        if (input.getDimSize(0) < 32 && input.getDimSize(1) >= 1024 && scratch.getDimSize(1) == input.getDimSize(1)) {
          line("intent_reduce_row_tiles<" + std::to_string(input.getDimSize(0)) + ", " +
              std::to_string(input.getDimSize(1)) + ", " + std::to_string(static_cast<int>(reduce.getKind())) + ">(" +
              name(reduce.getOutput()) + ", " + name(reduce.getInput()) + ", " + name(reduce.getScratch()) + ", " +
              name(reduce.getCount()) + ", " + name(reduce.getIdentity()) + ");", depth);
          return success();
        }
        std::string helper = reduce.getKind() == BinaryOperator::Add ? "intent_reduce_row_sums<" : "intent_reduce_row_extrema<";
        std::string kind = reduce.getKind() == BinaryOperator::Add ? "" : ", " + std::to_string(static_cast<int>(reduce.getKind()));
        line(helper + std::to_string(input.getDimSize(0)) + ", " +
            std::to_string(input.getDimSize(1)) + kind + ">(" + name(reduce.getOutput()) + ", " +
            name(reduce.getInput()) + ", " + name(reduce.getScratch()) + ", " +
            name(reduce.getCount()) + ", " + name(reduce.getIdentity()) + ");", depth);
        return success();
      }
      line("intent_reduce<" + count(reduce.getInput()) + ", " + std::to_string(static_cast<int>(reduce.getKind())) + ">(" +
          name(reduce.getOutput()) + ", " + name(reduce.getInput()) + ", " + name(reduce.getScratch()) + ", " +
          name(reduce.getCount()) + ", " + name(reduce.getIdentity()) + ");", depth); return success();
    }
    if (auto divide = dyn_cast<dsa::DivideRNOp>(op)) {
      line("intent_divide_rn<" + count(divide.getOutput()) + ", " +
          std::to_string(cast<MemRefType>(divide.getScratch().getType()).getDimSize(1)) + ">(" +
          name(divide.getOutput()) + ", " + name(divide.getLhs()) + ", " + name(divide.getRhs()) + ", " +
          name(divide.getScratch()) + ", " + name(divide.getLaneIndices()) + ");", depth);
      return success();
    }
    if (auto divide = dyn_cast<dsa::DivideCastOp>(op)) {
      line("intent_divide_cast_f16<" + count(divide.getOutput()) + ">(" +
          name(divide.getOutput()) + ", " + name(divide.getLhs()) + ", " + name(divide.getRhs()) + ", " +
          name(divide.getQuotient()) + ", " + name(divide.getBounds()) + ", " + name(divide.getAccepted()) + ", " +
          name(divide.getNarrowBounds()) + ", " + name(divide.getLaneIndices()) + ");", depth); return success();
    }
    if (auto prepare = dyn_cast<dsa::PrepareMatrixOp>(op)) {
      line("intent_prepare_matrix<" + shape(prepare.getOutput()) + ", " + (prepare.getInputTransposed() ? "true" : "false") + ">(" +
          name(prepare.getOutput()) + ", " + name(prepare.getInput()) + ", " +
          (prepare.getScratch() ? name(prepare.getScratch()) : "nullptr") + ", " +
          (prepare.getReshaped() ? name(prepare.getReshaped()) : "nullptr") + ");", depth); return success();
    }
    if (auto matrix = dyn_cast<dsa::MatrixTileOp>(op)) {
      auto a = cast<MemRefType>(matrix.getLhs().getType()), c = cast<MemRefType>(matrix.getAccumulator().getType());
      line("intent_matmul<" + ctype(a.getElementType()) + ", " + std::to_string(a.getDimSize(0)) + ", " +
          std::to_string(a.getDimSize(1)) + ", " + std::to_string(c.getDimSize(1)) +
          (matrix.getAccumulate() ? "" : ", false") + ">(" + name(matrix.getAccumulator()) + ", " +
          name(matrix.getLhs()) + ", " + name(matrix.getRhs()) + ");", depth); return success();
    }
    std::string expression;
    if (isa<dsa::TaskIdOp>(op)) expression = "taskId";
    else if (isa<dsa::TaskCountOp>(op)) expression = "taskDim";
    else if (isa<dsa::GroupIdOp>(op)) expression = "taskIdY";
    else if (isa<dsa::GroupCountOp>(op)) expression = "taskDimY";
    else if (isa<dsa::LocalIdOp>(op)) expression = "taskIdX";
    else if (isa<dsa::IsMemoryCoreOp>(op)) expression = "__is_mpu()";
    else if (auto stride = dyn_cast<dsa::StrideOp>(op)) expression = name(stride.getSource()) + "_s" + std::to_string(stride.getAxis());
    else if (auto dim = dyn_cast<memref::DimOp>(op)) {
      auto axis = dim.getConstantIndex();
      if (!axis) return op->emitError("BANG C requires a bound view dimension axis");
      expression = name(dim.getSource()) + "_d" + std::to_string(*axis);
    } else if (auto scalar = dyn_cast<dsa::LoadScalarOp>(op)) expression = name(scalar.getSource()) + "[" + name(scalar.getOffset()) + "]";
    else if (auto scalar = dyn_cast<memref::LoadOp>(op)) {
      auto type = cast<MemRefType>(scalar.getMemref().getType());
      expression = "intent_read_local(" + name(scalar.getMemref()) + " + " + name(scalar.getIndices()[0]) + " * " + std::to_string(type.getDimSize(1)) + " + " + name(scalar.getIndices()[1]) + ")";
    } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      if (auto floating = dyn_cast<FloatAttr>(constant.getValue())) expression = floatLiteral(floating);
      else expression = std::to_string(cast<IntegerAttr>(constant.getValue()).getInt());
    } else if (isa<arith::ExtFOp, arith::TruncFOp, arith::IndexCastOp, arith::IndexCastUIOp,
                   arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
                   arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp>(op)) {
      Type from = op->getOperand(0).getType(), to = op->getResult(0).getType();
      std::string value = name(op->getOperand(0));
      if (isa<arith::ExtUIOp, arith::UIToFPOp, arith::IndexCastUIOp>(op))
        value = unsignedValue(op->getOperand(0));
      if (from.isBF16()) value = "intent_bf16_to_float(" + value + ")";
      if (isa<arith::FPToUIOp>(op)) value = "static_cast<" + unsignedType(to) + ">(" + value + ")";
      expression = to.isBF16() ? "intent_number_to_bf16(" + value + ")" : "static_cast<" + ctype(to) + ">(" + value + ")";
    }
    else if (auto select = dyn_cast<arith::SelectOp>(op)) expression = name(select.getCondition()) + " ? " + name(select.getTrueValue()) + " : " + name(select.getFalseValue());
    else if (isa<arith::NegFOp>(op)) expression = "-" + name(op->getOperand(0));
    else if (isa<math::AbsIOp>(op)) {
      Value input = op->getOperand(0);
      expression = "(" + name(input) + " < 0 ? static_cast<" + ctype(input.getType()) +
          ">(" + unsignedType(input.getType()) + "(0) - " + unsignedValue(input) + ") : " + name(input) + ")";
    }
    else if (op->getName().getDialectNamespace() == "math" && op->getNumOperands() == 1) {
      std::string callee;
      if (isa<math::ExpOp>(op)) callee = "expf";
      if (isa<math::Exp2Op>(op)) callee = "exp2f";
      if (isa<math::SinOp>(op)) callee = "sinf";
      if (isa<math::CosOp>(op)) callee = "cosf";
      if (isa<math::FloorOp>(op)) callee = "floorf";
      if (isa<math::LogOp>(op)) callee = "logf";
      if (isa<math::SqrtOp>(op)) callee = "sqrtf";
      if (isa<math::RsqrtOp>(op)) callee = "1.0f / sqrtf";
      if (isa<math::TanhOp>(op)) callee = "tanhf";
      if (isa<math::AbsFOp>(op)) callee = "fabsf";
      if (op->getResult(0).getType().isF64() && !callee.empty()) callee.pop_back();
      if (!callee.empty()) expression = callee + "(" + name(op->getOperand(0)) + ")";
    }
    else if (op->getNumOperands() == 2 && op->getNumResults() == 1) {
      std::string lhs = name(op->getOperand(0)), rhs = name(op->getOperand(1)), symbol;
      if (isa<arith::AddIOp, arith::AddFOp>(op)) symbol = "+";
      if (isa<arith::SubIOp, arith::SubFOp>(op)) symbol = "-";
      if (isa<arith::MulIOp, arith::MulFOp>(op)) symbol = "*";
      if (isa<arith::DivSIOp, arith::DivUIOp, arith::DivFOp>(op)) symbol = "/";
      if (isa<arith::RemSIOp, arith::RemUIOp>(op)) symbol = "%";
      if (isa<arith::AndIOp>(op)) symbol = "&";
      if (isa<arith::OrIOp>(op)) symbol = "|";
      if (isa<arith::XOrIOp>(op)) symbol = "^";
      if (isa<arith::ShLIOp>(op)) symbol = "<<";
      if (isa<arith::ShRSIOp, arith::ShRUIOp>(op)) symbol = ">>";
      bool unsignedOperands = isa<arith::DivUIOp, arith::RemUIOp, arith::ShRUIOp,
                                  arith::MinUIOp, arith::MaxUIOp>(op);
      if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
        switch (cmp.getPredicate()) {
        case arith::CmpIPredicate::eq: symbol = "=="; break;
        case arith::CmpIPredicate::ne: symbol = "!="; break;
        case arith::CmpIPredicate::slt: symbol = "<"; break;
        case arith::CmpIPredicate::sle: symbol = "<="; break;
        case arith::CmpIPredicate::sgt: symbol = ">"; break;
        case arith::CmpIPredicate::sge: symbol = ">="; break;
        case arith::CmpIPredicate::ult: symbol = "<"; unsignedOperands = true; break;
        case arith::CmpIPredicate::ule: symbol = "<="; unsignedOperands = true; break;
        case arith::CmpIPredicate::ugt: symbol = ">"; unsignedOperands = true; break;
        case arith::CmpIPredicate::uge: symbol = ">="; unsignedOperands = true; break;
        }
      }
      if (unsignedOperands) {
        lhs = unsignedValue(op->getOperand(0));
        rhs = unsignedValue(op->getOperand(1));
      }
      if (auto cmp = dyn_cast<arith::CmpFOp>(op)) {
        switch (cmp.getPredicate()) {
        case arith::CmpFPredicate::OEQ: symbol = "=="; break;
        case arith::CmpFPredicate::UNE: symbol = "!="; break;
        case arith::CmpFPredicate::OLT: symbol = "<"; break;
        case arith::CmpFPredicate::OLE: symbol = "<="; break;
        case arith::CmpFPredicate::OGT: symbol = ">"; break;
        case arith::CmpFPredicate::OGE: symbol = ">="; break;
        default: return op->emitError("BANG C floating comparison predicate is not bound");
        }
      }
      if (!symbol.empty()) expression = lhs + " " + symbol + " " + rhs;
      else if (isa<arith::MinSIOp, arith::MinUIOp>(op)) expression = "(" + lhs + " < " + rhs + " ? " + lhs + " : " + rhs + ")";
      else if (isa<arith::MaxSIOp, arith::MaxUIOp>(op)) expression = "(" + lhs + " > " + rhs + " ? " + lhs + " : " + rhs + ")";
      else if (isa<arith::FloorDivSIOp>(op)) expression = "(" + lhs + " / " + rhs + " - (" + lhs + " % " + rhs + " != 0 && ((" + lhs + " < 0) != (" + rhs + " < 0))))";
      else if (isa<arith::CeilDivSIOp>(op)) expression = "(" + lhs + " / " + rhs + " + (" + lhs + " % " + rhs + " != 0 && ((" + lhs + " < 0) == (" + rhs + " < 0))))";
      else if (isa<arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp>(op)) {
        std::string callee = isa<arith::MaximumFOp, arith::MaxNumFOp>(op) ? "fmax" : "fmin";
        if (!op->getResult(0).getType().isF64()) callee += "f";
        expression = callee + "(" + lhs + ", " + rhs + ")";
        if (isa<arith::MaximumFOp, arith::MinimumFOp>(op)) expression = "(isnan(" + lhs + ") || isnan(" + rhs + ") ? NAN : " + expression + ")";
      }
      if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp>(op)) {
        Type type = op->getResult(0).getType();
        // Arithmetic below 32 bits must not be promoted back to signed int
        // before multiplication or left shift. The final cast retains low bits.
        std::string carrier = type.isInteger(64) || type.isIndex() ? "uint64_t" : "uint32_t";
        expression = "static_cast<" + ctype(type) + ">(static_cast<" + carrier + ">(" + lhs + ") " + symbol + " static_cast<" + carrier + ">(" + rhs + "))";
      }
      if (unsignedOperands && !isa<arith::CmpIOp>(op))
        expression = "static_cast<" + ctype(op->getResult(0).getType()) + ">(" + expression + ")";
    }
    if (expression.empty() || op->getNumResults() != 1) return op->emitError("operation has no BANG C spelling");
    line(ctype(op->getResult(0).getType()) + " " + bind(op->getResult(0)) + " = " + expression + ";", depth);
    return success();
  }
  func::FuncOp function;
  llvm::raw_ostream &out;
  DenseMap<Value, std::string> names;
  unsigned next = 0;
  SmallVector<std::string> signature, call;
};
}
LogicalResult serializeProgram(ModuleOp module, std::string &source, std::string &metadata) {
  if (failed(verifyProgram(module))) return failure();
  llvm::raw_string_ostream output(source);
  llvm::json::Object interface;
  Serializer serializer(*module.getOps<func::FuncOp>().begin(), output);
  if (failed(serializer.emit(interface))) return failure();
  llvm::raw_string_ostream metadataOutput(metadata);
  metadataOutput << llvm::json::Value(std::move(interface));
  return success();
}
}
