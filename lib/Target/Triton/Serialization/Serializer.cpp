#include "Intent/Target/Triton/Serialization/Serializer.h"

#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalParameters.h"
#include "Intent/Dialect/GPU/Analysis/ProgramInterface.h"
#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/GPUTypes.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Serialization/Interface.h"
#include "Intent/Dialect/GPU/Serialization/Python.h"
#include "Intent/Target/Triton/IR/Configuration.h"
#include "Intent/Target/Triton/IR/TritonOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"

#include <set>
#include <optional>

using namespace mlir;

namespace intent::triton {
namespace {

std::string pythonType(Type type) {
  static const gpu::PythonScalarSyntax syntax{
      "tl.", "int1", "float64", "float8e4nv", "float8e5"};
  return gpu::pythonScalarType(type, syntax);
}

std::string literal(Attribute value) {
  return gpu::pythonLiteral(value);
}

class Serializer {
public:
  Serializer(func::FuncOp kernel, raw_ostream &output)
      : kernel(kernel), output(output) {}

  LogicalResult emit(std::string &metadata) {
    bindArguments();
    if (failed) return failure();
    emitPreamble();
    emitHelpers();
    emitKernel();
    return failed ? failure() : emitMetadata(metadata);
  }

private:
  std::string expressionString(gpu::PhysicalExprAttr expression) {
    static const gpu::PythonExpressionSyntax syntax{
        "", "min", "max", "", "triton.next_power_of_2", true};
    return gpu::pythonExpression(expression, syntax, [&](gpu::PhysicalExprAttr leaf) {
      if (leaf.getKind() == gpu::PhysicalExprKind::Parameter)
        return leaf.getParameterReference().getName().getValue().str();
      return valueString(gpu::resolveArgument(kernel, leaf.getArgumentReference()));
    });
  }

  std::string fragmentShape(gpu::FragmentType fragment) {
    SmallVector<std::string> extents;
    for (Attribute extent : fragment.getShape())
      extents.push_back(expressionString(cast<gpu::PhysicalExprAttr>(extent)));
    return stringTuple(extents);
  }

  using ViewABI = gpu::PythonArgument;
  using ScalarABI = gpu::PythonArgument;
  using MetadataABI = gpu::PythonArgument;
  struct DescriptorABI {
    TensorDescriptorOp operation;
    std::string name;
  };

  void bindArguments() {
    auto configurationSchema = ConfigurationSchema::read(kernel);
    if (mlir::failed(configurationSchema)) {
      failed = true;
      return;
    }
    configuration = std::move(*configurationSchema);
    auto interface = gpu::PythonSignature::read(kernel);
    if (mlir::failed(interface)) {
      failed = true;
      return;
    }
    views = std::move(interface->views);
    scalars = std::move(interface->scalars);
    metadataArguments = std::move(interface->metadata);
    for (const ScalarABI &scalar : scalars)
      values[scalar.value] = scalar.name;
    for (const MetadataABI &metadata : metadataArguments)
      values[metadata.value] = metadata.name;
    for (const ViewABI &view : views)
      values[view.value] = view.name;
    llvm::StringSet<> argumentNames;
    for (const auto &entry : values) argumentNames.insert(entry.second);
    for (const MetadataABI &metadata : metadataArguments)
      constexprValues.insert(metadata.value);
    for (Attribute attribute : gpu::getParameterDeclarations(kernel)) {
      auto schema = cast<gpu::ParameterAttr>(attribute);
      std::string name = schema.getName().getValue().str();
      argumentNames.insert(name);
      if (schema.isDeferred())
        coverageNames.insert(name);
    }
    kernel.walk([&](TensorDescriptorChoiceOp choice) {
      descriptorChoice = choice;
      values[choice.getResult()] = choice.getConfigParameter().getName().getValue().str();
      argumentNames.insert(choice.getConfigParameter().getName().getValue());
      argumentNames.insert(choice.getEligibilityArgument());
    });
    kernel.walk([&](TensorDescriptorAllocatorOp allocator) {
      descriptorAllocator = allocator;
    });
    while (!argumentNames.insert(metadataArgument).second)
      metadataArgument += "_";
    kernel.walk([&](TensorDescriptorOp descriptor) {
      std::string name = "_intent_descriptor_" + std::to_string(descriptors.size());
      while (!argumentNames.insert(name).second)
        name += "_";
      values[descriptor.getResult()] = name;
      descriptors.push_back({descriptor, name});
    });
    while (!argumentNames.insert(overlapArgument).second)
      overlapArgument += "_";
    kernel.walk([&](gpu::ViewOverlapOp overlap) {
      values[overlap.getResult()] = overlapArgument + "[" +
                                   std::to_string(overlapFacts.size()) + "]";
      overlapFacts.push_back(overlap);
      constexprValues.insert(overlap.getResult());
    });
    if (descriptorChoice && !descriptorAllocator) {
      kernel.emitError(
          "tensor descriptor form has no declared allocator requirement");
      failed = true;
    }
  }

  void emitPreamble() {
    output << "import triton\nimport triton.language as tl\n"
              "from triton.language.extra import libdevice\n"
              "from intent.runtime.triton.math import contract_fma\n\n";
  }

  void emitHelper(Operation *owner, Region &region, StringRef role) {
    std::string name =
        ("_intent_" + role + "_" + Twine(helperCounter++)).str();
    helperNames[owner].push_back(name);
    Block &block = region.front();
    output << "@triton.jit\ndef " << name << "(";
    for (auto [index, argument] : llvm::enumerate(block.getArguments())) {
      if (index)
        output << ", ";
      std::string argumentName = "a" + std::to_string(index);
      output << argumentName;
      values[argument] = argumentName;
    }
    output << "):\n";
    unsigned savedIndent = indent;
    bool savedEmittingHelper = emittingHelper;
    indent = 1;
    emittingHelper = true;
    for (Operation &operation : block.without_terminator())
      emitOperation(operation);
    auto yield = dyn_cast<gpu::YieldOp>(block.getTerminator());
    if (!yield) {
      failed = true;
    } else {
      std::string result = "return ";
      if (yield.getValues().size() > 1)
        result += "(";
      for (auto [index, value] : llvm::enumerate(yield.getValues())) {
        if (index)
          result += ", ";
        result += valueString(value);
      }
      if (yield.getValues().size() > 1)
        result += ")";
      line(result);
    }
    indent = savedIndent;
    emittingHelper = savedEmittingHelper;
    output << "\n";
  }

  void emitHelpers() {
    kernel.walk([&](Operation *operation) {
      if (auto reduce = dyn_cast<ReduceOp>(operation))
        emitHelper(operation, reduce.getCombine(), "reduce");
      else if (auto scan = dyn_cast<ScanOp>(operation))
        emitHelper(operation, scan.getCombine(), "scan");
      else if (auto map = dyn_cast<MapElementwiseOp>(operation))
        emitHelper(operation, map.getBody(), "map");
    });
  }

  void emitKernel() {
    output << "@triton.jit\ndef _intent_kernel(";
    bool first = true;
    for (const ViewABI &view : views) {
      if (!first)
        output << ", ";
      first = false;
      output << view.name;
    }
    for (const ScalarABI &scalar : scalars) {
      if (!first)
        output << ", ";
      first = false;
      output << scalar.name;
      output << ": " << pythonType(scalar.value.getType());
    }
    if (!metadataArguments.empty()) {
      if (!first)
        output << ", ";
      first = false;
      output << metadataArgument << ": tl.constexpr";
    }
    if (!overlapFacts.empty()) {
      if (!first)
        output << ", ";
      first = false;
      output << overlapArgument << ": tl.constexpr";
    }
    if (descriptorChoice) {
      if (!first)
        output << ", ";
      first = false;
      output << descriptorChoice.getEligibilityArgument() << ": tl.constexpr";
    }
    for (StringAttr parameter : configuration.kernelParameters)
      output << ", " << parameter.getValue() << ": tl.constexpr";
    for (const DescriptorABI &descriptor : descriptors)
      output << ", " << descriptor.name;
    output << "):\n";
    indent = 1;
    for (auto [index, metadata] : llvm::enumerate(metadataArguments))
      line(metadata.name + ": tl.constexpr = " + metadataArgument + "[" +
           std::to_string(index) + "]");
    line(configuration.warps.getValue().str() +
         ": tl.constexpr = tl.extra.cuda.num_warps()");
    emitBlock(kernel.getBody().front(), /*isLoop=*/false, {});
    output << "\n";
  }

  void emitBlock(Block &block, bool isLoop,
                 ArrayRef<std::string> loopResults) {
    auto begin = output.tell();
    for (Operation &operation : block) {
      if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
        if (!loopResults.empty()) {
          SmallVector<std::string> yielded;
          for (Value value : yield.getOperands())
            yielded.push_back(controlValueString(value));
          if (loopResults.size() == 1)
            line(loopResults.front() + " = " + yielded.front());
          else
            line(stringTuple(loopResults) + " = " + stringTuple(yielded));
        }
        continue;
      }
      if (isa<func::ReturnOp>(operation))
        continue;
      emitOperation(operation);
    }
    if (output.tell() == begin)
      line("pass");
  }

  void emitOperation(Operation &operation) {
    if (auto assertion = dyn_cast<cf::AssertOp>(operation)) {
      if (!constexprValues.contains(assertion.getArg())) {
        assertion.emitOpError("Triton static assertion condition is not constexpr");
        failed = true;
        return;
      }
      std::string message;
      llvm::raw_string_ostream(message) << llvm::json::Value(assertion.getMsg());
      line("tl.static_assert(" + valueString(assertion.getArg()) + ", " + message + ")");
      return;
    }
    if (isa<CtaBarrierOp>(operation)) {
      line("tl.debug_barrier()");
      return;
    }
    if (isa<gpu::ViewOverlapOp, TensorDescriptorChoiceOp, TensorDescriptorAllocatorOp,
            TensorDescriptorOp>(operation))
      return;
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      std::string value = literal(constant.getValue());
      if (auto floating = dyn_cast<FloatAttr>(constant.getValue())) {
        value = "tl.full((), " + value + ", " +
                pythonType(constant.getType()) + ")";
        // Triton's scalar constructor canonicalizes both zero signs to +0.
        if (floating.getValue().isNegZero())
          value = "(-" + value + ")";
        assign(constant.getResult(), value);
      } else {
        values[constant.getResult()] = value;
        constexprValues.insert(constant.getResult());
      }
      return;
    }
    if (auto parameter = dyn_cast<gpu::ParameterOp>(operation)) {
      values[parameter.getResult()] =
          parameter.getReference().getName().getValue().str();
      constexprValues.insert(parameter.getResult());
      return;
    }
    if (auto physical = dyn_cast<gpu::PhysicalExprOp>(operation)) {
      assign(physical.getResult(),
             expressionString(physical.getExpression()),
             isConstexprExpression(physical.getExpression()));
      return;
    }
    if (auto program = dyn_cast<gpu::ProgramIdOp>(operation)) {
      assign(program.getResult(),
             "tl.program_id(" + std::to_string(program.getAxis()) + ")");
      return;
    }
    if (auto coordinate = dyn_cast<gpu::WorksetCoordinateOp>(operation)) {
      values[coordinate.getResult()] = valueString(coordinate.getCoordinate());
      return;
    }
    if (auto dim = dyn_cast<gpu::DimOp>(operation)) {
      auto view = dim.getView().getType();
      auto extent = cast<gpu::PhysicalExprAttr>(
          view.getLayout().getExtents()[dim.getAxis()]);
      assign(dim.getResult(), expressionString(extent),
             isConstexprExpression(extent));
      return;
    }
    if (auto range = dyn_cast<gpu::RangeOp>(operation)) {
      assign(range.getResult(), "(" + valueString(range.getStart()) + ", " +
                                    valueString(range.getStop()) + ", " +
                                    valueString(range.getStep()) + ")");
      return;
    }
    if (auto bound = dyn_cast<gpu::RangeBoundOp>(operation)) {
      assign(bound.getResult(), valueString(bound.getRange()) + "[" +
                                    std::to_string(bound.getBound()) + "]");
      return;
    }
    if (auto mapping = dyn_cast<gpu::DelinearizeOp>(operation)) {
      std::string remaining = valueString(mapping.getLinear());
      SmallVector<std::string> coordinates(mapping.getCoordinates().size());
      for (int64_t axis = static_cast<int64_t>(mapping.getCoordinates().size()) - 1;
           axis >= 0; --axis) {
        std::string extent = valueString(mapping.getExtents()[axis]);
        coordinates[axis] = "(" + remaining + " % " + extent + ")";
        remaining = "(" + remaining + " // " + extent + ")";
      }
      for (auto [coordinate, expression] :
           llvm::zip(mapping.getCoordinates(), coordinates))
        assign(coordinate, expression);
      return;
    }
    if (auto binary = dyn_cast<gpu::BinaryOp>(operation)) {
      assign(binary.getResult(), binaryExpression(binary),
             (binary.getResult().getType().isIndex() ||
              binary.getResult().getType().isInteger(1)) &&
                 constexprValues.contains(binary.getLhs()) &&
                 constexprValues.contains(binary.getRhs()));
      return;
    }
    if (auto unary = dyn_cast<gpu::UnaryOp>(operation)) {
      assign(unary.getResult(), unaryExpression(unary));
      return;
    }
    if (auto compare = dyn_cast<gpu::CompareOp>(operation)) {
      auto predicate = [&]() -> StringRef {
        switch (compare.getPredicate()) {
        case ComparePredicate::Eq: return "==";
        case ComparePredicate::Ne: return "!=";
        case ComparePredicate::Lt: return "<";
        case ComparePredicate::Le: return "<=";
        case ComparePredicate::Gt: return ">";
        case ComparePredicate::Ge: return ">=";
        }
        llvm_unreachable("unhandled Intent compare predicate");
      }();
      assign(compare.getResult(), "(" + valueString(compare.getLhs()) + " " +
                                      predicate.str() + " " +
                                      valueString(compare.getRhs()) + ")",
             compare.getResult().getType().isInteger(1) &&
                 constexprValues.contains(compare.getLhs()) &&
                 constexprValues.contains(compare.getRhs()));
      return;
    }
    if (auto range = dyn_cast<gpu::MakeRangeOp>(operation)) {
      auto fragment = cast<gpu::FragmentType>(range.getResult().getType());
      auto physicalExtent =
          cast<gpu::PhysicalExprAttr>(fragment.getShape()[0]);
      assign(range.getResult(), "(" + valueString(range.getStart()) +
                                    " + tl.arange(0, " +
                                    expressionString(physicalExtent) + ") * " +
                                    valueString(range.getStep()) + ")");
      return;
    }
    if (auto splat = dyn_cast<gpu::SplatOp>(operation)) {
      // Triton reduce/scan helpers receive accumulator values directly and
      // scalar operands broadcast through ordinary elementwise expressions.
      // Their ABI has no outer-kernel constexpr shape parameters, so spelling
      // a typed scalar splat with fragmentShape() would reference symbols that
      // are intentionally outside the helper scope.
      if (emittingHelper) {
        assign(splat.getResult(), valueString(splat.getValue()));
        return;
      }
      auto type = splat.getResult().getType();
      assign(splat.getResult(), "tl.full(" + fragmentShape(type) + ", " +
                                    valueString(splat.getValue()) + ", " +
                                    pythonType(type.getElementType()) + ")");
      return;
    }
    if (auto broadcast = dyn_cast<gpu::BroadcastOp>(operation)) {
      assign(broadcast.getResult(), broadcastValue(broadcast.getValue(),
                                                   broadcast.getResult().getType()));
      return;
    }
    if (auto cast = dyn_cast<gpu::CastOp>(operation)) {
      assign(cast.getResult(), "tl.cast(" + valueString(cast.getValue()) +
                                   ", " +
                                   pythonType(elementType(cast.getResult().getType())) +
                                   ")");
      return;
    }
    if (auto bitcast = dyn_cast<gpu::BitcastOp>(operation)) {
      assign(bitcast.getResult(), "tl.cast(" + valueString(bitcast.getValue()) +
                                      ", " +
                                      pythonType(elementType(bitcast.getResult().getType())) +
                                      ", bitcast=True)");
      return;
    }
    if (auto select = dyn_cast<gpu::SelectOp>(operation)) {
      assign(select.getResult(), "tl.where(" + valueString(select.getCondition()) +
                                      ", " + controlValueString(select.getTrueValue()) +
                                      ", " + controlValueString(select.getFalseValue()) + ")");
      return;
    }
    if (auto map = dyn_cast<MapElementwiseOp>(operation)) {
      bool scalar = map.getResult().getType().getShape().empty();
      std::string call = scalar ? helperNames[&operation].front() + "("
                                : "tl.map_elementwise(" + helperNames[&operation].front();
      for (auto [index, input] : llvm::enumerate(map.getInputs())) {
        if (!scalar || index)
          call += ", ";
        call += controlValueString(input);
      }
      assign(map.getResult(), call + ")");
      return;
    }
    if (auto reshape = dyn_cast<gpu::ReshapeOp>(operation)) {
      auto source = cast<gpu::FragmentType>(reshape.getValue().getType());
      auto target = cast<gpu::FragmentType>(reshape.getResult().getType());
      if (source.getShape().empty()) {
        assign(reshape.getResult(), "tl.broadcast_to(" +
                                         valueString(reshape.getValue()) + ", " +
                                         fragmentShape(target) + ")");
        return;
      }
      assign(reshape.getResult(), "tl.reshape(" + valueString(reshape.getValue()) +
                                       ", " + fragmentShape(target) +
                                       ", can_reorder=False)");
      return;
    }
    if (auto transpose = dyn_cast<gpu::TransposeOp>(operation)) {
      std::string permutation = "(";
      for (auto [index, axis] : llvm::enumerate(transpose.getPermutation())) {
        if (index)
          permutation += ", ";
        permutation += std::to_string(axis);
      }
      if (transpose.getPermutation().size() == 1)
        permutation += ",";
      permutation += ")";
      assign(transpose.getResult(), "tl.permute(" + valueString(transpose.getValue()) +
                                         ", " + permutation + ")") ;
      return;
    }
    if (auto join = dyn_cast<gpu::JoinOp>(operation)) {
      assign(join.getResult(), "tl.join(" + valueString(join.getLhs()) + ", " +
                                   valueString(join.getRhs()) + ")");
      return;
    }
    if (auto split = dyn_cast<SplitOp>(operation)) {
      assignResults(split.getResults(),
                    "tl.split(" + valueString(split.getSource()) + ")");
      return;
    }
    if (auto record = dyn_cast<gpu::MakeRecordOp>(operation)) {
      std::string tuple = "(";
      for (auto [index, field] : llvm::enumerate(record.getFields())) {
        if (index)
          tuple += ", ";
        // Records can become loop carries; preserve each declared scalar dtype
        // instead of letting Triton infer a narrower type from a literal.
        tuple += controlValueString(field);
      }
      if (record.getFields().size() == 1)
        tuple += ",";
      assign(record.getResult(), tuple + ")");
      return;
    }
    if (auto extract = dyn_cast<gpu::ExtractOp>(operation)) {
      assign(extract.getResult(), valueString(extract.getRecord()) + "[" +
                                       std::to_string(extract.getField()) + "]");
      return;
    }
    if (auto load = dyn_cast<DescriptorLoadOp>(operation)) {
      assign(load.getResult(),
             valueString(load.getDescriptor()) + ".load(" +
                 descriptorOffsets(load.getOffsets()) + ")");
      return;
    }
    if (auto load = dyn_cast<BlockLoadOp>(operation)) {
      auto fragment = load.getResult().getType();
      std::string call = "tl.load(" +
                         blockPointer(load.getView(), load.getOffsets(),
                                      load.getBlockAxes(), load.getOrder(),
                                      fragment);
      if (!load.getBoundaryAxes().empty())
        call += ", boundary_check=" + axisTuple(load.getBoundaryAxes()) +
                ", padding_option=\"" + load.getPadding().str() + "\"";
      call += ")";
      if (fragment.getElementType().isInteger(1))
        call = "tl.cast(" + call + ", tl.int1)";
      assign(load.getResult(), call);
      return;
    }
    if (auto load = dyn_cast<gpu::LoadOp>(operation)) {
      auto access = cast<gpu::AccessOpInterface>(load.getOperation());
      std::string call = "tl.load(" + pointer(access);
      if (access.getAccessValidity())
        call += ", mask=" + valueString(access.getAccessValidity()) +
                ", other=" + valueString(access.getAccessFill());
      assign(access.getAccessResult(), call + ")");
      return;
    }
    if (auto gather = dyn_cast<gpu::GatherOp>(operation)) {
      auto access = cast<gpu::AccessOpInterface>(gather.getOperation());
      Value coordinate = access.getAccessCoordinates().front();
      std::string indices = valueString(coordinate);
      bool scalar = !isa<gpu::FragmentType>(access.getAccessValueType());
      if (scalar)
        indices = "tl.full((1,), " + indices + ", " +
                  pythonType(coordinate.getType()) + ")";
      std::string call = "tl.gather(" + valueString(access.getAccessResource()) + ", " +
                         indices +
                         ", axis=" +
                         std::to_string(access.getAccessSourceAxes().front()) + ")";
      if (scalar)
        call = "tl.reshape(" + call + ", ())";
      assign(access.getAccessResult(), call);
      return;
    }
    if (auto contract = dyn_cast<gpu::ContractOp>(operation)) {
      auto form = contract->getAttrOfType<StringAttr>(
          "intent_gpu.triton.contract_form");
      if (form && form.getValue() == "fma") {
        assign(contract.getResult(),
               "contract_fma(" + valueString(contract.getLhs()) + ", " +
                   valueString(contract.getRhs()) + ", " +
                   valueString(contract.getAccumulator()) + ")");
        return;
      }
      if (form && form.getValue() == "multiply_sum") {
        unsigned rank = contract.getLhs().getType().getShape().size();
        std::string element = pythonType(elementType(contract.getResult().getType()));
        std::string lhs = "tl.expand_dims(" + valueString(contract.getLhs()) +
                          ", axis=" + std::to_string(rank) + ").to(" +
                          element + ")";
        std::string rhs = "tl.expand_dims(" + valueString(contract.getRhs()) +
                          ", axis=" + std::to_string(rank - 2) + ").to(" +
                          element + ")";
        std::string reduced =
            "tl.sum((" + lhs + " * " + rhs + "), axis=" +
            std::to_string(rank - 1) + ")";
        assign(contract.getResult(), "(" + reduced + " + " +
                                         valueString(contract.getAccumulator()) +
                                         ")");
        return;
      }
      assign(contract.getResult(), "tl.dot(" + valueString(contract.getLhs()) +
                                      ", " + valueString(contract.getRhs()) +
                                      ", " + valueString(contract.getAccumulator()) +
                                      ", input_precision=\"ieee\", max_num_imprecise_acc=0, out_dtype=" +
                                      pythonType(contract.getResult()
                                                     .getType()
                                                     .getElementType()) + ")");
      return;
    }
    if (auto contract = dyn_cast<gpu::ScaledContractOp>(operation)) {
      auto format = [](ScaledFormat value) -> StringRef {
        switch (value) {
        case ScaledFormat::E2M1: return "e2m1";
        case ScaledFormat::E4M3: return "e4m3";
        case ScaledFormat::E8M0: return "e8m0";
        }
        llvm_unreachable("unhandled Intent scaled format");
      };
      auto extent = [&](gpu::FragmentType type, unsigned axis) {
        return expressionString(
            cast<gpu::PhysicalExprAttr>(type.getShape()[axis]));
      };
      auto lhs = contract.getLhs().getType();
      auto rhs = contract.getRhs().getType();
      // The verified scaled-contract schema already fixes contiguous groups
      // and packed carriers. dot_scaled spells that grouped K as one axis.
      std::string lhsValue =
          "tl.reshape(" + valueString(contract.getLhs()) + ", (" +
          extent(lhs, 0) + ", " + extent(lhs, 1) + " * " + extent(lhs, 2) + "))";
      std::string rhsValue =
          "tl.reshape(" + valueString(contract.getRhs()) + ", (" +
          extent(rhs, 0) + " * " + extent(rhs, 1) + ", " + extent(rhs, 2) + "))";
      assign(contract.getResult(),
             "tl.dot_scaled(" + lhsValue + ", " +
                 valueString(contract.getLhsScale()) + ", \"" +
                 format(contract.getLhsFormat()).str() + "\", " +
                 rhsValue + ", " +
                 valueString(contract.getRhsScale()) + ", \"" +
                 format(contract.getRhsFormat()).str() + "\", " +
                 valueString(contract.getAccumulator()) + ")");
      return;
    }
    if (isa<ReduceOp, ScanOp>(operation)) {
      auto reduce = dyn_cast<ReduceOp>(operation);
      auto scan = dyn_cast<ScanOp>(operation);
      ValueRange sourceValues = reduce ? reduce.getSources() : scan.getSources();
      unsigned count = sourceValues.size();
      int64_t axis = reduce ? reduce.getAxis() : scan.getAxis();
      std::string sources = count == 1 ? valueString(sourceValues.front()) : "(";
      if (count != 1) {
        for (unsigned i = 0; i < count; ++i) {
          if (i) sources += ", ";
          sources += valueString(sourceValues[i]);
        }
        sources += ")";
      }
      std::string call = (reduce ? "tl.reduce(" : "tl.associative_scan(") + sources +
          ", axis=" + std::to_string(axis) + ", combine_fn=" + helperNames.lookup(&operation).front();
      if (scan) call += std::string(", reverse=") + (scan.getReverse() ? "True" : "False");
      assignResults(operation.getResults(), call + ")");
      return;
    }
    if (auto histogram = dyn_cast<gpu::HistogramOp>(operation)) {
      std::string counts = "tl.histogram(" +
                           valueString(histogram.getValues()) + ", " +
                           valueString(histogram.getBins()) + ", mask=" +
                           valueString(histogram.getValid()) + ")";
      Type resultElement = histogram.getResult().getType().getElementType();
      assign(histogram.getResult(),
             "tl.cast(" + counts + ", " + pythonType(resultElement) + ")");
      return;
    }
    if (auto random = dyn_cast<gpu::RandomBitsOp>(operation)) {
      assign(random.getResult(), "tl.randint(" + valueString(random.getSeed()) +
                                      ", " + valueString(random.getCounter()) +
                                      ").to(tl.uint32, bitcast=True)");
      return;
    }
    if (auto atomic = dyn_cast<gpu::AtomicStoreOp>(operation)) {
      auto access = cast<gpu::AccessOpInterface>(atomic.getOperation());
      std::string call = "tl.atomic_xchg(" +
                         pointer(access) + ", " + valueString(access.getAccessPayloads().front());
      if (access.getAccessValidity())
        call += ", mask=" + valueString(access.getAccessValidity());
      call += ", sem=\"" + atomicSemantics(atomic.getOrdering()) +
              "\", scope=\"" + atomicScope(atomic.getSharing()) + "\")";
      line(call);
      return;
    }
    if (auto atomic = dyn_cast<gpu::AtomicRMWOp>(operation)) {
      auto access = cast<gpu::AccessOpInterface>(atomic.getOperation());
      auto operationName = [](AtomicRMWKind kind) -> StringRef {
        switch (kind) {
        case AtomicRMWKind::Exchange: return "xchg";
        case AtomicRMWKind::Add: return "add";
        case AtomicRMWKind::Maximum: return "max";
        case AtomicRMWKind::Minimum: return "min";
        case AtomicRMWKind::BitwiseAnd: return "and";
        case AtomicRMWKind::BitwiseOr: return "or";
        case AtomicRMWKind::BitwiseXor: return "xor";
        }
        llvm_unreachable("unhandled Intent atomic RMW kind");
      };
      std::string call = "tl.atomic_" + operationName(atomic.getKind()).str() +
                         "(" +
                         pointer(access) + ", " + valueString(access.getAccessPayloads().front());
      if (access.getAccessValidity())
        call += ", mask=" + valueString(access.getAccessValidity());
      call += ", sem=\"" + atomicSemantics(atomic.getOrdering()) +
              "\", scope=\"" + atomicScope(atomic.getSharing()) + "\")";
      assign(atomic.getResult(), call);
      return;
    }
    if (auto atomic = dyn_cast<gpu::AtomicCompareExchangeOp>(operation)) {
      auto access = cast<gpu::AccessOpInterface>(atomic.getOperation());
      auto payloads = access.getAccessPayloads();
      std::string old = newName();
      line(old + " = tl.atomic_cas(" +
           pointer(access) + ", " + valueString(payloads[0]) + ", " +
           valueString(payloads[1]) + ", sem=\"" +
           atomicSemantics(atomic.getOrdering()) + "\", scope=\"" +
           atomicScope(atomic.getSharing()) + "\")");
      assign(atomic.getResult(), "(" + old + ", (" + old + " == " +
                                     valueString(payloads[0]) + "))");
      return;
    }
    if (auto store = dyn_cast<DescriptorStoreOp>(operation)) {
      auto fragment = cast<gpu::FragmentType>(store.getValue().getType());
      line(valueString(store.getDescriptor()) + ".store(" +
           descriptorOffsets(store.getOffsets()) +
           ", tl.cast(" + valueString(store.getValue()) + ", " +
           pythonType(fragment.getElementType()) + "))");
      return;
    }
    if (auto store = dyn_cast<BlockStoreOp>(operation)) {
      auto fragment = cast<gpu::FragmentType>(store.getValue().getType());
      std::string storageType = fragment.getElementType().isInteger(1)
                                    ? "tl.int8"
                                    : pythonType(fragment.getElementType());
      std::string call =
          "tl.store(" +
          blockPointer(store.getView(), store.getOffsets(),
                       store.getBlockAxes(), store.getOrder(), fragment) +
          ", tl.cast(" + valueString(store.getValue()) + ", " +
          storageType + ")";
      if (!store.getBoundaryAxes().empty())
        call += ", boundary_check=" + axisTuple(store.getBoundaryAxes());
      line(call + ")");
      return;
    }
    if (auto store = dyn_cast<gpu::StoreOp>(operation)) {
      auto access = cast<gpu::AccessOpInterface>(store.getOperation());
      std::string call = "tl.store(" +
                         pointer(access) + ", " + valueString(access.getAccessPayloads().front());
      if (access.getAccessValidity())
        call += ", mask=" + valueString(access.getAccessValidity());
      line(call + ")");
      return;
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      SmallVector<std::string> results;
      for (auto [result, initial] : llvm::zip(loop.getResults(), loop.getInitArgs())) {
        std::string name = newName();
        values[result] = name;
        line(name + " = " + controlValueString(initial));
        results.push_back(name);
      }
      std::string induction = "iv" + std::to_string(counter++);
      values[loop.getInductionVar()] = induction;
      for (auto [argument, name] : llvm::zip(loop.getRegionIterArgs(), results))
        values[argument] = name;
      auto unroll = loop->getAttrOfType<IntegerAttr>(
          "intent_gpu.triton.loop_unroll_factor");
      auto stages = loop->getAttrOfType<gpu::ParameterRefAttr>(loopStagesAttr);
      std::string range = unroll || stages ? "tl.range(" : "range(";
      range += valueString(loop.getLowerBound()) + ", " +
               valueString(loop.getUpperBound()) + ", " +
               valueString(loop.getStep());
      if (unroll)
        range += ", loop_unroll_factor=" + std::to_string(unroll.getInt());
      if (stages)
        range += ", num_stages=" + stages.getName().getValue().str();
      line("for " + induction + " in " + range + "):");
      ++indent;
      emitBlock(*loop.getBody(), true, results);
      --indent;
      return;
    }
    if (auto ifOperation = dyn_cast<scf::IfOp>(operation)) {
      SmallVector<std::string> results;
      for (Value result : ifOperation.getResults()) {
        std::string name = newName();
        values[result] = name;
        results.push_back(name);
      }
      line("if " + valueString(ifOperation.getCondition()) + ":");
      ++indent;
      emitBlock(ifOperation.getThenRegion().front(), false, results);
      --indent;
      if (!ifOperation.getElseRegion().empty() &&
          !ifOperation.getElseRegion().front().empty()) {
        line("else:");
        ++indent;
        emitBlock(ifOperation.getElseRegion().front(), false, results);
        --indent;
      }
      return;
    }
    if (auto whileOperation = dyn_cast<scf::WhileOp>(operation)) {
      SmallVector<std::string> carries;
      for (auto [result, initial] :
           llvm::zip(whileOperation.getResults(), whileOperation.getInits())) {
        std::string name = newName();
        values[result] = name;
        line(name + " = " + controlValueString(initial));
        carries.push_back(name);
      }
      Block &before = whileOperation.getBefore().front();
      for (auto [argument, name] : llvm::zip(before.getArguments(), carries))
        values[argument] = name;
      std::string active = newName();
      line(active + " = tl.full((), True, tl.int1)");
      line("while " + active + ":");
      ++indent;
      for (Operation &nested : before.without_terminator())
        emitOperation(nested);
      auto condition = cast<scf::ConditionOp>(before.getTerminator());
      line(active + " = " + valueString(condition.getCondition()));
      line("if " + active + ":");
      ++indent;
      Block &after = whileOperation.getAfter().front();
      for (auto [argument, forwarded] :
           llvm::zip(after.getArguments(), condition.getArgs()))
        values[argument] = controlValueString(forwarded);
      emitBlock(after, true, carries);
      --indent;
      --indent;
      return;
    }
    operation.emitOpError("has no terminal Triton spelling");
    failed = true;
  }

  Type elementType(Type type) const {
    if (auto fragment = dyn_cast<gpu::FragmentType>(type))
      return fragment.getElementType();
    return type;
  }

  std::string binaryExpression(gpu::BinaryOp binary) {
    auto infix = [&](StringRef spelling) {
      return "(" + valueString(binary.getLhs()) + " " + spelling.str() + " " +
             valueString(binary.getRhs()) + ")";
    };
    auto call = [&](StringRef function, StringRef propagateNan = {}) {
      std::string result = function.str() + "(" + valueString(binary.getLhs()) +
                           ", " + valueString(binary.getRhs());
      if (!propagateNan.empty())
        result += ", propagate_nan=tl.PropagateNan." + propagateNan.str();
      return result + ")";
    };
    if (binary.getResult().getType().isIndex() &&
        constexprValues.contains(binary.getLhs()) &&
        constexprValues.contains(binary.getRhs())) {
      if (binary.getOperatorKind() == BinaryOperator::Maximum)
        return call("max");
      if (binary.getOperatorKind() == BinaryOperator::Minimum)
        return call("min");
    }
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add: return infix("+");
    case BinaryOperator::Subtract: return infix("-");
    case BinaryOperator::Multiply: return infix("*");
    case BinaryOperator::TrueDivide: {
      if (binary.getApproximate())
        return "tl.inline_asm_elementwise(\"div.approx" +
               std::string(binary.getFlushToZero() ? ".ftz" : "") +
               ".f32 $0, $1, $2;\", constraints=\"=f,f,f\", args=[" +
               valueString(binary.getLhs()) + ", " + valueString(binary.getRhs()) +
               "], dtype=tl.float32, is_pure=True, pack=1)";
      Type element = elementType(binary.getResult().getType());
      auto floating = cast<FloatType>(element);
      std::string computation = floating.getWidth() < 32
                                    ? "tl.float32" : pythonType(element);
      std::string result =
          "libdevice.div_rn(tl.cast(" + valueString(binary.getLhs()) + ", " +
          computation + "), tl.cast(" + valueString(binary.getRhs()) + ", " +
          computation + "))";
      return floating.getWidth() < 32
                 ? "tl.cast(" + result + ", " + pythonType(element) + ")"
                 : result;
    }
    case BinaryOperator::FloorDivide:
    case BinaryOperator::Remainder: {
      Type element = elementType(binary.getResult().getType());
      auto integer = dyn_cast<IntegerType>(element);
      bool remainder = binary.getOperatorKind() == BinaryOperator::Remainder;
      if (integer && integer.isUnsigned())
        return infix(remainder ? "%" : "//");
      std::string rem = infix("%");
      std::string rhs = valueString(binary.getRhs());
      // Triton integer division truncates. A Python constexpr remainder already
      // has the divisor's sign, so this correction also preserves constexprs.
      std::string adjust = "((" + rem + " != 0) & ((" + rem +
                           " < 0) != (" + rhs + " < 0)))";
      return remainder ? "(" + rem + " + " + adjust + " * " + rhs + ")"
                       : "(" + infix("//") + " - " + adjust + ")";
    }
    case BinaryOperator::Power: return call("libdevice.pow");
    case BinaryOperator::Maximum: return call("tl.maximum", "ALL");
    case BinaryOperator::Minimum: return call("tl.minimum", "ALL");
    case BinaryOperator::MaximumNum: return call("tl.maximum", "NONE");
    case BinaryOperator::MinimumNum: return call("tl.minimum", "NONE");
    case BinaryOperator::LogicalAnd:
    case BinaryOperator::BitwiseAnd: return infix("&");
    case BinaryOperator::LogicalOr:
    case BinaryOperator::BitwiseOr: return infix("|");
    case BinaryOperator::BitwiseXor: return infix("^");
    case BinaryOperator::LeftShift: return infix("<<");
    case BinaryOperator::RightShift: return infix(">>");
    }
    llvm_unreachable("unhandled Intent binary operator");
  }

  std::string unaryExpression(gpu::UnaryOp unary) {
    std::string input = valueString(unary.getInput());
    auto libraryCall = [&](StringRef function) {
      Type element = elementType(unary.getResult().getType());
      auto floating = cast<FloatType>(element);
      std::string computation = floating.getWidth() < 32
                                    ? "tl.float32" : pythonType(element);
      std::string result = function.str() + "(tl.cast(" + input + ", " +
                           computation + "))";
      return floating.getWidth() < 32
                 ? "tl.cast(" + result + ", " + pythonType(element) + ")"
                 : result;
    };
    switch (unary.getOperatorKind()) {
    case UnaryOperator::Negate:
      return "(-" + input + ")";
    case UnaryOperator::Not:
      return "(~" + input + ")";
    case UnaryOperator::Exp:
      return libraryCall("libdevice.exp");
    case UnaryOperator::Exp2:
      if (unary.getApproximate())
        return "tl.inline_asm_elementwise(\"ex2.approx" +
               std::string(unary.getFlushToZero() ? ".ftz" : "") +
               ".f32 $0, $1;\", constraints=\"=f,f\", args=[" + input +
               "], dtype=tl.float32, is_pure=True, pack=1)";
      return libraryCall("libdevice.exp2");
    case UnaryOperator::Log:
      return libraryCall("libdevice.log");
    case UnaryOperator::Log1p:
      return libraryCall("libdevice.log1p");
    case UnaryOperator::Lgamma:
      return libraryCall("libdevice.lgamma");
    case UnaryOperator::Sin:
      return libraryCall("libdevice.sin");
    case UnaryOperator::Asin:
      return libraryCall("libdevice.asin");
    case UnaryOperator::Cos:
      return libraryCall("libdevice.cos");
    case UnaryOperator::Floor:
      return "tl.floor(" + input + ")";
    case UnaryOperator::Erf:
      return libraryCall("libdevice.erf");
    case UnaryOperator::Erfc:
      return libraryCall("libdevice.erfc");
    case UnaryOperator::I0:
      return libraryCall("libdevice.cyl_bessel_i0");
    case UnaryOperator::Rsqrt:
      return libraryCall("libdevice.rsqrt");
    case UnaryOperator::Sigmoid:
      return "tl.sigmoid(" + input + ")";
    case UnaryOperator::Tanh:
      if (unary.getApproximate())
        return "tl.inline_asm_elementwise(\"tanh.approx.f32 $0, $1;\", "
               "constraints=\"=f,f\", args=[" + input +
               "], dtype=tl.float32, is_pure=True, pack=1)";
      return libraryCall("libdevice.tanh");
    case UnaryOperator::Abs:
      return "tl.abs(" + input + ")";
    case UnaryOperator::Sqrt:
      return libraryCall("libdevice.sqrt_rn");
    default:
      failed = true;
      return "<unsupported-unary>";
    }
  }

  std::string atomicSemantics(AtomicOrdering ordering) const {
    switch (ordering) {
    case AtomicOrdering::Relaxed: return "relaxed";
    case AtomicOrdering::Acquire: return "acquire";
    case AtomicOrdering::Release: return "release";
    case AtomicOrdering::AcquireRelease: return "acq_rel";
    }
    llvm_unreachable("unhandled Intent atomic ordering");
  }

  std::string atomicScope(gpu::AtomicSharingDomain sharing) const {
    switch (sharing) {
    case gpu::AtomicSharingDomain::ProgramInstance: return "cta";
    case gpu::AtomicSharingDomain::KernelInvocation: return "gpu";
    }
    llvm_unreachable("unhandled atomic sharing domain");
  }

  std::string broadcastValue(Value value, gpu::FragmentType target,
                             std::optional<unsigned> coordinateAxis = std::nullopt) {
    auto source = dyn_cast<gpu::FragmentType>(value.getType());
    if (!source)
      return "tl.full(" + fragmentShape(target) + ", " + valueString(value) +
             ", " + pythonType(elementType(value.getType())) + ")";
    if (source == target)
      return valueString(value);
    gpu::BroadcastProjection projection;
    if (coordinateAxis) {
      projection.state = gpu::BroadcastProjectionState::Exact;
      projection.targetToSource.resize(target.getShape().size());
      projection.targetToSource[*coordinateAxis] = 0;
    } else {
      projection = gpu::queryBroadcastProjection(source, target);
    }
    if (!projection.isExact()) {
      kernel.emitError("Triton broadcast lost its shared axis projection")
          << "; source=" << source << "; target=" << target
          << "; value=" << value;
      failed = true;
      return {};
    }
    if (source.getShape().size() == target.getShape().size())
      return "tl.broadcast_to(" + valueString(value) + ", " +
             fragmentShape(target) + ")";
    SmallVector<std::string> selectors;
    for (std::optional<unsigned> sourceIndex : projection.targetToSource)
      selectors.push_back(sourceIndex ? ":" : "None");
    std::string result = valueString(value) + "[";
    for (auto [index, selector] : llvm::enumerate(selectors)) {
      if (index)
        result += ", ";
      result += selector;
    }
    return "tl.broadcast_to(" + result + "], " + fragmentShape(target) + ")";
  }

  std::string axisTuple(ArrayRef<int64_t> axes) const {
    std::string result = "(";
    for (auto [index, axis] : llvm::enumerate(axes)) {
      if (index)
        result += ", ";
      result += std::to_string(axis);
    }
    if (axes.size() == 1)
      result += ",";
    return result + ")";
  }

  std::string stringTuple(ArrayRef<std::string> values) const {
    std::string result = "(";
    for (auto [index, value] : llvm::enumerate(values)) {
      if (index)
        result += ", ";
      result += value;
    }
    if (values.size() == 1)
      result += ",";
    return result + ")";
  }

  std::string stringList(ArrayRef<std::string> values) const {
    std::string result = "[";
    for (auto [index, value] : llvm::enumerate(values)) {
      if (index)
        result += ", ";
      result += value;
    }
    return result + "]";
  }

  std::string descriptorOffsets(ValueRange offsets) {
    SmallVector<std::string> expressions;
    for (Value offset : offsets)
      expressions.push_back("tl.cast(" + valueString(offset) + ", tl.int32)");
    return stringList(expressions);
  }

  std::string blockPointer(Value viewValue, ValueRange offsets,
                           ArrayRef<int64_t> blockAxes,
                           ArrayRef<int64_t> order,
                           gpu::FragmentType fragment) {
    auto view = cast<gpu::ViewType>(viewValue.getType());
    auto strides = view.getLayout().getStrides();
    auto strideString = [&](int64_t axis) -> std::string {
      return expressionString(cast<gpu::PhysicalExprAttr>(strides[axis]));
    };

    SmallVector<std::string> baseOffsets;
    SmallVector<std::string> offsetLimits(view.getRank());
    for (auto [blockAxis, viewAxis] : llvm::enumerate(blockAxes))
      offsetLimits[viewAxis] = "(2147483648 - " +
          expressionString(cast<gpu::PhysicalExprAttr>(
                               fragment.getShape()[blockAxis])) + ")";
    std::string base = valueString(viewValue);
    for (int64_t viewAxis = 0;
         viewAxis < static_cast<int64_t>(view.getRank()); ++viewAxis) {
      std::string offset = "tl.cast(" + valueString(offsets[viewAxis]) +
                           ", tl.int64)";
      // Keep ordinary coordinates in the native i32 offset. Only rebase when
      // the end of the block would exceed that range, preserving a fixed view
      // shape for native loop-bound and memory-pipeline optimization.
      if (!offsetLimits[viewAxis].empty())
        offset = "(tl.maximum(" + offset + ", " + offsetLimits[viewAxis] +
                 ") - " + offsetLimits[viewAxis] + ")";
      baseOffsets.push_back(offset);
      base += " + " + offset + " * " + strideString(viewAxis);
    }

    SmallVector<std::string> shape;
    SmallVector<std::string> blockStrides;
    SmallVector<std::string> offsetExpressions;
    for (int64_t viewAxis : blockAxes) {
      shape.push_back(
          "tl.maximum((" +
          expressionString(cast<gpu::PhysicalExprAttr>(
                               view.getLayout().getExtents()[viewAxis])) +
          " - " + baseOffsets[viewAxis] + "), 0)");
      blockStrides.push_back(strideString(viewAxis));
      // A block has at most 2^20 lanes, so an offset below INT32_MIN is
      // entirely padding. Clamping it preserves that fact without wrapping.
      offsetExpressions.push_back(
          "tl.cast(tl.maximum(tl.minimum(tl.cast(" +
          valueString(offsets[viewAxis]) +
          ", tl.int64), " + offsetLimits[viewAxis] +
          "), -2147483648), tl.int32)");
    }
    return "tl.make_block_ptr(base=(" + base + ")" +
           ", shape=" + stringTuple(shape) +
           ", strides=" + stringTuple(blockStrides) +
           ", offsets=" + stringTuple(offsetExpressions) +
           ", block_shape=" + fragmentShape(fragment) +
           ", order=" + axisTuple(order) + ")";
  }

  std::string pointer(gpu::AccessOpInterface access) {
    Value resource = access.getAccessResource();
    auto coordinates = access.getAccessCoordinates();
    auto sourceAxes = access.getAccessSourceAxes();
    Type valueType = access.getAccessValueType();
    auto argument = dyn_cast<BlockArgument>(resource);
    if (!argument) {
      failed = true;
      return {};
    }
    auto view = cast<gpu::ViewType>(resource.getType());
    auto strides = view.getLayout().getStrides();
    auto fragment = dyn_cast<gpu::FragmentType>(valueType);
    SmallVector<Value> fragmentCoordinates;
    for (Value coordinate : coordinates)
      if (isa<gpu::FragmentType>(coordinate.getType()))
        fragmentCoordinates.push_back(coordinate);
    bool cartesian = fragment &&
        fragmentCoordinates.size() == fragment.getShape().size() &&
        llvm::all_of(fragmentCoordinates, [](Value coordinate) {
          return cast<gpu::FragmentType>(coordinate.getType()).getShape().size() == 1;
        });
    unsigned coordinateSlot = 0;
    std::string result = values.lookup(resource);
    for (auto [axis, coordinate] : llvm::enumerate(coordinates)) {
      std::string stride = expressionString(
          cast<gpu::PhysicalExprAttr>(strides[sourceAxes[axis]]));
      std::optional<unsigned> coordinateAxis;
      if (cartesian && isa<gpu::FragmentType>(coordinate.getType())) {
        auto source = cast<gpu::FragmentType>(coordinate.getType());
        auto sourceMap = cast<gpu::AxisMapAttr>(source.getAxisMaps()[0]);
        auto targetMap = cast<gpu::AxisMapAttr>(fragment.getAxisMaps()[coordinateSlot]);
        // A Cartesian coordinate slot already defines its result axis, even
        // when two slots carry the same logical source provenance.
        if (sourceMap.getSourceId() == targetMap.getSourceId() &&
            sourceMap.getSourceAxis() == targetMap.getSourceAxis() &&
            sourceMap.getDerived() == targetMap.getDerived() &&
            sourceMap.getDimensionId() == targetMap.getDimensionId() &&
            source.getShape()[0] == fragment.getShape()[coordinateSlot])
          coordinateAxis = coordinateSlot;
        ++coordinateSlot;
      }
      std::string coordinateExpression = fragment
          ? broadcastValue(coordinate, fragment, coordinateAxis)
          : valueString(coordinate);
      result += " + (" + coordinateExpression + ") * " + stride;
    }
    return "(" + result + ")";
  }

  std::string controlValueString(Value value) {
    std::string result = valueString(value);
    if (isa<IntegerType, IndexType, FloatType>(value.getType())) {
      if (value.getDefiningOp<arith::ConstantOp>())
        return "tl.full((), " + result + ", " + pythonType(value.getType()) + ")";
      return "tl.cast(" + result + ", " + pythonType(value.getType()) + ")";
    }
    return result;
  }

  std::string valueString(Value value) {
    auto found = values.find(value);
    if (found == values.end()) {
      failed = true;
      return "<missing>";
    }
    return found->second;
  }

  bool isConstexprExpression(gpu::PhysicalExprAttr expression) {
    if (expression.getKind() ==
        gpu::PhysicalExprKind::ScalarABI)
      return false;
    return llvm::all_of(expression.getOperands(), [&](Attribute operand) {
      return isConstexprExpression(cast<gpu::PhysicalExprAttr>(operand));
    });
  }

  void assign(Value value, const std::string &expression, bool compileTime = false) {
    std::string name = newName();
    values[value] = name;
    if (compileTime)
      constexprValues.insert(value);
    line(name + (compileTime ? ": tl.constexpr = " : " = ") + expression);
  }

  void assignResults(ResultRange results, const std::string &expression) {
    SmallVector<std::string> names;
    for (Value result : results) {
      names.push_back(newName());
      values[result] = names.back();
    }
    std::string statement;
    for (auto [index, name] : llvm::enumerate(names)) {
      if (index)
        statement += ", ";
      statement += name;
    }
    line(statement + " = " + expression);
  }

  std::string newName() {
    std::string name;
    do {
      name = "v" + std::to_string(counter++);
    } while (llvm::any_of(values, [&](const auto &entry) {
      return entry.second == name;
    }));
    return name;
  }

  void line(const std::string &text, unsigned explicitIndent = ~0U) {
    unsigned level = explicitIndent == ~0U ? indent : explicitIndent;
    output.indent(level * 4) << text << "\n";
  }

  FailureOr<llvm::json::Value> descriptorHostValue(Value value) {
    if (auto dimension = value.getDefiningOp<gpu::DimOp>()) {
      auto view = cast<gpu::ViewType>(dimension.getView().getType());
      return gpu::serializeExpression(cast<gpu::PhysicalExprAttr>(
          view.getLayout().getExtents()[dimension.getAxis()]));
    }
    if (auto physical = value.getDefiningOp<gpu::PhysicalExprOp>())
      return gpu::serializeExpression(physical.getExpression());
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
        return llvm::json::Value(llvm::json::Object{
            {"kind", "constant"}, {"value", integer.getInt()}});
    }
    if (auto binary = value.getDefiningOp<gpu::BinaryOp>()) {
      StringRef kind;
      switch (binary.getOperatorKind()) {
      case BinaryOperator::Add: kind = "add"; break;
      case BinaryOperator::Subtract: kind = "subtract"; break;
      case BinaryOperator::Multiply: kind = "multiply"; break;
      case BinaryOperator::FloorDivide: kind = "floor_div"; break;
      case BinaryOperator::Maximum: kind = "max"; break;
      case BinaryOperator::Minimum: kind = "min"; break;
      default: break;
      }
      if (!kind.empty()) {
        auto lhs = descriptorHostValue(binary.getLhs());
        auto rhs = descriptorHostValue(binary.getRhs());
        if (mlir::failed(lhs) || mlir::failed(rhs)) return failure();
        llvm::json::Array operands;
        operands.push_back(std::move(*lhs));
        operands.push_back(std::move(*rhs));
        return llvm::json::Value(llvm::json::Object{
            {"kind", kind}, {"operands", std::move(operands)}});
      }
    }
    if (auto expression = gpu::queryLaunchExpression(value))
      return gpu::serializeExpression(expression);
    return kernel.emitError("Triton descriptor has no host-evaluable value binding");
  }

  LogicalResult emitMetadata(std::string &metadata) {
    auto artifact = gpu::serializeInterface(
        kernel, "triton", [&](Value value) { return valueString(value); });
    if (mlir::failed(artifact)) return failure();
    llvm::json::Object details{{"kernel", "_intent_kernel"}};
    llvm::json::Array kernelParameters;
    for (StringAttr parameter : configuration.kernelParameters)
      kernelParameters.push_back(parameter.getValue());
    details["kernel_parameters"] = std::move(kernelParameters);
    details["native_options"] = llvm::json::Object{
        {"num_warps", configuration.warps.getValue()},
        {"num_stages", configuration.stages.getValue()},
        {"num_ctas", configuration.ctas.getValue()}};
    llvm::json::Array arguments, key;
    for (const ViewABI &view : views) arguments.push_back(view.name);
    for (const ScalarABI &scalar : scalars) arguments.push_back(scalar.name);
    details["metadata_argument"] = nullptr;
    if (!metadataArguments.empty()) {
      details["metadata_argument"] = metadataArgument;
      arguments.push_back(metadataArgument);
      key.push_back(metadataArgument);
    }
    details["overlap_argument"] = nullptr;
    if (!overlapFacts.empty()) {
      details["overlap_argument"] = overlapArgument;
      arguments.push_back(overlapArgument);
      key.push_back(overlapArgument);
    }
    details["descriptor_choice"] = nullptr;
    if (descriptorChoice) {
      details["descriptor_choice"] = llvm::json::Object{
          {"config", descriptorChoice.getConfigParameter().getName().getValue()},
          {"eligibility", descriptorChoice.getEligibilityArgument()}};
      arguments.push_back(descriptorChoice.getEligibilityArgument());
      key.push_back(descriptorChoice.getEligibilityArgument());
    }
    for (const std::string &name : coverageNames) key.push_back(name);
    llvm::json::Array encodedDescriptors;
    for (const DescriptorABI &descriptor : descriptors) {
      TensorDescriptorOp operation = descriptor.operation;
      llvm::json::Object encoded{
          {"name", descriptor.name},
          {"base", gpu::getArgumentReference(operation.getBase()).getId()},
          {"rank", cast<gpu::ViewType>(operation.getBase().getType()).getRank()},
          {"require_positive_shape", operation.getRequirePositiveShape()},
          {"require_positive_strides", operation.getRequirePositiveStrides()},
          {"alignment", operation.getAlignment()},
          {"maximum_shape_extent", operation.getMaximumShapeExtent()},
          {"padding", operation.getPadding()}};
      auto encodeValues = [&](StringRef field, ValueRange values) -> LogicalResult {
        llvm::json::Array expressions;
        for (Value value : values) {
          auto expression = descriptorHostValue(value);
          if (mlir::failed(expression)) return failure();
          expressions.push_back(std::move(*expression));
        }
        encoded[field] = std::move(expressions);
        return success();
      };
      if (mlir::failed(encodeValues("shape", operation.getShape())) ||
          mlir::failed(encodeValues("strides", operation.getStrides())) ||
          mlir::failed(encodeValues("block_shape", operation.getBlockShape())))
        return failure();
      llvm::json::Array aligned, unit;
      for (int64_t axis : operation.getAlignedStrideAxes()) aligned.push_back(axis);
      for (int64_t axis : operation.getUnitStrideAxes()) unit.push_back(axis);
      encoded["aligned_stride_axes"] = std::move(aligned);
      encoded["unit_stride_axes"] = std::move(unit);
      encodedDescriptors.push_back(std::move(encoded));
    }
    details["descriptors"] = std::move(encodedDescriptors);
    details["allocator"] = nullptr;
    if (descriptorAllocator)
      details["allocator"] = llvm::json::Object{
          {"size_argument", descriptorAllocator.getSizeArgument()},
          {"alignment_argument", descriptorAllocator.getAlignmentArgument()},
          {"stream_argument", descriptorAllocator.getStreamArgument()},
          {"lifetime", descriptorAllocator.getLifetime()},
          {"implementation", descriptorAllocator.getImplementation()}};
    details["kernel_arguments"] = std::move(arguments);
    details["autotune_key"] = std::move(key);
    (*artifact)["triton"] = std::move(details);
    llvm::raw_string_ostream(metadata) << llvm::json::Value(std::move(*artifact));
    return failed ? failure() : success();
  }

  func::FuncOp kernel;
  raw_ostream &output;
  llvm::DenseMap<Value, std::string> values;
  llvm::DenseSet<Value> constexprValues;
  SmallVector<ViewABI> views;
  SmallVector<ScalarABI> scalars;
  SmallVector<MetadataABI> metadataArguments;
  std::string metadataArgument = "_intent_metadata";
  SmallVector<gpu::ViewOverlapOp> overlapFacts;
  std::string overlapArgument = "_intent_overlaps";
  std::set<std::string> coverageNames;
  ConfigurationSchema configuration;
  llvm::DenseMap<Operation *, SmallVector<std::string>> helperNames;
  TensorDescriptorChoiceOp descriptorChoice;
  TensorDescriptorAllocatorOp descriptorAllocator;
  SmallVector<DescriptorABI> descriptors;
  unsigned indent = 0;
  unsigned counter = 0;
  unsigned helperCounter = 0;
  bool emittingHelper = false;
  bool failed = false;
};

} // namespace

LogicalResult serializeProgram(ModuleOp module, std::string &source,
                               std::string &metadata) {
  SmallVector<func::FuncOp> kernels;
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (function->hasAttr(gpu::kernelAttr))
      kernels.push_back(function);
  if (kernels.size() != 1)
    return module.emitError(
        "Triton serialization requires one provider-legalized kernel");
  func::FuncOp kernel = kernels.front();
  if (!kernel->hasAttr("intent_gpu.triton.legalized"))
    return kernel.emitError("Triton program was not provider-legalized");
  llvm::raw_string_ostream stream(source);
  Serializer serializer(kernel, stream);
  LogicalResult result = serializer.emit(metadata);
  stream.flush();
  return result;
}

} // namespace intent::triton
