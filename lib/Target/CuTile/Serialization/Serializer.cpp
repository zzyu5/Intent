#include "Intent/Target/CuTile/Serialization/Serializer.h"

#include "Intent/Dialect/GPU/Analysis/PhysicalProgram.h"
#include "Intent/Dialect/GPU/IR/GPUAttrs.h"
#include "Intent/Dialect/GPU/IR/GPUOps.h"
#include "Intent/Dialect/GPU/IR/Program.h"
#include "Intent/Dialect/GPU/Serialization/Interface.h"
#include "Intent/Dialect/GPU/Serialization/Python.h"
#include "Intent/Target/CuTile/Analysis/Tuning.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <numeric>

using namespace mlir;

namespace intent::cutile {
namespace {

std::string pythonType(Type type) {
  static const gpu::PythonScalarSyntax syntax{
      "ct.", "bool_", "float64", "float8_e4m3fn", "float8_e5m2"};
  return gpu::pythonScalarType(type, syntax);
}

std::string expressionString(gpu::PhysicalExprAttr expression) {
  static const gpu::PythonExpressionSyntax syntax{
      "ct.cdiv", "min", "max", "", "_intent_next_power_of_2", false};
  return gpu::pythonExpression(expression, syntax, [](gpu::PhysicalExprAttr leaf) {
    return leaf.getSymbol().getValue().str();
  });
}

std::string fragmentShape(gpu::FragmentType fragment) {
  std::string result = "(";
  for (auto [index, extent] : llvm::enumerate(fragment.getShape())) {
    if (index)
      result += ", ";
    result += expressionString(cast<gpu::PhysicalExprAttr>(extent));
  }
  if (fragment.getShape().size() == 1)
    result += ",";
  return result + ")";
}

std::string literal(Attribute value) {
  if (auto expression = dyn_cast<gpu::PhysicalExprAttr>(value))
    return expressionString(expression);
  return gpu::pythonLiteral(value);
}

StringRef providerHint(gpu::ParameterOp parameter) {
  auto role = static_cast<gpu::ParameterRole>(parameter.getParameter().getRole());
  if (role == gpu::ParameterRole::ProviderOccupancy)
    return "occupancy";
  if (role == gpu::ParameterRole::ProviderCTAs)
    return "num_ctas";
  if (role == gpu::ParameterRole::ProviderWarps)
    return "num_worker_warps";
  return {};
}

class Serializer {
public:
  Serializer(func::FuncOp kernel, raw_ostream &output)
      : kernel(kernel), output(output) {}

  LogicalResult emit(std::string &metadata) {
    bindArguments();
    if (failed) return failure();
    emitPreamble();
    emitCollectiveHelpers();
    emitKernel();
    return failed ? failure() : emitMetadata(metadata);
  }

private:
  using ViewABI = gpu::ViewArgument;
  using ScalarABI = gpu::ScalarArgument;
  using MetadataABI = gpu::MetadataArgument;
  struct ArrayViewABI {
    ArrayViewOp operation;
    unsigned sourceView;
    std::string name;
    std::string eligible;
  };

  void bindArguments() {
    auto interface = gpu::readInterface(kernel);
    if (mlir::failed(interface)) {
      failed = true;
      return;
    }
    views = std::move(interface->views);
    scalars = std::move(interface->scalars);
    metadataArguments = std::move(interface->metadata);
    for (const ScalarABI &scalar : scalars)
      values[kernel.getArgument(scalar.abi)] = scalar.name;
    for (const MetadataABI &metadata : metadataArguments)
      values[kernel.getArgument(metadata.abi)] = metadata.name;
    for (const ViewABI &view : views)
      values[kernel.getArgument(view.abi)] = view.name;
    kernel.walk([&](gpu::ParameterOp parameter) {
      if (StringRef hint = providerHint(parameter); !hint.empty()) {
        if (!providerHintParameters.emplace(
                hint.str(), parameter.getParameter().getName().getValue().str()).second) {
          parameter.emitOpError("duplicates a cuTile compiler hint");
          failed = true;
          return;
        }
        return;
      }
    });
    llvm::StringSet<> occupied;
    for (const auto &entry : values)
      occupied.insert(entry.second);
    kernel.walk([&](gpu::ParameterOp parameter) {
      occupied.insert(parameter.getParameter().getName().getValue());
    });
    auto fresh = [&](StringRef stem) {
      unsigned suffix = 0;
      std::string name;
      do {
        name = (Twine(stem) + Twine(suffix++)).str();
      } while (!occupied.insert(name).second);
      return name;
    };
    kernel.walk([&](ArrayViewOp array) {
      unsigned argument = cast<BlockArgument>(array.getBase()).getArgNumber();
      for (auto [index, view] : llvm::enumerate(views)) {
        if (view.abi != argument)
          continue;
        ArrayViewABI binding{
            array, static_cast<unsigned>(index), fresh("_intent_array_view_"),
            fresh("_intent_array_valid_")};
        values[array.getResult()] = binding.name;
        values[array.getEligible()] = binding.eligible;
        arrayViews.push_back(std::move(binding));
      }
    });
    kernel.walk([&](gpu::ViewOverlapOp overlap) {
      values[overlap.getResult()] = fresh("_intent_overlap_");
      overlapFacts.push_back(overlap);
    });
  }

  void emitPreamble() {
    bool libraryMath = false;
    kernel.walk([&](gpu::UnaryOp unary) {
      libraryMath |= unary.getOperatorKind() == UnaryOperator::Asin ||
                     unary.getOperatorKind() == UnaryOperator::Erf ||
                     unary.getOperatorKind() == UnaryOperator::Erfc ||
                     unary.getOperatorKind() == UnaryOperator::I0 ||
                     unary.getOperatorKind() == UnaryOperator::Lgamma ||
                     unary.getOperatorKind() == UnaryOperator::Log1p;
    });
    kernel.walk([&](MMAOp mma) {
      libraryMath |= mma.getReductionChunk().has_value();
    });
    if (libraryMath)
      output << "from intent.runtime import cutile_math\n";
    output << "from typing import Annotated\n"
              "import cuda.tile as ct\n"
              "from intent.runtime.cutile import array_index_kernels\n\n"
              "ConstInt = ct.Constant[int]\n\n"
              "@ct.function(host=True)\n"
              "def _intent_next_power_of_2(value):\n"
              "    value = max(value, 1) - 1\n"
              "    value = value | (value >> 1)\n"
              "    value = value | (value >> 2)\n"
              "    value = value | (value >> 4)\n"
              "    value = value | (value >> 8)\n"
              "    value = value | (value >> 16)\n"
              "    value = value | (value >> 32)\n"
              "    return value + 1\n\n";
  }

  void emitCollectiveHelpers() {
    SmallVector<Operation *> collectives;
    kernel.walk([&](Operation *operation) {
      if (isa<ReduceOp, ScanOp>(operation) &&
          !operation->getRegion(0).empty())
        collectives.push_back(operation);
    });
    for (Operation *collective : collectives) {
      std::string helper = "_intent_combine_" +
                           std::to_string(collectiveHelpers.size());
      collectiveHelpers[collective] = helper;
      output << "@ct.function\ndef " << helper << "(";
      Block &block = collective->getRegion(0).front();
      for (auto [index, argument] : llvm::enumerate(block.getArguments())) {
        if (index)
          output << ", ";
        std::string name = "arg" + std::to_string(index);
        values[argument] = name;
        output << name;
      }
      output << "):\n";
      indent = 1;
      for (Operation &operation : block) {
        if (auto yield = dyn_cast<gpu::YieldOp>(operation)) {
          if (yield.getValues().size() == 1)
            line("return " + valueString(yield.getValues().front()));
          else
            line("return " + tuple(yield.getValues()));
          continue;
        }
        emitOperation(operation);
      }
      output << "\n";
      indent = 0;
    }
  }

  void emitKernel() {
    if (!kernel->hasAttr(arrayIndexTileBoundsAttr))
      output << "@ct.kernel\n";
    output << "def _intent_kernel(";
    bool first = true;
    auto argument = [&](StringRef text) {
      if (!first)
        output << ", ";
      first = false;
      output << text;
    };
    auto arrayArgument = [&](StringRef name, unsigned rank) {
      // Shapes and strides specialize the metadata ABI and launch cache. Preserve
      // the same facts in the provider array type used for access lowering.
      std::string dimensions = "(";
      for (unsigned axis = 0; axis < rank; ++axis)
        dimensions += std::to_string(axis) + ", ";
      dimensions += ")";
      argument(name.str() +
               ": Annotated[ct.Array, ct.ArrayAnnotation(index_dtype=ct.int64, static_shape_dims=" +
               dimensions + ", static_stride_dims=" + dimensions + ")]");
    };
    for (const ViewABI &view : views)
      arrayArgument(view.name, view.type.getLayout().getExtents().size());
    for (ArrayViewABI view : arrayViews) {
      arrayArgument(view.name, view.operation.getGroupEnds().size());
      argument(view.eligible + ": ct.Constant[bool]");
    }
    for (const ScalarABI &scalar : scalars) {
      auto integer = dyn_cast<IntegerType>(scalar.type);
      bool wide = scalar.type.isIndex() ||
                  (integer && integer.getWidth() == 64);
      argument(scalar.name + (wide ? ": ct.ScalarInt64" : ""));
    }
    for (const MetadataABI &metadata : metadataArguments)
      argument(metadata.name + ": ConstInt");
    for (gpu::ViewOverlapOp overlap : overlapFacts)
      argument(valueString(overlap.getResult()) + ": ct.Constant[bool]");
    kernel.walk([&](gpu::ParameterOp parameter) {
      if (!providerHint(parameter).empty())
        return;
      std::string name = parameter.getParameter().getName().getValue().str();
      values[parameter.getResult()] = name;
      argument(name + ": ConstInt");
    });
    output << "):\n";
    indent = 1;
    emitBlock(kernel.getBody().front(), false, {});
    output << "\n";
    if (kernel->hasAttr(arrayIndexTileBoundsAttr)) {
      output << "_intent_i32_kernel, _intent_kernel = array_index_kernels(_intent_kernel, (";
      for (const ViewABI &view : views) {
        llvm::json::OStream(output).value(view.name);
        output << ", ";
      }
      for (const ArrayViewABI &view : arrayViews) {
        llvm::json::OStream(output).value(view.name);
        output << ", ";
      }
      output << "))\n\n";
    }
  }

  void emitBlock(Block &block, bool isLoop,
                 ArrayRef<std::string> loopResults) {
    for (Operation &operation : block) {
      if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
        if (isLoop && !loopResults.empty()) {
          std::string names;
          for (auto [index, name] : llvm::enumerate(loopResults)) {
            if (index)
              names += ", ";
            names += name;
          }
          line(names + " = " +
               (yield.getOperands().size() == 1
                    ? valueString(yield.getOperands().front())
                    : tuple(yield.getOperands())));
        }
        continue;
      }
      if (isa<func::ReturnOp>(operation))
        continue;
      emitOperation(operation);
    }
  }

  void emitIfBranch(Block &block, ArrayRef<std::string> resultNames) {
    bool emitted = false;
    for (Operation &operation : block) {
      if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
        if (yield.getOperands().size() != resultNames.size()) {
          yield.emitOpError(
              "cuTile if yield/result arity changed after provider legalization");
          failed = true;
          continue;
        }
        if (resultNames.empty())
          continue;
        std::string names;
        for (auto [index, name] : llvm::enumerate(resultNames)) {
          if (index)
            names += ", ";
          names += name;
        }
        line(names + " = " +
             (yield.getOperands().size() == 1
                  ? valueString(yield.getOperands().front())
                  : tuple(yield.getOperands())));
        emitted = true;
        continue;
      }
      if (isa<func::ReturnOp>(operation))
        continue;
      emitOperation(operation);
      emitted = true;
    }
    if (!emitted)
      line("pass");
  }

  void emitOperation(Operation &operation) {
    if (isa<ArrayViewOp, gpu::ViewOverlapOp>(operation)) {
      return;
    } else if (auto assertion = dyn_cast<cf::AssertOp>(operation)) {
      std::string message;
      llvm::raw_string_ostream(message) << llvm::json::Value(assertion.getMsg());
      auto comparison = assertion.getArg().getDefiningOp<gpu::CompareOp>();
      auto count = comparison.getLhs().getDefiningOp<gpu::PhysicalExprOp>();
      auto limit = comparison.getRhs().getDefiningOp<arith::ConstantIndexOp>();
      line("ct.static_assert((" + expressionString(count.getExpression()) +
           " <= " + std::to_string(limit.value()) + "), " + message + ")");
    } else if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      std::string value = literal(constant.getValue());
      if (constant.getType().isInteger(64))
        value = "ct.astype(" + value + ", ct.int64)";
      values[constant.getResult()] = value;
    } else if (auto parameter = dyn_cast<gpu::ParameterOp>(operation)) {
      values[parameter.getResult()] =
          parameter.getParameter().getName().getValue().str();
    } else if (auto physical = dyn_cast<gpu::PhysicalExprOp>(operation)) {
      assign(physical.getResult(), expressionString(physical.getExpression()));
    } else if (auto program = dyn_cast<gpu::ProgramIdOp>(operation)) {
      assign(program.getResult(), "ct.astype(ct.bid(" +
                                      std::to_string(program.getAxis()) + "), " +
                                      pythonType(program.getResult().getType()) + ")");
    } else if (auto coordinate = dyn_cast<gpu::WorksetCoordinateOp>(operation)) {
      values[coordinate.getResult()] = valueString(coordinate.getCoordinate());
    } else if (auto dim = dyn_cast<gpu::DimOp>(operation)) {
      auto extent = cast<gpu::PhysicalExprAttr>(
          dim.getView().getType().getLayout().getExtents()[dim.getAxis()]);
      assign(dim.getResult(), expressionString(extent));
    } else if (auto range = dyn_cast<gpu::RangeOp>(operation)) {
      assign(range.getResult(), "(" + valueString(range.getStart()) + ", " +
                                    valueString(range.getStop()) + ", " +
                                    valueString(range.getStep()) + ")");
    } else if (auto bound = dyn_cast<gpu::RangeBoundOp>(operation)) {
      assign(bound.getResult(), valueString(bound.getRange()) + "[" +
                                    std::to_string(bound.getBound()) + "]");
    } else if (auto mapping = dyn_cast<gpu::DelinearizeOp>(operation)) {
      std::string remaining = valueString(mapping.getLinear());
      SmallVector<std::string> coordinates(mapping.getNumResults());
      for (int64_t axis = mapping.getNumResults() - 1; axis >= 0; --axis) {
        std::string extent = valueString(mapping.getExtents()[axis]);
        coordinates[axis] = "(" + remaining + " % " + extent + ")";
        remaining = "(" + remaining + " // " + extent + ")";
      }
      for (auto [coordinate, text] : llvm::zip(mapping.getCoordinates(), coordinates))
        assign(coordinate, text);
    } else if (auto binary = dyn_cast<gpu::BinaryOp>(operation)) {
      assign(binary.getResult(), binaryExpression(binary));
    } else if (auto unary = dyn_cast<gpu::UnaryOp>(operation)) {
      assign(unary.getResult(), unaryExpression(unary));
    } else if (auto compare = dyn_cast<gpu::CompareOp>(operation)) {
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
                                      valueString(compare.getRhs()) + ")");
    } else if (auto range = dyn_cast<gpu::MakeRangeOp>(operation)) {
      assign(range.getResult(), "(" + valueString(range.getStart()) +
                                    " + ct.arange(" + valueString(range.getExtent()) +
                                    ", dtype=" +
                                    pythonType(range.getResult().getType().getElementType()) + ") * " +
                                    valueString(range.getStep()) + ")");
    } else if (auto splat = dyn_cast<gpu::SplatOp>(operation)) {
      auto type = splat.getResult().getType();
      assign(splat.getResult(), "ct.full(" + fragmentShape(type) + ", " +
                                    valueString(splat.getValue()) + ", dtype=" +
                                    pythonType(type.getElementType()) + ")");
    } else if (auto broadcast = dyn_cast<gpu::BroadcastOp>(operation)) {
      assign(broadcast.getResult(), broadcastValue(broadcast.getValue(),
                                                   broadcast.getResult().getType()));
    } else if (auto cast = dyn_cast<gpu::CastOp>(operation)) {
      assign(cast.getResult(), "ct.astype(" + valueString(cast.getValue()) +
                                   ", " + pythonType(elementType(cast.getResult().getType())) +
                                   ")");
    } else if (auto bitcast = dyn_cast<gpu::BitcastOp>(operation)) {
      assign(bitcast.getResult(), "ct.bitcast(" + valueString(bitcast.getValue()) +
                                      ", " +
                                      pythonType(elementType(bitcast.getResult().getType())) +
                                      ")");
    } else if (auto reshape = dyn_cast<gpu::ReshapeOp>(operation)) {
      assign(reshape.getResult(), "ct.reshape(" + valueString(reshape.getValue()) +
                                      ", " +
                                      fragmentShape(mlir::cast<gpu::FragmentType>(
                                          reshape.getResult().getType())) +
                                      ")");
    } else if (auto transpose = dyn_cast<gpu::TransposeOp>(operation)) {
      std::string permutation = "(";
      for (auto [index, axis] : llvm::enumerate(transpose.getPermutation())) {
        if (index)
          permutation += ", ";
        permutation += std::to_string(axis);
      }
      if (transpose.getPermutation().size() == 1)
        permutation += ",";
      permutation += ")";
      assign(transpose.getResult(), "ct.permute(" +
                                        valueString(transpose.getValue()) + ", " +
                                        permutation + ")");
    } else if (auto join = dyn_cast<gpu::JoinOp>(operation)) {
      auto input = join.getLhs().getType();
      std::string expanded = "(";
      for (Attribute extent : input.getShape())
        expanded += expressionString(
                        mlir::cast<gpu::PhysicalExprAttr>(extent)) +
                    ", ";
      expanded += "1)";
      assign(join.getResult(),
             "ct.cat((ct.reshape(" + valueString(join.getLhs()) + ", " +
                 expanded + "), ct.reshape(" + valueString(join.getRhs()) +
                 ", " + expanded + ")), axis=" +
                 std::to_string(join.getAxis()) + ")");
    } else if (auto record = dyn_cast<gpu::MakeRecordOp>(operation)) {
      values[record.getResult()] = tuple(record.getFields());
    } else if (auto extract = dyn_cast<gpu::ExtractOp>(operation)) {
      assign(extract.getResult(), valueString(extract.getRecord()) + "[" +
                                      std::to_string(extract.getField()) + "]");
    } else if (auto select = dyn_cast<gpu::SelectOp>(operation)) {
      assign(select.getResult(), "ct.where(" + valueString(select.getCondition()) +
                                      ", " + valueString(select.getTrueValue()) +
                                      ", " + valueString(select.getFalseValue()) + ")");
    } else if (auto load = dyn_cast<TileLoadOp>(operation)) {
      std::string padding = "ct.PaddingMode.ZERO";
      if (Value fullTiles = load.getFullTiles())
        padding = "(ct.PaddingMode.UNDETERMINED if " + valueString(fullTiles) +
                  " else ct.PaddingMode.ZERO)";
      std::string call = "ct.load(" + valueString(load.getResource()) +
                         ", index=" + tuple(load.getTileIndices()) +
                         ", shape=" + fragmentShape(load.getResult().getType()) +
                         ", padding_mode=" + padding + ", allow_tma=" +
                         valueString(load.getAllowTma());
      if (auto latency = load.getLatencyPolicy())
        call += ", latency=(None if " + valueString(latency) + " == " +
                std::to_string(inferredLoadPolicy) + " else " +
                valueString(latency) + ")";
      call += ")";
      assign(load.getResult(), call);
    } else if (auto load = dyn_cast<ScalarLoadOp>(operation)) {
      std::string call = "ct.gather(" + valueString(load.getResource()) + ", " +
                         tuple(load.getIndices());
      if (load.getValid())
        call += ", mask=" + valueString(load.getValid()) +
                ", padding_value=" + valueString(load.getFill());
      call += load.getInBounds() ? ", check_bounds=False)"
                                 : ", check_bounds=True)";
      assign(load.getResult(), call);
    } else if (auto gather = dyn_cast<GatherLoadOp>(operation)) {
      std::string call = "ct.gather(" + valueString(gather.getResource()) +
                         ", " + tuple(gather.getCoordinates());
      if (gather.getValid())
        call += ", mask=" + valueString(gather.getValid());
      if (gather.getFill())
        call += ", padding_value=" + valueString(gather.getFill());
      if (auto latency = gather.getLatencyPolicy())
        call += ", latency=(None if " + valueString(latency) + " == " +
                std::to_string(inferredLoadPolicy) + " else " +
                valueString(latency) + ")";
      call += gather.getInBounds() ? ", check_bounds=False)"
                                   : ", check_bounds=True)";
      assign(gather.getResult(), call);
    } else if (auto mma = dyn_cast<MMAOp>(operation)) {
      std::string call = mma.getReductionChunk() ? "cutile_math.mma_chunks(" : "ct.mma(";
      call += valueString(mma.getLhs()) + ", " + valueString(mma.getRhs()) +
              ", " + valueString(mma.getAccumulator());
      if (auto chunk = mma.getReductionChunk())
        call += ", " + std::to_string(*chunk);
      assign(mma.getResult(), call + ")");
    } else if (auto mma = dyn_cast<ScaledMMAOp>(operation)) {
      auto lhs = mma.getLhs().getType();
      auto rhs = mma.getRhs().getType();
      std::string lhsK =
          "(" + expressionString(
                     mlir::cast<gpu::PhysicalExprAttr>(lhs.getShape()[1])) +
          " * " + expressionString(
                        mlir::cast<gpu::PhysicalExprAttr>(lhs.getShape()[2])) +
          ")";
      std::string rhsK =
          "(" + expressionString(
                     mlir::cast<gpu::PhysicalExprAttr>(rhs.getShape()[0])) +
          " * " + expressionString(
                        mlir::cast<gpu::PhysicalExprAttr>(rhs.getShape()[1])) +
          ")";
      std::string lhsShape =
          "(" + expressionString(
                     mlir::cast<gpu::PhysicalExprAttr>(lhs.getShape()[0])) +
          ", " + lhsK + ")";
      std::string rhsShape =
          "(" + rhsK + ", " +
          expressionString(
              mlir::cast<gpu::PhysicalExprAttr>(rhs.getShape()[2])) +
          ")";
      assign(mma.getResult(),
             "ct.mma_scaled(ct.reshape(" + valueString(mma.getLhs()) + ", " +
                 lhsShape + "), ct.bitcast(" + valueString(mma.getLhsScale()) +
                 ", ct.float8_e8m0fnu)" +
                 ", ct.reshape(" + valueString(mma.getRhs()) + ", " + rhsShape +
                 "), ct.bitcast(" + valueString(mma.getRhsScale()) +
                 ", ct.float8_e8m0fnu), " +
                 valueString(mma.getAccumulator()) + ")");
    } else if (isa<ReduceOp, ScanOp>(operation) &&
               !operation.getRegion(0).empty()) {
      auto reduce = dyn_cast<ReduceOp>(operation);
      auto scan = dyn_cast<ScanOp>(operation);
      unsigned count = reduce ? reduce.getSourceCount() : scan.getSourceCount();
      ValueRange sources = operation.getOperands().take_front(count);
      ValueRange identities = operation.getOperands().slice(count, count);
      std::string source = count == 1 ? valueString(sources.front())
                                      : tuple(sources);
      std::string identity;
      if (count == 1) {
        identity = literal(getCompileTimeScalar(identities.front()));
      } else {
        identity = "(";
        for (auto [index, value] : llvm::enumerate(identities)) {
          if (index)
            identity += ", ";
          identity += literal(getCompileTimeScalar(value));
        }
        identity += ")";
      }
      std::string call = std::string(reduce ? "ct.reduce(" : "ct.scan(") +
                         source + ", axis=" +
                         std::to_string(reduce ? reduce.getAxis()
                                               : scan.getAxis()) +
                         ", func=" +
                         collectiveHelpers.lookup(&operation) +
                         ", identity=" + identity;
      if (scan)
        call += std::string(", reverse=") + (scan.getReverse() ? "True" : "False");
      call += ")";
      if (count == 1) {
        assign(operation.getResult(0), call);
      } else {
        std::string resultNames;
        for (auto [index, result] : llvm::enumerate(operation.getResults())) {
          if (index)
            resultNames += ", ";
          std::string name = newName();
          values[result] = name;
          resultNames += name;
        }
        line(resultNames + " = " + call);
      }
    } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
      auto nativeReduction = [](BinaryOperator kind) -> StringRef {
        switch (kind) {
        case BinaryOperator::Add: return "ct.sum";
        case BinaryOperator::MaximumNum:
        case BinaryOperator::LogicalOr: return "ct.max";
        case BinaryOperator::MinimumNum:
        case BinaryOperator::LogicalAnd: return "ct.min";
        default: llvm_unreachable("unverified cuTile native reduction kind");
        }
      };
      std::string expression =
          nativeReduction(*reduce.getKind()).str() + "(" +
          valueString(reduce.getInputs().front()) + ", axis=" +
          std::to_string(reduce.getAxis()) + ")";
      if (elementType(reduce.getResult(0).getType()).isInteger(1))
        expression = "ct.astype(" + expression + ", ct.bool_)";
      assign(reduce.getResult(0), expression);
    } else if (auto scan = dyn_cast<ScanOp>(operation)) {
      assign(scan.getResult(0), "ct.cumsum(" + valueString(scan.getInputs().front()) +
                                   ", axis=" + std::to_string(scan.getAxis()) +
                                   ", reverse=" +
                                   (scan.getReverse() ? "True" : "False") + ")");
    } else if (auto atomic = dyn_cast<AtomicRMWOp>(operation)) {
      auto atomicOperation = [](AtomicRMWKind kind) -> StringRef {
        switch (kind) {
        case AtomicRMWKind::Exchange: return "xchg";
        case AtomicRMWKind::Add: return "add";
        case AtomicRMWKind::Maximum: return "max";
        case AtomicRMWKind::Minimum: return "min";
        case AtomicRMWKind::BitwiseAnd: return "and";
        case AtomicRMWKind::BitwiseOr: return "or";
        case AtomicRMWKind::BitwiseXor: return "xor";
        }
        llvm_unreachable("unhandled atomic RMW kind");
      };
      auto atomicOrder = [](AtomicOrdering ordering) -> StringRef {
        switch (ordering) {
        case AtomicOrdering::Relaxed: return "RELAXED";
        case AtomicOrdering::Acquire: return "ACQUIRE";
        case AtomicOrdering::Release: return "RELEASE";
        case AtomicOrdering::AcquireRelease: return "ACQ_REL";
        }
        llvm_unreachable("unhandled atomic ordering");
      };
      auto atomicScope = [](gpu::AtomicSharingDomain sharing) -> StringRef {
        switch (sharing) {
        case gpu::AtomicSharingDomain::ProgramInstance: return "BLOCK";
        case gpu::AtomicSharingDomain::KernelInvocation: return "DEVICE";
        }
        llvm_unreachable("unhandled atomic sharing domain");
      };
      assign(atomic.getResult(),
             "ct.atomic_" + atomicOperation(atomic.getKind()).str() + "(" +
                 valueString(atomic.getResource()) + ", " +
                 tuple(atomic.getCoordinates()) + ", " +
                 valueString(atomic.getValue()) +
                 ", check_bounds=True, memory_order=ct.MemoryOrder." +
                 atomicOrder(atomic.getOrdering()).str() +
                 ", memory_scope=ct.MemoryScope." +
                 atomicScope(atomic.getSharing()).str() + ")");
    } else if (auto extract = dyn_cast<ExtractOp>(operation)) {
      std::string shape = "(";
      for (Attribute extent : extract.getExtractionShape())
        shape += expressionString(mlir::cast<gpu::PhysicalExprAttr>(extent)) + ", ";
      shape += ")";
      std::string result = "ct.extract(" + valueString(extract.getSource()) +
                           ", index=" + tuple(extract.getCoordinates()) +
                           ", shape=" + shape + ")";
      if (auto fragment = dyn_cast<gpu::FragmentType>(extract.getResult().getType()))
        result += ".reshape(" + fragmentShape(fragment) + ")";
      else
        result += ".item()";
      assign(extract.getResult(), result);
    } else if (auto atomic = dyn_cast<TileAtomicAddOp>(operation)) {
      line(valueString(atomic.getResource()) + ".tiled_view(" +
           fragmentShape(atomic.getValue().getType()) +
           ").atomic_store_add(" + tuple(atomic.getTileIndices()) + ", " +
           valueString(atomic.getValue()) + ")");
    } else if (auto store = dyn_cast<TileStoreOp>(operation)) {
      line("ct.store(" + valueString(store.getResource()) + ", index=" +
           tuple(store.getTileIndices()) + ", tile=" +
           valueString(store.getValue()) + ", allow_tma=" +
           valueString(store.getAllowTma()) + ")");
    } else if (auto store = dyn_cast<ScalarStoreOp>(operation)) {
      std::string call = "ct.scatter(" + valueString(store.getResource()) +
                         ", " + tuple(store.getIndices()) + ", " +
                         valueString(store.getValue());
      if (store.getValid())
        call += ", mask=" + valueString(store.getValid());
      line(call + (store.getInBounds() ? ", check_bounds=False)"
                                       : ", check_bounds=True)"));
    } else if (auto scatter = dyn_cast<ScatterStoreOp>(operation)) {
      std::string call = "ct.scatter(" + valueString(scatter.getResource()) +
                         ", " + tuple(scatter.getCoordinates()) +
                         ", " +
                         valueString(scatter.getValue());
      if (scatter.getValid())
        call += ", mask=" + valueString(scatter.getValid());
      line(call + (scatter.getInBounds() ? ", check_bounds=False)"
                                         : ", check_bounds=True)"));
    } else if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      SmallVector<std::string> results;
      for (auto [result, initial] : llvm::zip(loop.getResults(), loop.getInitArgs())) {
        std::string name = newName();
        values[result] = name;
        std::string initialValue = valueString(initial);
        if (initial.getType().isIntOrIndex())
          initialValue = "ct.astype(" + initialValue + ", " +
                         pythonType(initial.getType()) + ")";
        line(name + " = " + initialValue);
        results.push_back(name);
      }
      std::string induction = "iv" + std::to_string(counter++);
      values[loop.getInductionVar()] = induction;
      for (auto [argument, name] : llvm::zip(loop.getRegionIterArgs(), results))
        values[argument] = name;
      std::string indexType = pythonType(loop.getInductionVar().getType());
      line("for " + induction + " in range(ct.astype(" +
           valueString(loop.getLowerBound()) + ", " + indexType + "), ct.astype(" +
           valueString(loop.getUpperBound()) + ", " + indexType + "), ct.astype(" +
           valueString(loop.getStep()) + ", " + indexType + ")):");
      ++indent;
      emitBlock(*loop.getBody(), true, results);
      --indent;
    } else if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      SmallVector<std::string> carries;
      for (auto [result, initial] : llvm::zip(loop.getResults(), loop.getInits())) {
        std::string name = newName();
        values[result] = name;
        std::string initialValue = valueString(initial);
        if (initial.getType().isIntOrIndex())
          initialValue = "ct.astype(" + initialValue + ", " +
                         pythonType(initial.getType()) + ")";
        line(name + " = " + initialValue);
        carries.push_back(name);
      }
      Block &before = loop.getBefore().front();
      for (auto [argument, name] : llvm::zip(before.getArguments(), carries))
        values[argument] = name;
      line("while True:");
      ++indent;
      for (Operation &nested : before.without_terminator())
        emitOperation(nested);
      auto condition = mlir::cast<scf::ConditionOp>(before.getTerminator());
      line("if not " + valueString(condition.getCondition()) + ":");
      ++indent;
      line("break");
      --indent;
      Block &after = loop.getAfter().front();
      for (auto [argument, forwarded] : llvm::zip(after.getArguments(), condition.getArgs()))
        values[argument] = valueString(forwarded);
      emitBlock(after, true, carries);
      --indent;
    } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
      SmallVector<std::string> results;
      for (Value result : branch.getResults()) {
        std::string name = newName();
        values[result] = name;
        results.push_back(name);
      }
      line("if " + valueString(branch.getCondition()) + ":");
      ++indent;
      emitIfBranch(branch.getThenRegion().front(), results);
      --indent;
      if (!branch.getElseRegion().empty()) {
        line("else:");
        ++indent;
        emitIfBranch(branch.getElseRegion().front(), results);
        --indent;
      }
    } else {
      operation.emitOpError("has no terminal cuTile spelling");
      failed = true;
    }
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
    auto nativeMinMax = [&](StringRef spelling) {
      return spelling.str() + "(" + valueString(binary.getLhs()) + ", " +
             valueString(binary.getRhs()) + ")";
    };
    auto propagatingMinMax = [&](StringRef spelling) {
      std::string lhs = valueString(binary.getLhs());
      std::string rhs = valueString(binary.getRhs());
      std::string native = spelling.str() + "(" + lhs + ", " + rhs + ")";
      return "ct.where(ct.isnan(" + lhs + "), " + lhs +
             ", ct.where(ct.isnan(" + rhs + "), " + rhs + ", " + native + "))";
    };
    switch (binary.getOperatorKind()) {
    case BinaryOperator::Add: return infix("+");
    case BinaryOperator::Subtract: return infix("-");
    case BinaryOperator::Multiply: return infix("*");
    case BinaryOperator::TrueDivide:
      if (binary.getApproximate())
        return "ct.truediv(" + valueString(binary.getLhs()) + ", " +
               valueString(binary.getRhs()) +
               ", rounding_mode=ct.RoundingMode.APPROX, flush_to_zero=" +
               (binary.getFlushToZero() ? "True" : "False") + ")";
      return infix("/");
    case BinaryOperator::FloorDivide: return infix("//");
    case BinaryOperator::Remainder: return infix("%");
    case BinaryOperator::Power: return infix("**");
    case BinaryOperator::MaximumNum: return nativeMinMax("ct.maximum");
    case BinaryOperator::MinimumNum: return nativeMinMax("ct.minimum");
    case BinaryOperator::Maximum:
      return elementType(binary.getResult().getType()).isIntOrIndex()
                 ? nativeMinMax("ct.maximum")
                 : propagatingMinMax("ct.maximum");
    case BinaryOperator::Minimum:
      return elementType(binary.getResult().getType()).isIntOrIndex()
                 ? nativeMinMax("ct.minimum")
                 : propagatingMinMax("ct.minimum");
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
    auto call = [&](StringRef function) {
      return function.str() + "(" + input + ")";
    };
    switch (unary.getOperatorKind()) {
    case UnaryOperator::Negate: return "(-" + input + ")";
    case UnaryOperator::Not: return "(~" + input + ")";
    case UnaryOperator::Exp: return call("ct.exp");
    case UnaryOperator::Exp2:
      if (unary.getApproximate())
        return "ct.exp2(" + input + ", flush_to_zero=" +
               (unary.getFlushToZero() ? "True" : "False") + ")";
      return call("ct.exp2");
    case UnaryOperator::Log: return call("ct.log");
    case UnaryOperator::Sin: return call("ct.sin");
    case UnaryOperator::Cos: return call("ct.cos");
    case UnaryOperator::Floor: return call("ct.floor");
    case UnaryOperator::Rsqrt: return call("ct.rsqrt");
    case UnaryOperator::Sigmoid:
      return "ct.astype(1.0 / (1.0 + ct.exp(-ct.astype(" + input +
             (elementType(unary.getInput().getType()).isF64()
                  ? ", ct.float64))), " : ", ct.float32))), ") +
             pythonType(elementType(unary.getResult().getType())) + ")";
    case UnaryOperator::Tanh:
      return unary.getApproximate()
                 ? "ct.tanh(" + input + ", rounding_mode=ct.RoundingMode.APPROX)"
                 : call("ct.tanh");
    case UnaryOperator::Abs: return call("ct.abs");
    case UnaryOperator::Sqrt: return call("ct.sqrt");
    case UnaryOperator::Erf: return call("cutile_math.erf");
    case UnaryOperator::Log1p: return call("cutile_math.log1p");
    case UnaryOperator::Lgamma: return call("cutile_math.lgamma");
    case UnaryOperator::Erfc: return call("cutile_math.erfc");
    case UnaryOperator::I0: return call("cutile_math.i0");
    case UnaryOperator::Asin: return call("cutile_math.asin");
    }
    llvm_unreachable("unhandled Intent unary operator");
  }

  std::string broadcastValue(Value value, gpu::FragmentType target) {
    auto source = dyn_cast<gpu::FragmentType>(value.getType());
    if (!source)
      return "ct.full(" + fragmentShape(target) + ", " + valueString(value) +
             ", dtype=" + pythonType(target.getElementType()) + ")";
    if (source == target)
      return valueString(value);
    gpu::BroadcastProjection projection =
        gpu::queryBroadcastProjection(source, target);
    if (!projection.isExact()) {
      kernel.emitError("cuTile broadcast lost its shared axis projection");
      failed = true;
      return "<invalid-cutile-broadcast>";
    }
    if (source.getShape().size() == target.getShape().size())
      return "ct.broadcast_to(" + valueString(value) + ", " +
             fragmentShape(target) + ")";
    SmallVector<int64_t> targetForSource(source.getShape().size(), -1);
    for (auto [targetIndex, sourceIndex] :
         llvm::enumerate(projection.targetToSource))
      if (sourceIndex)
        targetForSource[*sourceIndex] = targetIndex;

    SmallVector<unsigned> sourceOrder(source.getShape().size());
    std::iota(sourceOrder.begin(), sourceOrder.end(), 0);
    llvm::sort(sourceOrder, [&](unsigned lhs, unsigned rhs) {
      return targetForSource[lhs] < targetForSource[rhs];
    });
    std::string input = valueString(value);
    bool permuted = llvm::any_of(
        llvm::enumerate(sourceOrder),
        [](auto item) { return item.index() != item.value(); });
    if (permuted) {
      std::string permutation = "(";
      for (unsigned axis : sourceOrder)
        permutation += std::to_string(axis) + ", ";
      permutation += ")";
      input = "ct.permute(" + input + ", " + permutation + ")";
    }

    std::string reshape = "(";
    unsigned orderedSource = 0;
    for (unsigned targetAxis = 0; targetAxis < target.getShape().size();
         ++targetAxis) {
      if (orderedSource < sourceOrder.size() &&
          targetForSource[sourceOrder[orderedSource]] ==
              static_cast<int64_t>(targetAxis)) {
        reshape += expressionString(
                       cast<gpu::PhysicalExprAttr>(
                           source.getShape()[sourceOrder[orderedSource]])) +
                   ", ";
        ++orderedSource;
      } else {
        reshape += "1, ";
      }
    }
    reshape += ")";
    return "ct.broadcast_to(ct.reshape(" + input + ", " + reshape + "), " +
           fragmentShape(target) + ")";
  }

  std::string tuple(ValueRange valuesRange) {
    std::string result = "(";
    for (auto [index, value] : llvm::enumerate(valuesRange)) {
      if (index)
        result += ", ";
      result += valueString(value);
    }
    if (valuesRange.size() == 1)
      result += ",";
    return result + ")";
  }

  std::string valueString(Value value) {
    auto found = values.find(value);
    if (found == values.end()) {
      if (Operation *producer = value.getDefiningOp())
        producer->emitOpError(
            "cuTile terminal translation encountered an unmapped SSA value");
      else
        kernel.emitError(
            "cuTile terminal translation encountered an unmapped block argument");
      failed = true;
      return "<missing>";
    }
    return found->second;
  }

  void assign(Value value, const std::string &expression) {
    std::string name = newName();
    values[value] = name;
    line(name + " = " + expression);
  }

  std::string newName() { return "v" + std::to_string(counter++); }

  void line(const std::string &text, unsigned explicitIndent = ~0U) {
    unsigned level = explicitIndent == ~0U ? indent : explicitIndent;
    output.indent(level * 4) << text << "\n";
  }

  LogicalResult emitMetadata(std::string &metadata) {
    auto artifact = gpu::serializeInterface(kernel, "cutile", [&](Value value) {
      return valueString(value);
    });
    if (mlir::failed(artifact)) return failure();
    llvm::json::Object details{{"kernel", "_intent_kernel"}};
    llvm::json::Array arguments, arrays;
    for (const ViewABI &view : views) arguments.push_back(view.name);
    for (const ArrayViewABI &view : arrayViews) {
      llvm::json::Array groups;
      ArrayViewOp operation = view.operation;
      for (int64_t end : operation.getGroupEnds()) groups.push_back(end);
      arrays.push_back(llvm::json::Object{
          {"name", view.name}, {"eligible", view.eligible},
          {"base", views[view.sourceView].name}, {"group_ends", std::move(groups)}});
      arguments.push_back(view.name);
      arguments.push_back(view.eligible);
    }
    for (const ScalarABI &scalar : scalars) arguments.push_back(scalar.name);
    for (const MetadataABI &argument : metadataArguments) arguments.push_back(argument.name);
    for (gpu::ViewOverlapOp overlap : overlapFacts)
      arguments.push_back(valueString(overlap.getResult()));
    kernel.walk([&](gpu::ParameterOp parameter) {
      if (providerHint(parameter).empty())
        arguments.push_back(parameter.getParameter().getName().getValue());
    });
    details["kernel_arguments"] = std::move(arguments);
    details["array_views"] = std::move(arrays);
    details["index_tile_bounds"] = nullptr;
    details["narrow_kernel"] = nullptr;
    if (auto bounds = kernel->getAttrOfType<ArrayAttr>(arrayIndexTileBoundsAttr)) {
      llvm::json::Array encoded;
      for (const ViewABI &view : views) {
        llvm::json::Array axes;
        for (Attribute bound : cast<ArrayAttr>(bounds[view.abi]))
          axes.push_back(gpu::serializeExpression(cast<gpu::PhysicalExprAttr>(bound)));
        encoded.push_back(std::move(axes));
      }
      details["index_tile_bounds"] = std::move(encoded);
      details["narrow_kernel"] = "_intent_i32_kernel";
    }
    llvm::json::Object hints;
    for (const auto &[hint, parameter] : providerHintParameters)
      hints[hint] = parameter;
    details["compiler_hints"] = std::move(hints);
    details["inferred_worker_warps"] = inferredWorkerWarps;
    llvm::json::Array scalarKeys;
    llvm::SmallBitVector keyArguments = getTuningKeyScalarArguments(kernel);
    for (const ScalarABI &scalar : scalars)
      if (keyArguments.test(scalar.abi)) scalarKeys.push_back(scalar.name);
    details["tuning_key_scalars"] = std::move(scalarKeys);
    (*artifact)["cutile"] = std::move(details);
    llvm::raw_string_ostream(metadata) << llvm::json::Value(std::move(*artifact));
    return failed ? failure() : success();
  }

  func::FuncOp kernel;
  raw_ostream &output;
  llvm::DenseMap<Value, std::string> values;
  llvm::DenseMap<Operation *, std::string> collectiveHelpers;
  SmallVector<ViewABI> views;
  SmallVector<ArrayViewABI> arrayViews;
  SmallVector<ScalarABI> scalars;
  SmallVector<MetadataABI> metadataArguments;
  SmallVector<gpu::ViewOverlapOp> overlapFacts;
  std::map<std::string, std::string> providerHintParameters;
  unsigned indent = 0;
  unsigned counter = 0;
  bool failed = false;
};

} // namespace

LogicalResult serializeProgram(ModuleOp module, std::string &source,
                               std::string &metadata) {
  FailureOr<func::FuncOp> kernel = gpu::getPhysicalKernel(module);
  if (failed(kernel))
    return failure();
  if (!(*kernel)->hasAttr("intent_cutile.legalized"))
    return (*kernel).emitError("cuTile program was not provider-legalized");
  llvm::raw_string_ostream stream(source);
  Serializer serializer(*kernel, stream);
  LogicalResult result = serializer.emit(metadata);
  stream.flush();
  return result;
}

} // namespace intent::cutile
