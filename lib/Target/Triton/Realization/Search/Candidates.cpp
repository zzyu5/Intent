#include "Support/Model.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/Triton/IR/TritonOps.h"

using namespace mlir;

namespace intent::triton::realization {
namespace {

IntegerAttr i64(OpBuilder &builder, int64_t value) {
  return builder.getI64IntegerAttr(value);
}

StringAttr string(OpBuilder &builder, StringRef value) {
  return builder.getStringAttr(value);
}

} // namespace

void emitAutotuneSpace(ModuleOp module, func::FuncOp entry,
                       const PolicyDecision &policy, OpBuilder &builder) {
  builder.setInsertionPointToEnd(module.getBody());
  auto search = builder.create<intent::plan::SearchSpaceOp>(
      entry.getLoc(), FlatSymbolRefAttr::get(module.getContext(), entry.getName()),
      string(builder, "triton"));
  Block &body = search.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  SmallVector<Attribute> keys;
  for (const std::string &key : policy.autotuneKeys)
    keys.push_back(string(builder, key));
  builder.create<plan::AutotuneOp>(entry.getLoc(), builder.getArrayAttr(keys));
  if (policy.stateStream) {
    struct StreamConfig {
      int64_t query, stream, stages, warps;
    };
    constexpr StreamConfig configs[] = {
        {64, 32, 3, 4},  {64, 64, 3, 4},   {128, 32, 3, 4},
        {128, 64, 3, 4}, {128, 64, 2, 8},  {128, 128, 2, 8},
    };
    for (const StreamConfig &config : configs)
      builder.create<plan::ConfigOp>(
          entry.getLoc(),
          builder.getDictionaryAttr(
              {builder.getNamedAttr("BLOCK_SIZE_Q",
                                    i64(builder, config.query)),
               builder.getNamedAttr("BLOCK_SIZE_K",
                                    i64(builder, config.stream))}),
          i64(builder, config.stages), i64(builder, config.warps));
  } else {
    struct ContractConfig {
      int64_t m, n, k, group, stages, warps;
    };
    constexpr ContractConfig configs[] = {
        {128, 256, 64, 8, 3, 8}, {64, 256, 32, 8, 4, 4},
        {128, 128, 32, 8, 4, 4}, {128, 64, 32, 8, 4, 4},
        {64, 128, 32, 8, 4, 4},  {128, 32, 32, 8, 4, 4},
        {64, 32, 32, 8, 5, 2},   {32, 64, 32, 8, 5, 2},
        {128, 256, 128, 8, 3, 8}, {256, 128, 128, 8, 3, 8},
        {256, 64, 128, 8, 4, 4},  {64, 256, 128, 8, 4, 4},
        {128, 128, 128, 8, 4, 4}, {128, 64, 64, 8, 4, 4},
        {64, 128, 64, 8, 4, 4},   {128, 32, 64, 8, 4, 4},
    };
    for (const ContractConfig &config : configs)
      builder.create<plan::ConfigOp>(
          entry.getLoc(),
          builder.getDictionaryAttr(
              {builder.getNamedAttr("BLOCK_SIZE_M", i64(builder, config.m)),
               builder.getNamedAttr("BLOCK_SIZE_N", i64(builder, config.n)),
               builder.getNamedAttr("BLOCK_SIZE_K", i64(builder, config.k)),
               builder.getNamedAttr("GROUP_SIZE_M",
                                    i64(builder, config.group))}),
          i64(builder, config.stages), i64(builder, config.warps));
  }
  builder.create<intent::plan::YieldOp>(entry.getLoc());
}

} // namespace intent::triton::realization
