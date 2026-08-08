#include "Support/Model.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/TileLang/IR/TileLangOps.h"

using namespace mlir;

namespace intent::tilelang::realization {
namespace {

IntegerAttr i64(OpBuilder &builder, int64_t value) {
  return builder.getI64IntegerAttr(value);
}

StringAttr string(OpBuilder &builder, StringRef value) {
  return builder.getStringAttr(value);
}

struct ContractConfig {
  int64_t m, n, k, stages, threads;
};

struct StreamConfig {
  int64_t program, stream, stages, threads;
};

} // namespace

LogicalResult emitAutotuneSpace(ModuleOp module, func::FuncOp entry,
                                const TargetOptions &target,
                                const PolicyDecision &policy,
                                OpBuilder &builder) {
  builder.setInsertionPointToEnd(module.getBody());
  auto search = builder.create<intent::plan::SearchSpaceOp>(
      entry.getLoc(), FlatSymbolRefAttr::get(module.getContext(), entry.getName()),
      string(builder, "tilelang"));
  Block &body = search.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  SmallVector<Attribute> keys;
  for (const std::string &key : policy.autotuneKeys)
    keys.push_back(string(builder, key));
  builder.create<plan::AutotuneOp>(entry.getLoc(), builder.getArrayAttr(keys));
  if (policy.stateStream) {
    SmallVector<StreamConfig> configs = {
        {64, 64, 1, 128}, {128, 64, 2, 128},
        {128, 128, 2, 256}, {64, 128, 1, 256}};
    for (const StreamConfig &config : configs)
      builder.create<plan::ConfigOp>(
          entry.getLoc(),
          builder.getDictionaryAttr(
              {builder.getNamedAttr("TILE_SIZE_M", i64(builder, config.program)),
               builder.getNamedAttr("TILE_SIZE_N", i64(builder, config.stream))}),
          i64(builder, config.stages), i64(builder, config.threads));
  } else {
    SmallVector<ContractConfig> configs;
    if (target.architecture == "sm_120" || target.architecture == "sm_121")
      configs = {{128, 64, 64, 2, 128}, {128, 128, 32, 3, 256},
                 {64, 128, 64, 2, 128}};
    else
      configs = {{128, 128, 32, 3, 128}, {128, 256, 32, 3, 256},
                 {64, 128, 64, 2, 128}, {128, 128, 64, 2, 256}};
    for (const ContractConfig &config : configs)
      builder.create<plan::ConfigOp>(
          entry.getLoc(),
          builder.getDictionaryAttr(
              {builder.getNamedAttr("TILE_SIZE_M", i64(builder, config.m)),
               builder.getNamedAttr("TILE_SIZE_N", i64(builder, config.n)),
               builder.getNamedAttr("TILE_SIZE_K", i64(builder, config.k))}),
          i64(builder, config.stages), i64(builder, config.threads));
  }
  builder.create<intent::plan::YieldOp>(entry.getLoc());
  return success();
}

} // namespace intent::tilelang::realization
