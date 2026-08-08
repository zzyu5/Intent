#include "Support/Model.h"

#include "Intent/Dialect/Plan/IR/PlanOps.h"
#include "Intent/Target/CuTile/IR/CuTileOps.h"

using namespace mlir;

namespace intent::cutile::realization {
namespace {

IntegerAttr i64(OpBuilder &builder, int64_t value) {
  return builder.getI64IntegerAttr(value);
}

StringAttr string(OpBuilder &builder, StringRef value) {
  return builder.getStringAttr(value);
}

struct Config {
  int64_t m, n, k, numCtas, occupancy;
};

} // namespace

LogicalResult emitAutotuneSpace(ModuleOp module, func::FuncOp entry,
                                const TargetOptions &target,
                                const PolicyDecision &policy,
                                OpBuilder &builder) {
  SmallVector<Config> configs;
  if (target.architecture == "sm_120" || target.architecture == "sm_121") {
    configs = {{128, 64, 64, 1, 1}, {128, 64, 32, 1, 2}};
  } else {
    configs = {{128, 128, 32, 1, 1}, {256, 256, 64, 2, 1},
               {256, 256, 64, 4, 1}, {512, 256, 64, 2, 1}};
  }

  builder.setInsertionPointToEnd(module.getBody());
  auto search = builder.create<intent::plan::SearchSpaceOp>(
      entry.getLoc(), FlatSymbolRefAttr::get(module.getContext(), entry.getName()),
      string(builder, "cutile"));
  Block &body = search.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  SmallVector<Attribute> keys;
  for (const std::string &key : policy.autotuneKeys)
    keys.push_back(string(builder, key));
  builder.create<plan::AutotuneOp>(entry.getLoc(), builder.getArrayAttr(keys));
  for (const Config &config : configs)
    builder.create<plan::ConfigOp>(
        entry.getLoc(),
        builder.getDictionaryAttr(
            {builder.getNamedAttr("TILE_SIZE_M", i64(builder, config.m)),
             builder.getNamedAttr("TILE_SIZE_N", i64(builder, config.n)),
             builder.getNamedAttr("TILE_SIZE_K", i64(builder, config.k))}),
        i64(builder, config.numCtas), i64(builder, config.occupancy));
  builder.create<intent::plan::YieldOp>(entry.getLoc());
  return success();
}

} // namespace intent::cutile::realization
