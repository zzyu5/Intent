#include "Intent/Target/GPU/Transforms/Analysis/PhysicalProgram.h"

#include "Intent/Target/GPU/Realization/Analysis.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace intent::gpu {

PhysicalProgramAnalysis::PhysicalProgramAnalysis(target::KernelModel model)
    : kernel(std::make_unique<target::KernelModel>(std::move(model))),
      facts(std::make_unique<target::KernelFacts>(*kernel)) {}

plan::RangeOp PhysicalProgramAnalysis::getRange(int64_t axisNode,
                                                StringRef purpose,
                                                int64_t level) const {
  auto found = ranges.find(axisNode);
  if (found == ranges.end())
    return {};
  auto range = llvm::find_if(found->second, [&](plan::RangeOp candidate) {
    return candidate.getPurpose() == purpose &&
           static_cast<int64_t>(candidate.getLevel()) == level;
  });
  return range == found->second.end() ? plan::RangeOp() : *range;
}

bool PhysicalProgramAnalysis::hasWorkerReuse() const {
  return llvm::any_of(axes, [](const auto &entry) {
    plan::AxisOp axis = entry.second;
    return axis.getReuseWorker();
  });
}

bool PhysicalProgramAnalysis::axisHasRole(int64_t node,
                                          StringRef expected) const {
  plan::AxisOp axis = getAxis(node);
  return axis && llvm::any_of(axis.getRoles(), [&](Attribute attribute) {
           auto role = dyn_cast<StringAttr>(attribute);
           return role && role.getValue() == expected;
         });
}

bool PhysicalProgramAnalysis::isScalarAxis(int64_t node) const {
  plan::AxisOp axis = getAxis(node);
  if (!axis)
    return false;
  plan::RangeOp range;
  if (axisHasRole(node, "parallel"))
    range = getRange(node, "ownership");
  else if (axisHasRole(node, "ordered"))
    range = getRange(node, "traversal");
  else if (axisHasRole(node, "reduction"))
    range = getRange(node, "reduction");
  else if (axisHasRole(node, "lane"))
    range = getRange(node, "lane");
  return range && range.getTile() == "one";
}

bool PhysicalProgramAnalysis::isPackedScalarAxis(int64_t node) const {
  return axisHasRole(node, "parallel") && axisHasRole(node, "lane") &&
         axisHasRole(node, "packed_lane");
}

FailureOr<std::unique_ptr<PhysicalProgramAnalysis>>
PhysicalProgramAnalysis::compute(plan::ProgramOp program) {
  FailureOr<func::FuncOp> entry = plan::getPhysicalEntry(program);
  FailureOr<target::KernelModel> model =
      succeeded(entry) ? target::analyzeKernel(*entry)
                       : FailureOr<target::KernelModel>(failure());
  if (failed(entry) || failed(model))
    return failure();
  auto analysis = std::unique_ptr<PhysicalProgramAnalysis>(
      new PhysicalProgramAnalysis(std::move(*model)));
  if (failed(realization::analyzeOperations(analysis->getFacts())))
    return failure();
  analysis->program = program;
  for (Operation &operation : program.getBody().front()) {
    if (auto launch = dyn_cast<plan::LaunchOp>(operation))
      analysis->launch = launch;
    else if (auto axis = dyn_cast<plan::AxisOp>(operation))
      analysis->axes[axis.getNode()] = axis;
    else if (auto range = dyn_cast<plan::RangeOp>(operation))
      analysis->ranges[range.getAxisNode()].push_back(range);
    else if (auto stage = dyn_cast<plan::StageOp>(operation))
      analysis->stages.push_back(stage);
  }
  return analysis;
}

} // namespace intent::gpu
