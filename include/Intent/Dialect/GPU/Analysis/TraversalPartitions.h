#ifndef INTENT_DIALECT_GPU_ANALYSIS_TRAVERSALPARTITIONS_H
#define INTENT_DIALECT_GPU_ANALYSIS_TRAVERSALPARTITIONS_H

namespace mlir { class Operation; }

namespace intent::gpu {
class PhysicalProgramAnalysis;

// Prove disjoint active Store member domains from their actual access ranges
// and matched control paths. An ancestor loop/If shape alone is not evidence;
// scalar effects without an actual traversal-coordinate anchor are unknown.
bool areDisjointTraversalPartitions(mlir::Operation *first,
                                   mlir::Operation *second,
                                   PhysicalProgramAnalysis &analysis);

} // namespace intent::gpu
#endif
