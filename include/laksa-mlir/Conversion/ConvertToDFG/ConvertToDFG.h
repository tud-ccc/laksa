/// Declaration of the lowering pipeline that converts programs to normalized
/// DFG IR.

#pragma once

#include "mlir/Pass/PassManager.h"

namespace mlir::laksa {

/// Normalize the input and outline each computation node as a func.func.
/// This is the common prefix used by the DFG and CPU-profiling pipelines.
void addComputationNodeOutliningPasses(OpPassManager &pm);

void addConvertToDFGPasses(OpPassManager &pm);
void registerConvertToDFGPipeline();

} // namespace mlir::laksa
