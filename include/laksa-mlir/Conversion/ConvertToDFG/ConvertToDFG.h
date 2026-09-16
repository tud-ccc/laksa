/// Declaration of the lowering pipeline that converts programs to normalized
/// DFG IR.

#pragma once

#include "mlir/Pass/PassManager.h"

namespace mlir::laksa {

void addConvertToDFGPasses(OpPassManager &pm);
void registerConvertToDFGPipeline();

} // namespace mlir::laksa
