/// Declaration of the everything to emitc lowering pipeline that lowers
/// programs to the upstream emitc dialect.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/PassManager.h"

#include <cstdint>

namespace mlir {

namespace laksa {
void registerConvertToEmitCPipelines();
void addConvertToEmitCPasses(OpPassManager &pm, uint32_t maxAllocSizeInBytes);
} // namespace laksa

} // namespace mlir
