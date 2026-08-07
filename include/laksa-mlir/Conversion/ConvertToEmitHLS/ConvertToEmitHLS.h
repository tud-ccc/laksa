/// Declaration of the everything to emithls lowering pass that lowers
/// programs to the emithls dialect.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "mlir/Pass/PassManager.h"

#include <cstdint>

namespace mlir {

namespace laksa {
void registerConvertToEmitHLSPipelines();
void addConvertToEmitHLSPasses(
    OpPassManager &pm,
    int64_t availableBRAM,
    int64_t availableDSP);
} // namespace laksa

} // namespace mlir
