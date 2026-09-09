/// Convenience include for the EmitHLSToLaksaRunScript translation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"

namespace mlir::emithls {

/// Translates a program in EmitHLS IR to a shell script loading the design on
/// a Kria board and checking what it computes against the scalar reference.
LogicalResult translateEmitHLSToLaksaRunScript(ModuleOp op, raw_ostream &os);

} // namespace mlir::emithls
