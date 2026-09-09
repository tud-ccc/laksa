/// Convenience include for the EmitHLSToHLSBuildScript translation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#pragma once

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"

namespace mlir::emithls {

/// Translates a program in EmitHLS IR to a shell script running the generated
/// Vitis HLS and Vivado scripts and extracting the bitstream from the XSA.
LogicalResult translateEmitHLSToHLSBuildScript(ModuleOp op, raw_ostream &os);

} // namespace mlir::emithls
