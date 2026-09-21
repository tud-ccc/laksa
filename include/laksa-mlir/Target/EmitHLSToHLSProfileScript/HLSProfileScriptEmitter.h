/// Convenience include for the EmitHLSToHLSProfileScript translation.
///
/// @file
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#pragma once

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"

namespace mlir::emithls {

/// Translates a program in EmitHLS IR to a shell script extracting process
/// cycle profiles from the Vitis HLS reports.
LogicalResult translateEmitHLSToHLSProfileScript(ModuleOp op, raw_ostream &os);

} // namespace mlir::emithls
