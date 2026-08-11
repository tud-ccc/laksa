/// Convenience include for the EmitHLSToHLSTcl translation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"

namespace mlir::emithls {

/// Translates a program in EmitHLS IR to a Vitis HLS `run_hls.tcl` script
LogicalResult translateEmitHLSToHLSTcl(ModuleOp op, raw_ostream &os);

} // namespace mlir::emithls
