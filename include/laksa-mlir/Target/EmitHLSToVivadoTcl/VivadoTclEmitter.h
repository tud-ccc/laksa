/// Convenience include for the EmitHLSToVivadoTcl translation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"

namespace mlir::emithls {

/// Translates a program in EmitHLS IR to a Vivado `run_vivado.tcl` script
LogicalResult translateEmitHLSToVivadoTcl(ModuleOp op, raw_ostream &os);

} // namespace mlir::emithls
