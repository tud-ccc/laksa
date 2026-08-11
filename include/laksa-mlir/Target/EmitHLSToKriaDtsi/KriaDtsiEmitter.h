/// Convenience include for the EmitHLSToKriaDtsi translation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"

namespace mlir::emithls {

/// Translates a program in EmitHLS IR to a device tree overlay source
/// (`.dts`) that declares a `tud,laksa`-compatible node for the design's
/// AXI-Lite control interface, ready to be compiled with `dtc` and applied
/// with `fpgautil -o` on a Kria board.
LogicalResult translateEmitHLSToKriaDtsi(ModuleOp op, raw_ostream &os);

} // namespace mlir::emithls
