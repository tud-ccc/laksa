/// Convenience include for the EmitHLSToDot translation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"

namespace mlir::emithls {

/// Translates the dataflow of the top function in EmitHLS IR to Dot
LogicalResult translateEmitHLSToDot(ModuleOp op, raw_ostream &os);

} // namespace mlir::emithls
