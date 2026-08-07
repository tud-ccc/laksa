/// Convenience include for the EmitHLSToCpp translation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"

namespace mlir::emithls {

/// Translates a program in EmitHLS IR to HLS C++ representation
LogicalResult translateEmitHLSToCpp(Operation* op, raw_ostream &os);

} // namespace mlir::emithls
