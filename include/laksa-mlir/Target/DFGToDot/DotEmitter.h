/// Convenience include for the DFGToDot translation.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"

namespace mlir::dfg {

/// Translates an DFG graph in DFG dialect to Dot representation
LogicalResult translateDFGToDot(Operation* op, raw_ostream &os);

} // namespace mlir::dfg
