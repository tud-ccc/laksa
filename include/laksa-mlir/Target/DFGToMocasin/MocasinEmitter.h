/// Convenience include for the DFGToMocasin translation.
///
/// @file
/// @author     Giuseppe Meloni (giuseppe.meloni@abinsula.com)

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"

namespace mlir::dfg {

/// Translates a DFG graph in the DFG dialect to the YAML format consumed by
/// the Mocasin mapping/simulation tool.
LogicalResult translateDFGToMocasinYAML(Operation* op, raw_ostream &os);

} // namespace mlir::dfg
