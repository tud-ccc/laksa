/// Convenience include for the DFGToMocasin translation.
///
/// @file
/// @author     Giuseppe Meloni (giuseppe.meloni@abinsula.com)
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"

namespace mlir::dfg {

/// Translates a DFG graph in the DFG dialect to the YAML format consumed by
/// the Mocasin mapping/simulation tool.
LogicalResult translateDFGToMocasinYAML(Operation* op, raw_ostream &os);

/// Emits the helper script that combines profile fragments with the generated
/// Mocasin application template.
LogicalResult emitMocasinMergeScript(raw_ostream &os);

} // namespace mlir::dfg
