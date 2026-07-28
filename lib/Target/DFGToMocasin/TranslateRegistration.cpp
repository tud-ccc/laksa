/// Translation registration for DFGToMocasin
///
/// @file
/// @author     Giuseppe Meloni (giuseppe.meloni@abinsula.com)

#include "mlir/Tools/mlir-translate/Translation.h"
#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Target/DFGToMocasin/MocasinEmitter.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerDFGToMocasinTranslation()
{
    TranslateFromMLIRRegistration reg(
        "dfg-to-mocasin",
        "translate DFG graph to the Mocasin YAML input format",
        [](Operation* op, raw_ostream &output) {
            return dfg::translateDFGToMocasinYAML(op, output);
        },
        [](DialectRegistry &registry) { registry.insert<dfg::DFGDialect>(); });
}

} // namespace mlir
