/// Translation registration for DFGToMocasin
///
/// @file
/// @author     Giuseppe Meloni (giuseppe.meloni@abinsula.com)

#include "mlir/InitAllDialects.h"
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
        [](DialectRegistry &registry) {
            // Actor bodies retain computation dialects after DFG extraction,
            // even though the exporter only uses the graph topology.
            registerAllDialects(registry);
            registry.insert<dfg::DFGDialect>();
        });
}

} // namespace mlir
