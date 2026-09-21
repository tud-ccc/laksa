/// Translation registration for DFGToDot
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "mlir/InitAllDialects.h"
#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Target/DFGToDot/DotEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerDFGToDotTranslation()
{
    TranslateFromMLIRRegistration reg(
        "dfg-to-dot",
        "translate DFG graph to Dot representation",
        [](Operation* op, raw_ostream &output) {
            return dfg::translateDFGToDot(op, output);
        },
        [](DialectRegistry &registry) {
            // Actor bodies retain computation dialects after DFG extraction,
            // even though the exporter only uses the graph topology.
            registerAllDialects(registry);
            registry.insert<dfg::DFGDialect>();
        });
}

} // namespace mlir
