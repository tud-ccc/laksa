/// Translation registration for DFGToMocasin
///
/// @file
/// @author     Giuseppe Meloni (giuseppe.meloni@abinsula.com)
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

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
    auto registerDialects = [](DialectRegistry &registry) {
        // Actor bodies retain computation dialects after DFG extraction, even
        // though the exporters only use the graph topology.
        registerAllDialects(registry);
        registry.insert<dfg::DFGDialect>();
    };

    TranslateFromMLIRRegistration reg(
        "dfg-to-mocasin",
        "translate DFG graph to the Mocasin YAML input format",
        [](Operation* op, raw_ostream &output) {
            return dfg::translateDFGToMocasinYAML(op, output);
        },
        registerDialects);

    TranslateFromMLIRRegistration mergeScriptReg(
        "dfg-to-mocasin-merge-script",
        "generate the script that merges Mocasin execution profiles",
        [](Operation*, raw_ostream &output) {
            return dfg::emitMocasinMergeScript(output);
        },
        registerDialects);
}

} // namespace mlir
