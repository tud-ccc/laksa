/// Translation registration for EmitHLSToLaksaRunScript
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "laksa-mlir/Target/EmitHLSToLaksaRunScript/LaksaRunScriptEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerEmitHLSToLaksaRunScriptTranslation()
{
    TranslateFromMLIRRegistration reg(
        "emithls-to-laksa-run-script",
        "translate the design to a shell script loading and checking it on a Kria board",
        [](ModuleOp op, raw_ostream &output) {
            return emithls::translateEmitHLSToLaksaRunScript(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emithls::EmitHLSDialect>();
        });
}

} // namespace mlir
