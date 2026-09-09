/// Translation registration for EmitHLSToHLSBuildScript
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "laksa-mlir/Target/EmitHLSToHLSBuildScript/HLSBuildScriptEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerEmitHLSToHLSBuildScriptTranslation()
{
    TranslateFromMLIRRegistration reg(
        "emithls-to-hls-build-script",
        "translate the design to a shell script building its bitstream",
        [](ModuleOp op, raw_ostream &output) {
            return emithls::translateEmitHLSToHLSBuildScript(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emithls::EmitHLSDialect>();
        });
}

} // namespace mlir
