/// Translation registration for EmitHLSToHLSProfileScript
///
/// @file
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "laksa-mlir/Target/EmitHLSToHLSProfileScript/HLSProfileScriptEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

void registerEmitHLSToHLSProfileScriptTranslation()
{
    TranslateFromMLIRRegistration registration(
        "emithls-to-hls-profile-script",
        "generate a script extracting profiles from Vitis HLS reports",
        [](ModuleOp op, raw_ostream &output) {
            return emithls::translateEmitHLSToHLSProfileScript(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emithls::EmitHLSDialect>();
        });
}

} // namespace mlir
