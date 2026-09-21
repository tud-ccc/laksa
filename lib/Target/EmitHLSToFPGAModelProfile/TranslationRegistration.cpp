/// Translation registration for EmitHLSToFPGAModelProfile.
///
/// @file
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "laksa-mlir/Target/EmitHLSToFPGAModelProfile/FPGAModelProfileEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

void registerEmitHLSToFPGAModelProfileTranslation()
{
    TranslateFromMLIRRegistration registration(
        "emithls-to-model-profile",
        "translate LAKSA FPGA model cycle counts to a Mocasin profile",
        [](ModuleOp op, llvm::raw_ostream &output) {
            return emithls::translateEmitHLSToFPGAModelProfile(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emithls::EmitHLSDialect>();
        });
}

} // namespace mlir
