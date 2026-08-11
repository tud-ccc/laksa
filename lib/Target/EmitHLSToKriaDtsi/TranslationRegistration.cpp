/// Translation registration for EmitHLSToKriaDtsi
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "laksa-mlir/Target/EmitHLSToKriaDtsi/KriaDtsiEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerEmitHLSToKriaDtsiTranslation()
{
    TranslateFromMLIRRegistration reg(
        "emithls-to-kria-dtsi",
        "translate EmitHLS IR to a Kria device tree overlay source (.dts) "
        "for the laksa-hls-kria-driver",
        [](ModuleOp op, raw_ostream &output) {
            return emithls::translateEmitHLSToKriaDtsi(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emithls::EmitHLSDialect>();
        });
}

} // namespace mlir
