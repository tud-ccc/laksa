/// Translation registration for EmitHLSToLaksaApp
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "laksa-mlir/Target/EmitHLSToLaksaApp/LaksaAppEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerEmitHLSToLaksaAppTranslation()
{
    TranslateFromMLIRRegistration reg(
        "emithls-to-laksa-app",
        "translate EmitHLS IR to a C application driving the design through "
        "the laksa-hls-kria-driver's userspace API",
        [](ModuleOp op, raw_ostream &output) {
            return emithls::translateEmitHLSToLaksaApp(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emithls::EmitHLSDialect>();
        });
}

} // namespace mlir
