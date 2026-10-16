/// Translation registration for EmitHLSToDot
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "laksa-mlir/Target/EmitHLSToDot/HLSDotEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerEmitHLSToDotTranslation()
{
    TranslateFromMLIRRegistration reg(
        "emithls-to-dot",
        "translate the EmitHLS dataflow to Dot representation",
        [](ModuleOp op, raw_ostream &output) {
            return emithls::translateEmitHLSToDot(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emithls::EmitHLSDialect>();
        });
}

} // namespace mlir
