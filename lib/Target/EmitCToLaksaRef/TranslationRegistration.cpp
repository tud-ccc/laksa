/// Translation registration for EmitCToLaksaRef
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Target/EmitCToLaksaRef/LaksaRefEmitter.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerEmitCToLaksaRefTranslation()
{
    TranslateFromMLIRRegistration reg(
        "emitc-to-laksa-ref",
        "translate the emitc IR of the scalar reference to a C program "
        "checking the board's output against it",
        [](ModuleOp op, raw_ostream &output) {
            return laksa::translateEmitCToLaksaRef(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emitc::EmitCDialect>();
        });
}

} // namespace mlir
