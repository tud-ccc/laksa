/// Translation registration for EmitHLSToLaksaHeader
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "laksa-mlir/Target/EmitHLSToLaksaHeader/LaksaHeaderEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerEmitHLSToLaksaHeaderTranslation()
{
    TranslateFromMLIRRegistration reg(
        "emithls-to-laksa-header",
        "translate EmitHLS IR to a C header of buffer sizes and AXI-Lite "
        "register offsets for the laksa-hls-kria-driver's userspace API",
        [](ModuleOp op, raw_ostream &output) {
            return emithls::translateEmitHLSToLaksaHeader(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emithls::EmitHLSDialect>();
        });
}

} // namespace mlir
