/// Translation registration for EmitHLSToHLSTcl
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "laksa-mlir/Target/EmitHLSToHLSTcl/HLSTclEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerEmitHLSToHLSTclTranslation()
{
    TranslateFromMLIRRegistration reg(
        "emithls-to-hls-tcl",
        "translate EmitHLS IR to a Vitis HLS run_hls.tcl script",
        [](ModuleOp op, raw_ostream &output) {
            return emithls::translateEmitHLSToHLSTcl(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emithls::EmitHLSDialect>();
        });
}

} // namespace mlir
