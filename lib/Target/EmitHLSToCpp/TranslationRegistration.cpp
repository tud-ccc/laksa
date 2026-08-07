/// Translation registration for EmitHLSToCpp
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "laksa-mlir/Target/EmitHLSToCpp/HLSCppEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerEmitHLSToCppTranslation()
{
    TranslateFromMLIRRegistration reg(
        "emithls-to-cpp",
        "translate EmitHLS IR to HLS C++",
        [](Operation* op, raw_ostream &output) {
            return emithls::translateEmitHLSToCpp(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emithls::EmitHLSDialect>();
        });
}

} // namespace mlir
