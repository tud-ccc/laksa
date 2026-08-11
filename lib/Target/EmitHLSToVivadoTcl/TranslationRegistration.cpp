/// Translation registration for EmitHLSToVivadoTcl
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSBase.h"
#include "laksa-mlir/Target/EmitHLSToVivadoTcl/VivadoTclEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

void registerEmitHLSToVivadoTclTranslation()
{
    TranslateFromMLIRRegistration reg(
        "emithls-to-vivado-tcl",
        "translate EmitHLS IR to a Vivado run_vivado.tcl script",
        [](ModuleOp op, raw_ostream &output) {
            return emithls::translateEmitHLSToVivadoTcl(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emithls::EmitHLSDialect>();
        });
}

} // namespace mlir
