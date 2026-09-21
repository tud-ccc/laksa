/// Translation registration for EmitCToCPUProfileRunScript.
///
/// @file
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#include "laksa-mlir/Target/EmitCToCPUProfileRunScript/CPUProfileRunScriptEmitter.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

void registerEmitCToCPUProfileRunScriptTranslation()
{
    TranslateFromMLIRRegistration registration(
        "emitc-to-cpu-profile-run-script",
        "generate the CPU profiling benchmark run script",
        [](ModuleOp op, llvm::raw_ostream &output) {
            return laksa::translateEmitCToCPUProfileRunScript(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emitc::EmitCDialect>();
        });
}

} // namespace mlir
