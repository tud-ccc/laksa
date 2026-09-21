/// Translation registration for EmitCToCPUProfile.
///
/// @file
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#include "laksa-mlir/Target/EmitCToCPUProfile/CPUProfileEmitter.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

void registerEmitCToCPUProfileTranslation()
{
    TranslateFromMLIRRegistration benchmarkRegistration(
        "emitc-to-cpu-profile",
        "generate a native per-node cycle benchmark from emitc IR",
        [](ModuleOp op, llvm::raw_ostream &output) {
            return laksa::translateEmitCToCPUProfile(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emitc::EmitCDialect>();
        });

    TranslateFromMLIRRegistration nodesRegistration(
        "emitc-to-cpu-profile-nodes",
        "translate outlined emitc nodes to a self-contained C++ source",
        [](ModuleOp op, llvm::raw_ostream &output) {
            return laksa::translateEmitCToCPUProfileNodes(op, output);
        },
        [](DialectRegistry &registry) {
            registry.insert<emitc::EmitCDialect>();
        });
}

} // namespace mlir
