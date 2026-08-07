/// Main entry point for the laksa-mlir MLIR language server.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/InitAllDialects.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

int main(int argc, char* argv[])
{
    // MLIR Upstream Registration
    DialectRegistry registry;
    registerAllDialects(registry);

    // LAKSA-MLIR Registration
    registerAllLAKSAMLIRDialects(registry);

    return failed(MlirLspServerMain(argc, argv, registry));
}
