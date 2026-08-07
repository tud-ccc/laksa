/// Main entry point for the laksa-mlir optimizer driver.
///
/// @file
/// @author     Felix Suchert (felix.suchert@tu-dresden.de)
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/InitAllDialects.h"
#include "laksa-mlir/InitAllPasses.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include <mlir/InitAllExtensions.h>

using namespace mlir;

int main(int argc, char* argv[])
{
    // MLIR Upstream Registration
    DialectRegistry registry;
    registerAllDialects(registry);
    registerAllPasses();
    registerAllExtensions(registry);

    // LAKSA-MLIR Registration
    registerAllLAKSAMLIRDialects(registry);
    registerAllLAKSAMLIRPasses();

    return asMainReturnCode(
        MlirOptMain(argc, argv, "laksa-mlir optimizer driver\n", registry));
}
