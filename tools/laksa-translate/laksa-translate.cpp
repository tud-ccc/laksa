/// Main entry point for the laksa-mlir optimizer driver.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/InitAllTranslations.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

int main(int argc, char* argv[])
{
    registerAllTranslations();
    registerAllLAKSAMLIRTranslations();
    return failed(
        mlirTranslateMain(argc, argv, "LAKSA Translation Testing Tool"));
}
