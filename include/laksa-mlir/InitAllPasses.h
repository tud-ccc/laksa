/// Register all passes in this project.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Conversion/ConvertToEmitC/ConvertToEmitC.h"
#include "laksa-mlir/Conversion/ConvertToEmitHLS/ConvertToEmitHLS.h"
#include "laksa-mlir/Conversion/Passes.h"
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h"
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h"
#include "laksa-mlir/Dialect/Func/Transforms/Passes.h"
#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>

namespace mlir {

inline void registerAllLAKSAMLIRPasses()
{
    // Transformation passes
    dfg::registerDFGPasses();
    emithls::registerEmitHLSPasses();
    func::registerFuncExtPasses();
    linalg::registerLinalgExtPasses();

    // Conversion passes
    laksa::registerLAKSAConversionPasses();

    // Conversion pipelines
    laksa::registerConvertToEmitCPipelines();
    laksa::registerConvertToEmitHLSPipelines();
}

} // namespace mlir
