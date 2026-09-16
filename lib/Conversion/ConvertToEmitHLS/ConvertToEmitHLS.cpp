/// Implementation of the ConvertToEmitHLS pass pipeline.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Conversion/ConvertToEmitHLS/ConvertToEmitHLS.h"

#include "laksa-mlir/Conversion/ConvertToDFG/ConvertToDFG.h"
#include "laksa-mlir/Conversion/Passes.h"
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h"
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h"
#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

namespace {
struct ConvertToEmitHLSPipelineOptions
        : public PassPipelineOptions<ConvertToEmitHLSPipelineOptions> {
    Option<int64_t> availableBRAM{
        *this,
        "available-bram",
        llvm::cl::desc("Total Number of BRAM on the hardware"),
        llvm::cl::init(288)};
    Option<int64_t> availableDSP{
        *this,
        "available-dsp",
        llvm::cl::desc("Total Number of DSP on the hardware"),
        llvm::cl::init(1248)};
};
} // namespace

void mlir::laksa::addConvertToEmitHLSPasses(
    OpPassManager &pm,
    int64_t availableBRAM,
    int64_t availableDSP)
{
    addConvertToDFGPasses(pm);
    pm.addPass(dfg::createDFGOperatorToProcessPass());
    pm.addPass(bufferization::createOneShotBufferizePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(linalg::createLinalgMapToGenericPass());
    pm.addPass(createConvertLinalgToLAKSALoopsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(dfg::createDFGIONormalizationPass());
    pm.addPass(createConvertMemRefPadToLAKSALoopsPass());
    pm.addPass(createConvertAffineToEmitHLSPass());
    pm.addPass(createConvertArithToEmitHLSPass());
    pm.addPass(dfg::createDFGCollapseUnitDimsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createConvertDFGToEmitHLSPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(emithls::createEmitHLSFuseOperatorPass());
    pm.addPass(emithls::createEmitHLSResolveHelpersPass());
    pm.addPass(emithls::createEmitHLSLoopFusionPass());
    pm.addPass(emithls::createEmitHLSFuseOperatorPass());
    pm.addPass(emithls::createEmitHLSFoldExpressionPass());
    pm.addPass(emithls::createEmitHLSAddIOFunctionsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(
        emithls::createEmitHLSPragmaDSEPass(availableBRAM, availableDSP));
    pm.addPass(emithls::createEmitHLSPragmaInsertionPass());
    pm.addPass(emithls::createEmitHLSInsertIncludesPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
}

void mlir::laksa::registerConvertToEmitHLSPipelines()
{
    PassPipelineRegistration<ConvertToEmitHLSPipelineOptions>(
        "convert-to-emithls",
        "Convert everything to emithls dialect",
        [](OpPassManager &pm, const ConvertToEmitHLSPipelineOptions &options) {
            addConvertToEmitHLSPasses(
                pm,
                options.availableBRAM,
                options.availableDSP);
        });
}
