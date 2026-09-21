/// Implementation of the pipeline that converts programs to normalized DFG
/// IR.
///
/// @file
/// @author     Robert Khasanov (robert.khasanov@tu-dresden.de)

#include "laksa-mlir/Conversion/ConvertToDFG/ConvertToDFG.h"

#include "laksa-mlir/Conversion/Passes.h"
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h"
#include "laksa-mlir/Dialect/Func/Transforms/Passes.h"
#include "laksa-mlir/Dialect/Linalg/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

void mlir::laksa::addComputationNodeOutliningPasses(OpPassManager &pm)
{
    pm.addPass(linalg::createLinalgSoftTransposePass());
    pm.addPass(createLinalgGeneralizeNamedOpsPass());
    pm.addPass(linalg::createLinalgScalarizeSplatDensePass());
    pm.addPass(createLinalgInlineScalarOperandsPass());
    pm.nest("func.func").addPass(func::createFuncOutlineComputationLeafPass());
}

void mlir::laksa::addConvertToDFGPasses(OpPassManager &pm)
{
    addComputationNodeOutliningPasses(pm);
    pm.addPass(createConvertFuncToDFGPass());
    pm.addPass(dfg::createDFGChannelFanoutExpansionPass());
}

void mlir::laksa::registerConvertToDFGPipeline()
{
    PassPipelineRegistration<>(
        "convert-to-dfg",
        "Convert a program to normalized DFG IR",
        [](OpPassManager &pm) { addConvertToDFGPasses(pm); });
}
