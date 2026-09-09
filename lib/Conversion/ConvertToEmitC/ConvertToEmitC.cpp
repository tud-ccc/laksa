/// Implementation of the ConvertToEmitC pass pipeline.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Conversion/ConvertToEmitC/ConvertToEmitC.h"

#include "laksa-mlir/Conversion/ReshapedCopyToLoops/ReshapedCopyToLoops.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ConvertToEmitC/ConvertToEmitCPass.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <mlir/Transforms/Passes.h>

using namespace mlir;

namespace {
struct ConvertToEmitCPipelineOptions
        : public PassPipelineOptions<ConvertToEmitCPipelineOptions> {
    Option<uint32_t> maxAllocSizeInBytes{
        *this,
        "max-alloc-size-in-bytes",
        llvm::cl::desc(
            "Maximal size in bytes of a heap allocation that is "
            "promoted to the stack. EmitC has no lowering for "
            "memref.alloc, so anything above this limit makes the "
            "conversion fail."),
        llvm::cl::init(100000000)};
};
} // namespace

void mlir::laksa::addConvertToEmitCPasses(
    OpPassManager &pm,
    uint32_t maxAllocSizeInBytes)
{
    // Bufferize with identity layout maps: EmitC has no notion of a strided
    // memref, so the function boundaries must come out as plain arrays.
    bufferization::OneShotBufferizePassOptions bufferizeOptions;
    bufferizeOptions.bufferizeFunctionBoundaries = true;
    bufferizeOptions.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    // MemRefToEmitC rejects memref.global/memref.alloc carrying an alignment
    // attribute, so do not let bufferization attach one.
    bufferizeOptions.bufferAlignment = 0;
    pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));
    // A memref-returning function has no C equivalent; turn the result into an
    // out-parameter instead. Public functions are opted in explicitly, as the
    // entry point is the one that matters here.
    bufferization::BufferResultsToOutParamsPassOptions outParamsOptions;
    outParamsOptions.modifyPublicFunctions = true;
    outParamsOptions.hoistStaticAllocs = true;
    pm.addPass(
        bufferization::createBufferResultsToOutParamsPass(outParamsOptions));
    pm.addPass(createConvertLinalgToLoopsPass());
    // Fold the reshapes bufferization left behind into the loads and stores
    // that consume them.
    pm.addPass(memref::createFoldMemRefAliasOpsPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createConvertReshapedCopyToLoopsPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    // Whatever allocation survived out-param hoisting has to become a stack
    // array, the only kind of buffer EmitC can express.
    bufferization::PromoteBuffersToStackPassOptions promoteOptions;
    promoteOptions.maxAllocSizeInBytes = maxAllocSizeInBytes;
    pm.nest("func.func")
        .addPass(
            bufferization::createPromoteBuffersToStackPass(promoteOptions));
    // ArithToEmitC has no pattern for the min/max ops; expand them into the
    // cmpi/select pairs it does handle.
    pm.addPass(arith::createArithExpandOpsPass());
    pm.addPass(createConvertToEmitC());
}

void mlir::laksa::registerConvertToEmitCPipelines()
{
    PassPipelineRegistration<ConvertToEmitCPipelineOptions>(
        "convert-to-laksa-emitc",
        "Convert everything to the upstream emitc dialect",
        [](OpPassManager &pm, const ConvertToEmitCPipelineOptions &options) {
            addConvertToEmitCPasses(pm, options.maxAllocSizeInBytes);
        });
}
