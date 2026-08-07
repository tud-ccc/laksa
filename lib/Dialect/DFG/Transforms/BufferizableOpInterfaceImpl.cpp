/// BufferizableOpInterface implementation
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/Transforms/BufferizableOpInterfaceImpl.h"

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGBase.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"

#include <llvm/Support/Debug.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::dfg;

namespace mlir::dfg {
namespace {

struct PullOpInterface : public BufferizableOpInterface::
                             ExternalModel<PullOpInterface, PullAsTensorOp> {
    bool
    bufferizesToMemoryRead(Operation*, OpOperand &, const AnalysisState &) const
    { return false; }
    bool bufferizesToMemoryWrite(Operation*, OpOperand &, const AnalysisState &)
        const
    { return false; }
    AliasingValueList
    getAliasingValues(Operation* op, OpOperand &, const AnalysisState &) const
    {
        return {
            {op->getOpResult(0) /*result*/,
             BufferRelation::Equivalent,
             /*isDefinite=*/false}
        };
    }
    bool hasTensorSemantics(Operation*) const { return true; }
    LogicalResult bufferize(
        Operation* op,
        RewriterBase &rewriter,
        const BufferizationOptions &,
        BufferizationState &) const
    {
        auto pullOp = cast<PullAsTensorOp>(op);
        auto newPull = PullAsMemRefOp::create(
            rewriter,
            op->getLoc(),
            pullOp.getReadPort());
        replaceOpWithBufferizedValues(rewriter, op, newPull.getTokenMemref());
        return success();
    }
};

struct PushOpInterface : public BufferizableOpInterface::
                             ExternalModel<PushOpInterface, PushTensorOp> {
    bool
    bufferizesToMemoryRead(Operation*, OpOperand &, const AnalysisState &) const
    { return true; }
    bool bufferizesToMemoryWrite(Operation*, OpOperand &, const AnalysisState &)
        const
    { return false; }
    AliasingValueList
    getAliasingValues(Operation*, OpOperand &, const AnalysisState &) const
    { return {}; }
    bool hasTensorSemantics(Operation*) const { return true; }
    LogicalResult bufferize(
        Operation* op,
        RewriterBase &rewriter,
        const BufferizationOptions &options,
        BufferizationState &state) const
    {
        auto pushOp = cast<PushTensorOp>(op);
        FailureOr<Value> buffer =
            getBuffer(rewriter, pushOp.getTokenTensor(), options, state);
        if (failed(buffer)) return failure();
        rewriter.replaceOpWithNewOp<PushMemRefOp>(
            op,
            *buffer,
            pushOp.getWritePort());
        return success();
    }
};

} // namespace
} // namespace mlir::dfg

void mlir::dfg::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry)
{
    registry.addExtension(+[](MLIRContext* ctx, dfg::DFGDialect*) {
        PullAsTensorOp::attachInterface<PullOpInterface>(*ctx);
        PushTensorOp::attachInterface<PushOpInterface>(*ctx);
    });
}
