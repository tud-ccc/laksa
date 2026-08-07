/// Implementation of ChannelFanoutExpansion transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGOps.h"
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Value.h>
#include <string>
#include <utility>

#define DEBUG_TYPE "dfg-channel-fanout-expansion"
#define LAKSA_DEBUG(X)                                                           \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[dfg-channel-fanout-expansion] "; X;                  \
        llvm::dbgs() << "\n")

using namespace mlir;
using namespace mlir::dfg;

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGCHANNELFANOUTEXPANSION
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

namespace {
struct MultiplyChannels : public OpRewritePattern<ChannelOp> {
    MultiplyChannels(MLIRContext* context)
            : OpRewritePattern<ChannelOp>(context) {};

    LogicalResult
    matchAndRewrite(ChannelOp op, PatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto outputChanPort = op.getOutputPort();
        auto numChanFanout = outputChanPort.getNumUses();
        LAKSA_DEBUG(
            llvm::dbgs() << "Channel at " << loc << " has output port used "
                         << numChanFanout << " times.");

        auto instantiateOp = dyn_cast<InstantiateOp>(op.getInputConnectedOp());
        if (!instantiateOp)
            return rewriter.notifyMatchFailure(
                loc,
                "Channel's input port should be used by an instantiate "
                "operation.");
        auto instantiatedOperator =
            dyn_cast<OperatorOp>(instantiateOp.getInstantiatedOperation());
        if (!instantiatedOperator)
            return rewriter.notifyMatchFailure(
                loc,
                "Should instantiate an operator here.");

        IRMapping mapper;
        SmallVector<std::pair<Value, Value>> clonedChannelPorts;
        rewriter.setInsertionPointAfter(op);
        for (unsigned i = 0; i < numChanFanout - 1; ++i) {
            auto clonedOp = rewriter.clone(*op.getOperation(), mapper);
            auto clonedChannel = cast<ChannelOp>(clonedOp);
            clonedChannelPorts.push_back(
                std::make_pair(
                    clonedChannel.getInputPort(),
                    clonedChannel.getOutputPort()));
        }
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Cloned the channel " << numChanFanout - 1 << " time(s).");

        // Replace the last N-1 uses of output port to the new ports.
        // Snapshot uses first to avoid iterating a list that set() modifies.
        SmallVector<OpOperand*> outputChanUses;
        for (auto &use : outputChanPort.getUses())
            outputChanUses.push_back(&use);
        for (auto [idx, use] : llvm::enumerate(outputChanUses)) {
            if (idx == numChanFanout - 1) continue;
            use->set(clonedChannelPorts[idx].second);
        }

        // Replace the instantiate operation with new output list
        unsigned outputIdx = 0;
        SmallVector<Value> newInstantiateOutputs;
        for (auto [i, outputOperand] :
             llvm::enumerate(instantiateOp.getOutputPorts())) {
            if (outputOperand == op.getInputPort()) {
                outputIdx = i;
                newInstantiateOutputs.push_back(outputOperand);
                llvm::append_range(
                    newInstantiateOutputs,
                    llvm::make_first_range(clonedChannelPorts));
            } else {
                newInstantiateOutputs.push_back(outputOperand);
            }
        }
        rewriter.setInsertionPoint(instantiateOp);
        rewriter.replaceOpWithNewOp<InstantiateOp>(
            instantiateOp,
            instantiateOp.getNodeName(),
            instantiateOp.getInputs(),
            newInstantiateOutputs);

        // Update OperatorOp's FunctionType to include the new output ports
        auto &operatorBody = instantiatedOperator.getBody();
        auto outputOp = cast<OutputOp>(operatorBody.front().getTerminator());

        SmallVector<Value> newOutputOperands(outputOp.getOperands());
        Value duplicatedValue = newOutputOperands[outputIdx];
        newOutputOperands.insert(
            newOutputOperands.begin() + outputIdx + 1,
            numChanFanout - 1,
            duplicatedValue);

        auto funcType = instantiatedOperator.getFunctionType();
        SmallVector<Type> newResultTypes(funcType.getResults());
        Type duplicatedType = newResultTypes[outputIdx];
        newResultTypes.insert(
            newResultTypes.begin() + outputIdx + 1,
            numChanFanout - 1,
            duplicatedType);
        auto newFuncType = FunctionType::get(
            rewriter.getContext(),
            funcType.getInputs(),
            newResultTypes);

        // Update operator operation and its output operation
        rewriter.modifyOpInPlace(instantiatedOperator, [&]() {
            instantiatedOperator.setFunctionType(newFuncType);
        });
        rewriter.setInsertionPoint(outputOp);
        rewriter.replaceOpWithNewOp<OutputOp>(outputOp, newOutputOperands);

        return success();
    }
};
} // namespace

namespace {
struct DFGChannelFanoutExpansionPass
        : public dfg::impl::DFGChannelFanoutExpansionBase<
              DFGChannelFanoutExpansionPass> {
    void runOnOperation() override
    {
        ConversionTarget target(getContext());
        RewritePatternSet patterns(&getContext());

        patterns.add<MultiplyChannels>(&getContext());

        target.addLegalDialect<DFGDialect>();
        target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
        target.addDynamicallyLegalOp<ChannelOp>(
            [](ChannelOp op) { return op.getOutputPort().hasOneUse(); });

        if (failed(applyPartialConversion(
                getOperation(),
                target,
                std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // namespace

std::unique_ptr<Pass> mlir::dfg::createDFGChannelFanoutExpansionPass()
{ return std::make_unique<DFGChannelFanoutExpansionPass>(); }
