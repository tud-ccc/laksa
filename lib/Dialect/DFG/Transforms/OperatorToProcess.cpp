/// Implementation of OperatorToProcess transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/DFG/IR/DFG.h"
#include "laksa-mlir/Dialect/DFG/IR/DFGOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

#define DEBUG_TYPE "dfg-operator-to-process"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[dfg-operator-to-process] "; X; llvm::dbgs() << "\n")

using namespace mlir;
using namespace mlir::dfg;

namespace mlir {
namespace dfg {
#define GEN_PASS_DEF_DFGOPERATORTOPROCESS
#include "laksa-mlir/Dialect/DFG/Transforms/Passes.h.inc"
} // namespace dfg
} // namespace mlir

namespace {
struct OperatorToEquivalentProcess : public OpRewritePattern<OperatorOp> {
    OperatorToEquivalentProcess(MLIRContext* context)
            : OpRewritePattern<OperatorOp>(context) {};

    LogicalResult
    matchAndRewrite(OperatorOp op, PatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        auto funcType = op.getFunctionType();

        LAKSA_DEBUG(
            llvm::dbgs()
            << "Creating equivalent process based on current operator at "
            << loc);
        auto processOp = ProcessOp::create(
            rewriter,
            loc,
            op.getNodeName(),
            funcType,
            ArrayRef<int64_t>{},
            [&](OpBuilder &processBuilder,
                Location processLoc,
                ValueRange processBlkArgs) {
                IRMapping mapper;
                ValueRange inputPorts =
                    processBlkArgs.take_front(op.getNumInputPorts());
                ValueRange outputPorts =
                    processBlkArgs.drop_front(op.getNumInputPorts());

                LoopOp::create(
                    processBuilder,
                    processLoc,
                    inputPorts,
                    outputPorts,
                    [&](OpBuilder &loopBuilder, Location loopLoc) {
                        // Create pull operations for inputs
                        for (auto [inTy, inPort] :
                             llvm::zip(funcType.getInputs(), inputPorts)) {
                            LAKSA_DEBUG(
                                llvm::dbgs()
                                << "Creating pull op for input port "
                                << cast<BlockArgument>(inPort).getArgNumber()
                                << " with type " << inTy);
                            // Create different pulls for based on input types
                            Value pulledValue =
                                TypeSwitch<Type, Value>(inTy)
                                    .Case<RankedTensorType>([&](auto) {
                                        return PullAsTensorOp::create(
                                            loopBuilder,
                                            loopLoc,
                                            inPort);
                                    })
                                    .Case<MemRefType>([&](auto) {
                                        return PullAsMemRefOp::create(
                                            loopBuilder,
                                            loopLoc,
                                            inPort);
                                    })
                                    .Default([&](auto) {
                                        return PullOp::create(
                                            loopBuilder,
                                            loopLoc,
                                            inPort);
                                    });
                            mapper.map(
                                op.getBody().getArgument(
                                    cast<BlockArgument>(inPort).getArgNumber()),
                                pulledValue);
                        }

                        // Copy the original operations into this block
                        for (auto &opInOperator : op.getBody().getOps()) {
                            if (auto outputOp =
                                    dyn_cast<OutputOp>(opInOperator);
                                outputOp) {
                                // Create push operations based on the value
                                for (auto [outValue, outPort] : llvm::zip(
                                         outputOp.getOperands(),
                                         outputPorts)) {
                                    auto newOutValue = mapper.lookup(outValue);
                                    LAKSA_DEBUG(
                                        llvm::dbgs()
                                        << "Creating push op for output port "
                                        << cast<BlockArgument>(outPort)
                                               .getArgNumber()
                                        << " with type "
                                        << newOutValue.getType());
                                    TypeSwitch<Type>(newOutValue.getType())
                                        .Case<RankedTensorType>([&](auto) {
                                            PushTensorOp::create(
                                                loopBuilder,
                                                loopLoc,
                                                newOutValue,
                                                outPort);
                                        })
                                        .Case<MemRefType>([&](auto) {
                                            PushMemRefOp::create(
                                                loopBuilder,
                                                loopLoc,
                                                newOutValue,
                                                outPort);
                                        })
                                        .Default([&](auto) {
                                            PushOp::create(
                                                loopBuilder,
                                                loopLoc,
                                                newOutValue,
                                                outPort);
                                        });
                                }
                            } else {
                                LAKSA_DEBUG(
                                    llvm::dbgs()
                                    << "Cloning op: "
                                    << opInOperator.getName().getStringRef());
                                loopBuilder.clone(opInOperator, mapper);
                            }
                        }
                    });
            });

        LAKSA_DEBUG(
            llvm::dbgs() << "Replacing OperatorOp '" << op.getNodeName()
                         << "' with ProcessOp");
        rewriter.replaceOp(op, processOp);

        return success();
    }
};
} // namespace

namespace {
struct DFGOperatorToProcessPass
        : public dfg::impl::DFGOperatorToProcessBase<DFGOperatorToProcessPass> {
    void runOnOperation() override
    {
        ConversionTarget target(getContext());
        RewritePatternSet patterns(&getContext());

        patterns.add<OperatorToEquivalentProcess>(patterns.getContext());

        target.addLegalDialect<DFGDialect>();
        target.addIllegalOp<OperatorOp>();
        target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

        if (failed(applyPartialConversion(
                getOperation(),
                target,
                std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // namespace

std::unique_ptr<Pass> mlir::dfg::createDFGOperatorToProcessPass()
{ return std::make_unique<DFGOperatorToProcessPass>(); }
