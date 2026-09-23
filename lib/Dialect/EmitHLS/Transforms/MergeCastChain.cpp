/// Implementation of MergeCastChain transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "emithls-merge-cast-chain"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[emithls-merge-cast-chain] "; X;                      \
        llvm::dbgs() << "\n")

using namespace mlir;
using namespace emithls;

namespace mlir {
namespace emithls {
#define GEN_PASS_DEF_EMITHLSMERGECASTCHAIN
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h.inc"
} // namespace emithls
} // namespace mlir

namespace {
struct ReplaceCastValueToSrc : public OpRewritePattern<ArithCastOp> {
    ReplaceCastValueToSrc(MLIRContext* context)
            : OpRewritePattern<ArithCastOp>(context) {};

    LogicalResult
    matchAndRewrite(ArithCastOp op, PatternRewriter &) const override
    {
        auto operand = op.getOperand();
        auto operandDefiningCastOp = cast<ArithCastOp>(operand.getDefiningOp());
        auto operandDefiningCastOperand = operandDefiningCastOp.getOperand();
        op->replaceUsesOfWith(operand, operandDefiningCastOperand);
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Replace operand of chained cast operation at " << op.getLoc()
            << " with the value from the source cast at "
            << operandDefiningCastOp.getLoc());
        return success();
    }
};
} // namespace

namespace {
struct EmitHLSMergeCastChainPass
        : public emithls::impl::EmitHLSMergeCastChainBase<
              EmitHLSMergeCastChainPass> {
    void runOnOperation() override;
};
} // namespace

void EmitHLSMergeCastChainPass::runOnOperation()
{
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    patterns.add<ReplaceCastValueToSrc>(&getContext());

    target.addLegalDialect<EmitHLSDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    target.addDynamicallyLegalOp<ArithCastOp>([](ArithCastOp castOp) {
        if (isa<ArithCastOp>(castOp.getOperand().getDefiningOp())) return false;
        return true;
    });

    if (failed(applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> mlir::emithls::createEmitHLSMergeCastChainPass()
{ return std::make_unique<EmitHLSMergeCastChainPass>(); }
