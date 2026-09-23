/// Implementation of FuseOperator transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "emithls-fuse-operator"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[emithls-fuse-operator] "; X; llvm::dbgs() << "\n")

using namespace mlir;
using namespace emithls;

namespace mlir {
namespace emithls {
#define GEN_PASS_DEF_EMITHLSFUSEOPERATOR
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h.inc"
} // namespace emithls
} // namespace mlir

namespace {

template<typename ArithOp, FusedOperator OpCode>
struct FuseAssignPattern : public OpRewritePattern<UpdateOp> {
    using OpRewritePattern::OpRewritePattern;

    /// If @p val was new of the most recent `UpdateOp` before @p before in the
    /// same block, return that update's variable; otherwise return @p val.
    /// Intervening writes to the same variable invalidate the proxy.
    static Value resolveToVariable(Value value, Operation* before)
    {
        Block* block = before->getBlock();
        llvm::SmallPtrSet<Value, 4> overwritten;

        for (auto it = before->getIterator(); it != block->begin();) {
            --it;
            auto updateOp = dyn_cast<UpdateOp>(&*it);
            if (!updateOp || !updateOp.getIndices().empty()) continue;
            Value updateVar = updateOp.getVariable();
            if (updateOp.getNewValue() == value) {
                if (overwritten.contains(updateVar)) {
                    LAKSA_DEBUG(
                        llvm::dbgs()
                        << "Resolve: " << value << " -> stale proxy for "
                        << updateVar << " (overwritten since update)");
                    return value;
                }
                LAKSA_DEBUG(
                    llvm::dbgs() << "Resolve: " << value << " -> " << updateVar
                                 << " (via update)");
                return updateVar;
            }
            overwritten.insert(updateVar);
        }
        LAKSA_DEBUG(
            llvm::dbgs() << "Resolve: " << value << " -> no proxy found");
        return value;
    }

    LogicalResult
    matchAndRewrite(UpdateOp updateOp, PatternRewriter &rewriter) const override
    {
        if (!updateOp.getIndices().empty()) return failure();

        auto arithOp = updateOp.getNewValue().getDefiningOp<ArithOp>();
        if (!arithOp) return failure();

        // Resolve each operand: if it was the last value written into some
        // variable (with no intervening write to that variable), treat it as
        // that variable so we can recognise e.g. var0 *= var1 even when the
        // operands are SSA temporaries from earlier arith ops.
        Value variable = updateOp.getVariable();
        Operation* mulPoint = arithOp.getOperation();
        Value lhs = resolveToVariable(arithOp.getLhs(), mulPoint);
        Value rhs = resolveToVariable(arithOp.getRhs(), mulPoint);

        Value val;
        if (lhs == variable)
            val = rhs;
        else if (rhs == variable)
            val = lhs;
        else {
            LAKSA_DEBUG(
                llvm::dbgs() << "Skip update of " << variable
                             << ": neither operand resolves to it");
            return failure();
        }

        // Check before any erasure; if arithOp's only use is this update, it
        // becomes dead after we erase the update and can be removed too.
        bool arithOnlyUsedHere = arithOp->hasOneUse();
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Fusing " << arithOp->getName() << " + update of " << variable
            << " into arith.fused " << stringifyFusedOperator(OpCode));
        ArithFusedOp::create(
            rewriter,
            updateOp.getLoc(),
            OpCode,
            variable,
            val);
        rewriter.eraseOp(updateOp);
        if (arithOnlyUsedHere) {
            LAKSA_DEBUG(
                llvm::dbgs() << "  Erasing dead " << arithOp->getName());
            rewriter.eraseOp(arithOp);
        }
        return success();
    }
};

using FuseAddAssign = FuseAssignPattern<ArithAddOp, FusedOperator::add>;
using FuseSubAssign = FuseAssignPattern<ArithSubOp, FusedOperator::sub>;
using FuseMulAssign = FuseAssignPattern<ArithMulOp, FusedOperator::mul>;

struct EmitHLSFuseOperatorPass : public emithls::impl::EmitHLSFuseOperatorBase<
                                     EmitHLSFuseOperatorPass> {
    void runOnOperation() override
    {
        RewritePatternSet patterns(&getContext());
        patterns.add<FuseAddAssign, FuseSubAssign, FuseMulAssign>(
            &getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::emithls::createEmitHLSFuseOperatorPass()
{ return std::make_unique<EmitHLSFuseOperatorPass>(); }
