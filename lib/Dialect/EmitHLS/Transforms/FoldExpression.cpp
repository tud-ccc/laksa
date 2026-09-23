/// Implementation of FoldExpression transform pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLS.h"
#include "laksa-mlir/Dialect/EmitHLS/IR/EmitHLSOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <llvm/ADT/APInt.h>
#include <llvm/Support/Debug.h>
#include <optional>

#define DEBUG_TYPE "emithls-fold-expression"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[emithls-fold-expression] "; X; llvm::dbgs() << "\n")

using namespace mlir;
using namespace emithls;

namespace mlir {
namespace emithls {
#define GEN_PASS_DEF_EMITHLSFOLDEXPRESSION
#include "laksa-mlir/Dialect/EmitHLS/Transforms/Passes.h.inc"
} // namespace emithls
} // namespace mlir

namespace {

// Bit width used to represent an integer constant of "type". IndexType values
// are always stored as 64-bit APInts by the builders in this file, so it is
// treated the same way here.
static unsigned getIntWidth(Type type)
{
    if (auto intType = dyn_cast<IntegerType>(type)) return intType.getWidth();
    return 64;
}

// Returns the value of "value" if it is defined by a const VariableOp with an
// integer/index initial value.
static std::optional<APInt> getConstantInt(Value value)
{
    auto varOp = value.getDefiningOp<VariableOp>();
    if (!varOp || !varOp.getIsConst()) return std::nullopt;
    auto intAttr = dyn_cast_or_null<IntegerAttr>(varOp.getInitNumberAttr());
    if (!intAttr) return std::nullopt;
    return intAttr.getValue();
}

static Attribute makeIntAttr(OpBuilder &builder, Type type, const APInt &value)
{
    if (isa<IndexType>(type)) return builder.getIndexAttr(value.getSExtValue());
    return builder.getIntegerAttr(cast<IntegerType>(type), value);
}

static CmpPredicate flipPredicate(CmpPredicate predicate)
{
    switch (predicate) {
    case CmpPredicate::lt: return CmpPredicate::gt;
    case CmpPredicate::gt: return CmpPredicate::lt;
    case CmpPredicate::le: return CmpPredicate::ge;
    case CmpPredicate::ge: return CmpPredicate::le;
    case CmpPredicate::eq: return CmpPredicate::eq;
    case CmpPredicate::ne: return CmpPredicate::ne;
    }
    llvm_unreachable("unknown emithls::CmpPredicate");
}

// A value expressed as "base * scale + offset", where "base" is null when the
// whole expression collapses to a compile-time constant.
struct LinearForm {
    Value base;
    APInt scale;
    APInt offset;
    bool folded;
};

// Decomposes "value" into "base * scale + offset" by walking through chains of
// "emithls.arith.add"/"sub"/"mul" that have a constant operand.
static LinearForm resolveLinear(Value value)
{
    if (auto c = getConstantInt(value))
        return LinearForm{Value{}, APInt(c->getBitWidth(), 1), *c, false};

    if (auto addOp = value.getDefiningOp<ArithAddOp>()) {
        LinearForm lhs = resolveLinear(addOp.getLhs());
        LinearForm rhs = resolveLinear(addOp.getRhs());
        if (!lhs.base && !rhs.base)
            return LinearForm{
                Value{},
                lhs.scale,
                lhs.offset + rhs.offset,
                true};
        if (lhs.base && !rhs.base)
            return LinearForm{
                lhs.base,
                lhs.scale,
                lhs.offset + rhs.offset,
                true};
        if (!lhs.base && rhs.base)
            return LinearForm{
                rhs.base,
                rhs.scale,
                rhs.offset + lhs.offset,
                true};
    } else if (auto subOp = value.getDefiningOp<ArithSubOp>()) {
        LinearForm lhs = resolveLinear(subOp.getLhs());
        LinearForm rhs = resolveLinear(subOp.getRhs());
        if (!lhs.base && !rhs.base)
            return LinearForm{
                Value{},
                lhs.scale,
                lhs.offset - rhs.offset,
                true};
        if (lhs.base && !rhs.base)
            return LinearForm{
                lhs.base,
                lhs.scale,
                lhs.offset - rhs.offset,
                true};
        if (!lhs.base && rhs.base)
            return LinearForm{
                rhs.base,
                -rhs.scale,
                lhs.offset - rhs.offset,
                true};
    } else if (auto mulOp = value.getDefiningOp<ArithMulOp>()) {
        LinearForm lhs = resolveLinear(mulOp.getLhs());
        LinearForm rhs = resolveLinear(mulOp.getRhs());
        if (!lhs.base && !rhs.base)
            return LinearForm{
                Value{},
                lhs.scale,
                lhs.offset * rhs.offset,
                true};
        if (!lhs.base && rhs.base && lhs.offset.isOne())
            return LinearForm{rhs.base, rhs.scale, rhs.offset, true};
        if (!lhs.base && rhs.base && lhs.offset.isAllOnes())
            return LinearForm{rhs.base, -rhs.scale, -rhs.offset, true};
        if (!rhs.base && lhs.base && rhs.offset.isOne())
            return LinearForm{lhs.base, lhs.scale, lhs.offset, true};
        if (!rhs.base && lhs.base && rhs.offset.isAllOnes())
            return LinearForm{lhs.base, -lhs.scale, -lhs.offset, true};
    }

    // "value" itself is the base of its own linear form.
    unsigned width = getIntWidth(value.getType());
    return LinearForm{value, APInt(width, 1), APInt(width, 0), false};
}

// Computes the linear form of ”lhs - rhs“
static std::optional<LinearForm>
combineDiff(const LinearForm &lhs, const LinearForm &rhs)
{
    if (!lhs.base && !rhs.base)
        return LinearForm{Value{}, lhs.scale, lhs.offset - rhs.offset, false};
    if (lhs.base && !rhs.base)
        return LinearForm{lhs.base, lhs.scale, lhs.offset - rhs.offset, false};
    if (!lhs.base && rhs.base)
        return LinearForm{rhs.base, -rhs.scale, lhs.offset - rhs.offset, false};

    if (lhs.base != rhs.base) return std::nullopt;
    APInt combinedScale = lhs.scale - rhs.scale;
    if (combinedScale.isZero())
        return LinearForm{
            Value{},
            combinedScale,
            lhs.offset - rhs.offset,
            false};
    if (combinedScale.isOne() || combinedScale.isAllOnes())
        return LinearForm{
            lhs.base,
            combinedScale,
            lhs.offset - rhs.offset,
            false};
    return std::nullopt;
}

static bool evalPredicate(CmpPredicate predicate, const APInt &diff)
{
    APInt zero(diff.getBitWidth(), 0);
    switch (predicate) {
    case CmpPredicate::eq: return diff == zero;
    case CmpPredicate::ne: return diff != zero;
    case CmpPredicate::lt: return diff.slt(zero);
    case CmpPredicate::le: return diff.sle(zero);
    case CmpPredicate::gt: return diff.sgt(zero);
    case CmpPredicate::ge: return diff.sge(zero);
    }
    llvm_unreachable("unknown emithls::CmpPredicate");
}

// Simplifies "emithls.arith.cmp" operands that carry constants added or
// multiplied in, e.g. "(idx * -1) + 30 >= 0" becomes "idx <= 30", and fully
// constant comparisons fold to a boolean.
struct SimplifyCompare : public OpRewritePattern<ArithCmpOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(ArithCmpOp op, PatternRewriter &rewriter) const override
    {
        LinearForm lhs = resolveLinear(op.getLhs());
        LinearForm rhs = resolveLinear(op.getRhs());
        bool bothConst = !lhs.base && !rhs.base;
        // Nothing was absorbed on either side and this isn't a direct
        // constant-vs-constant comparison: the op is already minimal.
        if (!lhs.folded && !rhs.folded && !bothConst) return failure();

        auto diff = combineDiff(lhs, rhs);
        if (!diff) return failure();

        Location loc = op.getLoc();

        if (!diff->base) {
            bool result = evalPredicate(op.getPredicate(), diff->offset);
            LAKSA_DEBUG(
                llvm::dbgs() << "Fold constant compare at " << loc << " to "
                             << (result ? "true" : "false"));
            auto i1Ty = rewriter.getI1Type();
            auto resultVar = VariableOp::create(
                rewriter,
                loc,
                i1Ty,
                rewriter.getIntegerAttr(i1Ty, result ? 1 : 0),
                /*isConst=*/true);
            rewriter.replaceOp(op, resultVar.getVariable());
            return success();
        }

        CmpPredicate newPredicate = op.getPredicate();
        APInt newConst = diff->offset;
        if (diff->scale.isOne())
            newConst = -diff->offset;
        else // diff->scale is -1: multiplying both sides by -1 flips it.
            newPredicate = flipPredicate(newPredicate);

        Type operandType = op.getLhs().getType();
        rewriter.setInsertionPoint(op);
        auto constVar = VariableOp::create(
            rewriter,
            loc,
            operandType,
            makeIntAttr(rewriter, operandType, newConst),
            /*isConst=*/true);
        LAKSA_DEBUG(
            llvm::dbgs()
            << "Simplify compare at " << loc << " to "
            << stringifyCmpPredicate(newPredicate) << " " << newConst);
        rewriter.replaceOpWithNewOp<ArithCmpOp>(
            op,
            op.getType(),
            newPredicate,
            diff->base,
            constVar.getVariable());
        return success();
    }
};

struct EmitHLSFoldExpressionPass
        : public emithls::impl::EmitHLSFoldExpressionBase<
              EmitHLSFoldExpressionPass> {
    void runOnOperation() override
    {
        RewritePatternSet patterns(&getContext());
        patterns.add<SimplifyCompare>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::emithls::createEmitHLSFoldExpressionPass()
{ return std::make_unique<EmitHLSFoldExpressionPass>(); }
