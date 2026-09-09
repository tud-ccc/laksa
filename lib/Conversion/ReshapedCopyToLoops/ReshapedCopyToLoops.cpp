/// Implementation of ReshapedCopyToLoops pass.
///
/// @file
/// @author     Jiahong Bi (jiahong.bi@tu-dresden.de)

#include "laksa-mlir/Conversion/ReshapedCopyToLoops/ReshapedCopyToLoops.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#define DEBUG_TYPE "reshaped-copy-to-loops"
#define LAKSA_DEBUG(X)                                                         \
    LLVM_DEBUG(                                                                \
        llvm::dbgs() << "[reshaped-copy-to-loops] "; X; llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_CONVERTRESHAPEDCOPYTOLOOPS
#include "laksa-mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

// A contiguous memref laid out row-major from offset 0.
static bool isFlat(Type type)
{
    auto memrefTy = dyn_cast<MemRefType>(type);
    return memrefTy && memrefTy.hasStaticShape() && memrefTy.getNumElements() > 0
           && memrefTy.getLayout().isIdentity();
}

// True if castOp only restates a contiguous buffer under another shape.
static bool isPureReshape(memref::ReinterpretCastOp castOp)
{
    auto sourceTy = dyn_cast<MemRefType>(castOp.getSource().getType());
    auto resultTy = dyn_cast<MemRefType>(castOp.getType());
    if (!sourceTy || !resultTy) return false;
    if (!isFlat(sourceTy) || !isFlat(resultTy)) return false;
    if (sourceTy.getElementType() != resultTy.getElementType()) return false;
    if (sourceTy.getNumElements() != resultTy.getNumElements()) return false;
    ArrayRef<int64_t> offsets = castOp.getStaticOffsets();
    return offsets.size() == 1 && offsets.front() == 0;
}

// Where the dimensions of type cut the linear index space. Dimension i spans
// bounds[i] down to bounds[i + 1].
static SmallVector<int64_t> boundsOf(MemRefType type)
{
    SmallVector<int64_t> bounds;
    int64_t rest = type.getNumElements();
    bounds.push_back(rest);
    for (int64_t size : type.getShape()) {
        rest /= size;
        bounds.push_back(rest);
    }
    return bounds;
}

// The coarsest cut of the linear index space that both shapes refine.
static FailureOr<SmallVector<int64_t>>
commonBounds(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs)
{
    SmallVector<int64_t> bounds(lhs);
    bounds.append(rhs.begin(), rhs.end());
    llvm::sort(bounds, std::greater<int64_t>());
    bounds.erase(llvm::unique(bounds), bounds.end());

    for (size_t i = 0; i + 1 < bounds.size(); ++i)
        if (bounds[i] % bounds[i + 1] != 0) return failure();
    return bounds;
}

// Rewrites a copy reaching its source through a reshape into an element loop
// over the unreshaped buffers.
struct ReshapedCopyToLoops : public OpRewritePattern<memref::CopyOp> {
    using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(memref::CopyOp op, PatternRewriter &rewriter) const override
    {
        auto castOp = op.getSource().getDefiningOp<memref::ReinterpretCastOp>();
        if (!castOp || !isPureReshape(castOp)) return failure();

        Value source = castOp.getSource();
        Value target = op.getTarget();
        if (!isFlat(target.getType())) return failure();

        auto sourceTy = cast<MemRefType>(source.getType());
        auto targetTy = cast<MemRefType>(target.getType());
        if (sourceTy.getNumElements() != targetTy.getNumElements())
            return failure();

        SmallVector<int64_t> sourceBounds = boundsOf(sourceTy);
        SmallVector<int64_t> targetBounds = boundsOf(targetTy);
        FailureOr<SmallVector<int64_t>> nestBounds =
            commonBounds(sourceBounds, targetBounds);
        if (failed(nestBounds)) return failure();

        Location loc = op.getLoc();
        LAKSA_DEBUG(
            llvm::dbgs() << "Expanding copy of " << sourceTy << " into "
                         << targetTy << " over " << nestBounds->size() - 1
                         << " loops");

        Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

        OpBuilder::InsertionGuard guard(rewriter);
        SmallVector<Value> inductionVars;
        for (size_t i = 0; i + 1 < nestBounds->size(); ++i) {
            Value extent = arith::ConstantIndexOp::create(
                rewriter,
                loc,
                (*nestBounds)[i] / (*nestBounds)[i + 1]);
            auto loop = scf::ForOp::create(rewriter, loc, zero, extent, one);
            inductionVars.push_back(loop.getInductionVar());
            rewriter.setInsertionPointToStart(loop.getBody());
        }

        // One index per dimension, from the induction variables it spans.
        auto indicesFor = [&](ArrayRef<int64_t> bounds) {
            SmallVector<Value> indices;
            for (size_t dim = 0; dim + 1 < bounds.size(); ++dim) {
                Value index;
                for (size_t i = 0; i + 1 < nestBounds->size(); ++i) {
                    if ((*nestBounds)[i] > bounds[dim]) continue;
                    if ((*nestBounds)[i + 1] < bounds[dim + 1]) continue;
                    Value term = inductionVars[i];
                    int64_t weight = (*nestBounds)[i + 1] / bounds[dim + 1];
                    if (weight != 1) {
                        Value factor = arith::ConstantIndexOp::create(
                            rewriter,
                            loc,
                            weight);
                        term =
                            arith::MulIOp::create(rewriter, loc, term, factor);
                    }
                    index = index
                                ? arith::AddIOp::create(
                                      rewriter,
                                      loc,
                                      index,
                                      term)
                                : term;
                }
                indices.push_back(index ? index : zero);
            }
            return indices;
        };

        Value element = memref::LoadOp::create(
            rewriter,
            loc,
            source,
            indicesFor(sourceBounds));
        memref::StoreOp::create(
            rewriter,
            loc,
            element,
            target,
            indicesFor(targetBounds));

        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertReshapedCopyToLoopsPass
        : public impl::ConvertReshapedCopyToLoopsBase<
              ConvertReshapedCopyToLoopsPass> {
    void runOnOperation() final
    {
        RewritePatternSet patterns(&getContext());
        populateReshapedCopyToLoopsPatterns(patterns);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

void mlir::populateReshapedCopyToLoopsPatterns(RewritePatternSet &patterns)
{ patterns.add<ReshapedCopyToLoops>(patterns.getContext()); }

std::unique_ptr<Pass> mlir::createConvertReshapedCopyToLoopsPass()
{ return std::make_unique<ConvertReshapedCopyToLoopsPass>(); }
