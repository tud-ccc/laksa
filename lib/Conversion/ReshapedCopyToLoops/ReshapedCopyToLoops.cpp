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
#include <optional>

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
    return memrefTy && memrefTy.hasStaticShape()
           && memrefTy.getNumElements() > 0
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

// Row-major strides of a contiguous buffer with the given shape.
static SmallVector<int64_t> contiguousStrides(ArrayRef<int64_t> shape)
{
    SmallVector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (size_t dim = shape.size(); dim-- > 0;) {
        strides[dim] = stride;
        stride *= shape[dim];
    }
    return strides;
}

// A block of a flat buffer, starting at starts along each dimension.
struct Window {
    Value buffer;
    SmallVector<int64_t> starts;
    SmallVector<int64_t> steps;
};

static bool hasDynamicValue(ArrayRef<int64_t> values)
{
    return llvm::any_of(
        values,
        [](int64_t value) { return ShapedType::isDynamic(value); });
}

static bool isInBounds(const Window &window, ArrayRef<int64_t> shape)
{
    auto bufferType = cast<MemRefType>(window.buffer.getType());
    for (auto [start, step, size, bufferSize] : llvm::zip_equal(
             window.starts,
             window.steps,
             shape,
             bufferType.getShape())) {
        if (start < 0 || step <= 0 || size <= 0) return false;
        if (start + (size - 1) * step >= bufferSize) return false;
    }
    return true;
}

// The block of a flat buffer that value addresses, either the whole buffer, a
// static same-rank subview, or a same-rank reinterpret_cast of it.
static std::optional<Window> windowOf(Value value)
{
    auto valueTy = dyn_cast<MemRefType>(value.getType());
    if (!valueTy || !valueTy.hasStaticShape()) return std::nullopt;

    if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
        auto sourceTy = dyn_cast<MemRefType>(subview.getSource().getType());
        if (!sourceTy || sourceTy.getRank() != valueTy.getRank())
            return std::nullopt;

        ArrayRef<int64_t> offsets = subview.getStaticOffsets();
        ArrayRef<int64_t> sizes = subview.getStaticSizes();
        ArrayRef<int64_t> strides = subview.getStaticStrides();
        if (hasDynamicValue(offsets) || hasDynamicValue(sizes)
            || hasDynamicValue(strides) || sizes != valueTy.getShape())
            return std::nullopt;

        std::optional<Window> source = windowOf(subview.getSource());
        if (!source) return std::nullopt;

        Window window{source->buffer, {}, {}};
        for (auto [sourceStart, sourceStep, offset, stride] : llvm::zip_equal(
                 source->starts,
                 source->steps,
                 offsets,
                 strides)) {
            window.starts.push_back(sourceStart + offset * sourceStep);
            window.steps.push_back(sourceStep * stride);
        }
        return isInBounds(window, sizes) ? std::optional<Window>(window)
                                         : std::nullopt;
    }

    if (isFlat(valueTy)) {
        return Window{
            value,
            SmallVector<int64_t>(valueTy.getRank(), 0),
            SmallVector<int64_t>(valueTy.getRank(), 1)};
    }

    auto castOp = value.getDefiningOp<memref::ReinterpretCastOp>();
    if (!castOp) return std::nullopt;
    auto bufferTy = dyn_cast<MemRefType>(castOp.getSource().getType());
    if (!bufferTy || !isFlat(bufferTy)) return std::nullopt;
    if (bufferTy.getElementType() != valueTy.getElementType()
        || bufferTy.getRank() != valueTy.getRank())
        return std::nullopt;

    ArrayRef<int64_t> offsets = castOp.getStaticOffsets();
    SmallVector<int64_t> strides = contiguousStrides(bufferTy.getShape());
    if (offsets.size() != 1 || offsets.front() < 0
        || castOp.getStaticSizes() != valueTy.getShape()
        || castOp.getStaticStrides() != ArrayRef<int64_t>(strides))
        return std::nullopt;

    Window window{castOp.getSource(), {}, {}};
    int64_t rest = offsets.front();
    for (auto [dim, stride] : llvm::enumerate(strides)) {
        window.starts.push_back(rest / stride);
        window.steps.push_back(1);
        rest %= stride;
        if (window.starts.back() + valueTy.getDimSize(dim)
            > bufferTy.getDimSize(dim))
            return std::nullopt;
    }
    return window;
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
            llvm::dbgs()
            << "Expanding copy of " << sourceTy << " into " << targetTy
            << " over " << nestBounds->size() - 1 << " loops");

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
                    index =
                        index
                            ? arith::AddIOp::create(rewriter, loc, index, term)
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

// Rewrites a copy between blocks of flat buffers into an element loop over the
// buffers.
struct WindowedCopyToLoops : public OpRewritePattern<memref::CopyOp> {
    using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(memref::CopyOp op, PatternRewriter &rewriter) const override
    {
        std::optional<Window> source = windowOf(op.getSource());
        std::optional<Window> target = windowOf(op.getTarget());
        if (!source || !target) return failure();
        if (source->buffer == op.getSource()
            && target->buffer == op.getTarget())
            return failure();

        auto sourceTy = cast<MemRefType>(op.getSource().getType());
        auto targetTy = cast<MemRefType>(op.getTarget().getType());
        if (sourceTy.getShape() != targetTy.getShape()) return failure();

        Location loc = op.getLoc();
        LAKSA_DEBUG(
            llvm::dbgs() << "Expanding windowed copy of " << sourceTy
                         << " into " << targetTy);

        Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

        OpBuilder::InsertionGuard guard(rewriter);
        SmallVector<Value> inductionVars;
        for (int64_t size : sourceTy.getShape()) {
            if (size == 1) {
                inductionVars.push_back(nullptr);
                continue;
            }
            Value extent = arith::ConstantIndexOp::create(rewriter, loc, size);
            auto loop = scf::ForOp::create(rewriter, loc, zero, extent, one);
            inductionVars.push_back(loop.getInductionVar());
            rewriter.setInsertionPointToStart(loop.getBody());
        }

        auto indicesFor = [&](const Window &window) {
            SmallVector<Value> indices;
            for (auto [start, step, inductionVar] : llvm::zip_equal(
                     window.starts,
                     window.steps,
                     inductionVars)) {
                Value index = inductionVar;
                if (index && step != 1) {
                    Value factor =
                        arith::ConstantIndexOp::create(rewriter, loc, step);
                    index = arith::MulIOp::create(rewriter, loc, index, factor);
                }
                if (!index) {
                    index = arith::ConstantIndexOp::create(rewriter, loc, start);
                } else if (start != 0) {
                    Value offset =
                        arith::ConstantIndexOp::create(rewriter, loc, start);
                    index = arith::AddIOp::create(rewriter, loc, index, offset);
                }
                indices.push_back(index);
            }
            return indices;
        };

        Value element = memref::LoadOp::create(
            rewriter,
            loc,
            source->buffer,
            indicesFor(*source));
        memref::StoreOp::create(
            rewriter,
            loc,
            element,
            target->buffer,
            indicesFor(*target));

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
{
    patterns.add<ReshapedCopyToLoops, WindowedCopyToLoops>(
        patterns.getContext());
}

std::unique_ptr<Pass> mlir::createConvertReshapedCopyToLoopsPass()
{ return std::make_unique<ConvertReshapedCopyToLoopsPass>(); }
